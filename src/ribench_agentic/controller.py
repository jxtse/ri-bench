"""Serial, resumable plan execution with per-episode failure isolation."""

import json
import os
from pathlib import Path
import time
import traceback

from .batch import _context_core, episodes, load_plan, reconcile_ledger, run_episode
from .casefs import write_json
from .extract import extract_all
from .pi_runner import TOKEN_FIELDS
from .summary import summarize


def run_controller(plan, *, runs_root="runs/agentic", max_episodes=None,
                   concurrency=1, resume=False, pi_binary="pi", order="forward", progress_name="progress.json"):
    if concurrency != 1 or (max_episodes is not None and max_episodes < 0):
        raise ValueError("Controller requires concurrency=1 and a nonnegative episode limit")
    plan = load_plan(plan)
    entries = episodes(plan, runs_root)
    # Several single-threaded controllers may work the same batch (per-episode flock makes each episode run once);
    # giving each a different traversal order keeps them from queueing on the same episode. 2026-09-13.
    if order == "reverse":
        entries = list(reversed(entries))
    elif order == "middle":
        half = len(entries) // 2
        entries = entries[half:] + entries[:half]
    elif order.startswith("shard:"):
        # shard:i/n keeps every n-th entry starting at i, so n controllers split a batch without queueing (2026-09-25)
        index, count = (int(x) for x in order[len("shard:"):].split("/"))
        if not 0 <= index < count:
            raise ValueError("shard index must satisfy 0 <= i < n")
        entries = entries[index::count]
    elif order != "forward":
        raise ValueError("order must be forward, reverse, middle or shard:i/n")
    batch_root = Path(runs_root).resolve() / plan["batch"]
    batch_root.mkdir(parents=True, exist_ok=True)
    ledger = batch_root / "ledger.jsonl"
    started = time.monotonic()
    progress = {"batch": plan["batch"], "total_episodes": len(entries), "attempted": 0,
                "skipped": 0, "counts": {}, "last_episode_dir": None, "stopped": False}

    def save_progress():
        models = summarize(ledger) if ledger.exists() else {}
        missing = {key: sum(model["missing_usage"][key] for model in models.values()) for key in TOKEN_FIELDS}
        known = {key: sum(model["known_usage"][key] for model in models.values()) for key in TOKEN_FIELDS}
        progress.update(elapsed_seconds=time.monotonic() - started,
                        token_totals={key: None if missing[key] else known[key] for key in TOKEN_FIELDS},
                        known_token_totals=known, missing_usage=missing)
        write_json(batch_root / progress_name, progress)

    for entry in entries:
        if (batch_root / "STOP").exists():
            progress["stopped"] = True
            break
        if max_episodes is not None and progress["attempted"] >= max_episodes:
            break
        destination, context = entry[1:]
        skipped = False
        failure_status = "error"
        try:
            recorded = destination / "episode.json"
            if resume and recorded.exists() and json.loads(recorded.read_text())["status"] == "complete":
                existing = json.loads((destination / "context.json").read_text())
                if _context_core(existing) != _context_core(context):
                    raise ValueError("Existing episode context differs; use a new batch")
                reconcile_ledger(destination, context, ledger)
                extract_all(destination)
                status, skipped = "complete", True
            else:
                result = run_episode(*entry, ledger, False, pi_binary)
                status = result["status"]
                if status == "skipped":
                    status = json.loads(recorded.read_text())["status"]
            if status != "complete":
                failure_status = status
                raise RuntimeError(f"Episode returned status {status}")
        except Exception:
            failure_traceback = traceback.format_exc()
            status = failure_status
            with (batch_root / "failures.jsonl").open("a", encoding="utf-8") as failures:
                failures.write(json.dumps({"episode_dir": str(destination), "status": status,
                                           "traceback": failure_traceback}, ensure_ascii=False) + "\n")
                failures.flush()
                os.fsync(failures.fileno())
        progress["skipped" if skipped else "attempted"] += 1
        progress["counts"][status] = progress["counts"].get(status, 0) + 1
        progress["last_episode_dir"] = str(destination)
        save_progress()
        print(f"{'skipped' if skipped else status} {destination}", flush=True)
    save_progress()
    return 2 if any(status != "complete" for status in progress["counts"]) else 0
