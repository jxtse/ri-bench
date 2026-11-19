#!/usr/bin/env python3
"""follow-up subject runs: plan writer + lane runner for the OpenReview designs.

  rebuttal_run.py plan                            # write configs/agentic/plans/rebuttal/<scenario>__<lane>.json
  rebuttal_run.py lane <scenario> <lane> [--shard i/n] [--kill-after N]
  rebuttal_run.py status                          # per scenario x model completion and failure counts

Each lane is one provider/model with its own batch (rebuttal-<scenario>-<lane>-20261107), so batches stay
single-provider as in the paper. A lane may run several controllers on disjoint shards (controller order shard:i/n).
Kill-check (first N episodes of shard 0): decision extraction >= 0.95; in designs with a control arm C, the
program chair follows the specific, correct majority in >= 0.70 of C episodes. Failure writes STOP and halts the lane.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv/bin/python"
PLANS = REPO / "configs/agentic/plans/rebuttal"
QA = REPO / "build/qa/rebuttal-20261107"
FROZEN = json.load(open(REPO / "configs/agentic/cases_frozen_v3.json"))
E3 = json.load(open(REPO / "configs/agentic/rebuttal_e3_cases.json"))
LANES = {  # lane -> model id (all eight paper subjects; DeepSeek-V4.1-Flash through MinT)
    "opus": "foundry-a/claude-opus-5", "sol": "foundry-b/gpt-5.6-sol", "luna": "foundry-b/gpt-5.6-luna",
    "terra": "foundry-b/gpt-5.6-terra", "glm": "mint/glm-5.3", "glmflash": "ark/glm-5.3-flash",
    "qwen": "openrouter/qwen/qwen3.5-35b-a3b", "deepseek": "mint/deepseek-flash",
    # Qwen3.5-35B-A3B follow-up designs run on its open weights with vLLM. The partial
    # OpenRouter `qwen` batches are kept as a serving-route comparison, not pooled.
    "qwen-vllm": "vllm/qwen3.5-35b-a3b",
}
DESIGNS = {  # scenario -> (arms, cases, lanes)
    "or_r": (["C", "X"], FROZEN["good"] + FROZEN["bad"], list(LANES)),
    "or_e1": (["P"], FROZEN["good"] + FROZEN["bad"], list(LANES)),
    "or_e2a": (["C", "X"], FROZEN["good"] + FROZEN["bad"], list(LANES)),
    "or_e2b": (["SP", "SM"], FROZEN["good"] + FROZEN["bad"], [l for l in LANES if l != "qwen"]),
    "or_e3a": (["X"], E3["cases"], ["opus", "sol", "glm", "qwen-vllm"]),
    "or_e3b": (["X"], E3["cases"], ["opus", "sol", "glm", "qwen-vllm"]),
    "or_e3c": (["X"], E3["cases"], ["opus", "sol", "glm", "qwen-vllm"]),
}


def batch_name(scenario, lane):
    return f"rebuttal-{scenario.replace('_', '-')}-{lane}-20261107"


def plan_path(scenario, lane):
    return PLANS / f"{scenario}__{lane}.json"


def cmd_plan(_):
    PLANS.mkdir(parents=True, exist_ok=True)
    n = 0
    for scenario, (arms, cases, lanes) in DESIGNS.items():
        for lane in lanes:
            plan = {"batch": batch_name(scenario, lane), "scenario": scenario, "cases": cases, "arms": arms,
                    "conditions": ["obligation", "vanilla"], "models": [LANES[lane]], "thinking": "medium",
                    "timeout_s": 1200, "pi_agent_dir": "configs/agentic/pi-agent"}
            plan_path(scenario, lane).write_text(json.dumps(plan, indent=2) + "\n")
            n += 1
    print(f"wrote {n} plans to {PLANS}")


def load_rows(batch_root):
    rows = []
    for ep in sorted((batch_root / "episodes").glob("*")):
        try:
            ctx = json.loads((ep / "context.json").read_text())
            ex = json.loads((ep / "extraction.json").read_text()) if (ep / "extraction.json").exists() else None
            st = json.loads((ep / "episode.json").read_text())["status"] if (ep / "episode.json").exists() else None
        except Exception:  # noqa: BLE001
            continue
        rows.append((ctx, ex, st))
    return rows


def kill_check(batch_root):
    rows = [r for r in load_rows(batch_root) if r[2] is not None]
    done = [r for r in rows if r[2] == "complete" and r[1]]
    extracted = sum(r[1]["final"].get("decision") in ("accept", "reject") for r in done)
    rate = extracted / max(1, len(rows))
    control = [r for r in done if r[0]["arm"] == "C" and r[1]["final"].get("decision") in ("accept", "reject")]
    follow = sum(r[1]["final"].get("majority_alignment") is True for r in control) / max(1, len(control))
    ok = rate >= 0.95 and (len(control) < 8 or follow >= 0.70)
    return ok, {"episodes": len(rows), "complete": len(done), "extraction": round(rate, 3), "control_n": len(control),
                "control_follow": round(follow, 3)}


def controller(plan, shard, max_episodes=None, log=None):
    cmd = [str(PY), "-m", "ribench_agentic.cli", "controller", "--plan", str(plan), "--runs-root", "runs/agentic",
           "--concurrency", "1", "--resume", "--order", f"shard:{shard}" if shard else "forward"]
    if max_episodes:
        cmd += ["--max-episodes", str(max_episodes)]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    return subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def cmd_lane(a):
    plan = plan_path(a.scenario, a.lane)
    batch_root = REPO / "runs/agentic" / batch_name(a.scenario, a.lane)
    shard = a.shard or ""
    QA.mkdir(parents=True, exist_ok=True)
    log = open(QA / f"lane_{a.scenario}_{a.lane}_{shard.replace('/', 'of') or 'all'}.log", "a")
    if a.kill_after:
        rc = controller(plan, shard, a.kill_after, log)
        ok, info = kill_check(batch_root)
        print(f"{a.scenario}/{a.lane} kill-check {'PASS' if ok else 'FAIL'} {info} (controller rc={rc})", flush=True)
        if not ok:
            batch_root.mkdir(parents=True, exist_ok=True)
            (batch_root / "STOP").write_text(f"kill-check failed {info}\n")
            return 3
    rc = controller(plan, shard, None, log)
    print(f"{a.scenario}/{a.lane} shard={shard or 'all'} controller exit {rc}", flush=True)
    return rc


def cmd_status(_):
    table = collections.defaultdict(dict)
    for scenario, (arms, cases, lanes) in DESIGNS.items():
        for lane in lanes:
            root = REPO / "runs/agentic" / batch_name(scenario, lane)
            total = len(arms) * len(cases) * 2
            if not root.exists():
                table[scenario][lane] = f"0/{total}"
                continue
            rows = load_rows(root)
            st = collections.Counter(r[2] for r in rows)
            table[scenario][lane] = f"{st.get('complete', 0)}/{total}" + (f" err{st.get('error')}" if st.get("error") else "") + \
                (" STOP" if (root / "STOP").exists() else "")
    print(json.dumps(table, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    lane = sub.add_parser("lane"); lane.add_argument("scenario"); lane.add_argument("lane")
    lane.add_argument("--shard"); lane.add_argument("--kill-after", type=int)
    sub.add_parser("status")
    a = p.parse_args()
    return {"plan": cmd_plan, "lane": cmd_lane, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
