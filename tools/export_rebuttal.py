#!/usr/bin/env python3
"""Export the follow-up OpenReview runs (prereg docs/prereg/REBUTTAL_OPENREVIEW_DESIGNS.md) into results/rebuttal/.

One row per completed episode (all role statuses complete), same row schema as tools/export_results.py. Qwen rows come
from the local-vLLM lane `qwen-vllm`; the partial OpenRouter `qwen` batches are a serving-route control and are
not exported. Shortfalls against the planned counts do not abort: they are recorded per scenario and model, with every
non-complete episode listed by reason, so the rebuttal can disclose them. Archived failed attempts (retried_episodes/)
are counted per batch. Frozen results/*.jsonl are never touched. Deterministic; no model calls.
Usage: .venv/bin/python tools/export_rebuttal.py
"""
import glob, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_results as base  # noqa: E402
import rebuttal_run as rr  # noqa: E402

OUT = os.path.join(base.R, "results", "rebuttal")
ROLES = ["area_chair", "program_chair"]
LANES = [l for l in rr.LANES if l != "qwen"]


def episode_errors(ed):
    out = {}
    for r in ROLES:
        p = os.path.join(ed, r, "episode.json")
        if os.path.exists(p):
            e = base.read(p)
            if e.get("status") != "complete":
                out[r] = {"status": e.get("status"), "errors": sorted(set(map(str, e.get("errors") or [])))}
        elif os.path.exists(os.path.join(ed, r, "invocation.json")):
            out[r] = {"status": "interrupted"}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    index = {"source": "runs/agentic/rebuttal-<scenario>-<lane>-20261107", "excluded_lanes": {"qwen": "OpenRouter route, superseded by qwen-vllm"},
             "scenarios": {}}
    for scen, (arms, cases, lanes) in rr.DESIGNS.items():
        rows, info = [], {}
        for lane in [l for l in lanes if l in LANES]:
            b = os.path.join(base.RUNS, rr.batch_name(scen, lane))
            planned = len(arms) * len(cases) * 2
            got, missing = 0, []
            for ed in sorted(glob.glob(os.path.join(b, "episodes", "*"))):
                c = base.read(ed + "/context.json")
                roles = {r: base.role_stats(ed, r, False, "open_review_rebuttal", c["case_id"], c.get("arm")) for r in ROLES}
                if not os.path.exists(ed + "/extraction.json") or any(v is None or v["status"] != "complete" for v in roles.values()):
                    missing.append({"episode_id": os.path.basename(ed), "case_id": c["case_id"], "arm": c.get("arm"),
                                    "condition": c["condition"], "roles": episode_errors(ed)})
                    continue
                x = base.read(ed + "/extraction.json")
                rows.append({"scenario": scen, "episode_id": os.path.basename(ed), "batch": os.path.basename(b),
                             "lane": lane, "case_id": c["case_id"], "arm": c.get("arm"), "condition": c["condition"],
                             "model": base.short_model(c["model"]), "model_requested": c["model"], "final": x["final"],
                             "pattern_label": x.get("regex"), "episode_status": x.get("episode_status", "complete"),
                             "roles": roles})
                got += 1
            archived = len(glob.glob(os.path.join(b, "retried_episodes", "*")))
            info[lane] = {"planned": planned, "exported": got, "not_complete": missing, "archived_failed_attempts": archived}
        rows.sort(key=lambda r: (r["model"], r["condition"], r["arm"] or "", r["case_id"]))
        path = os.path.join(OUT, f"{scen}.jsonl")
        with open(path + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(path + ".tmp", path)
        index["scenarios"][scen] = {"file": os.path.basename(path), "rows": len(rows),
                                    "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "lanes": info}
        short = {l: f"{v['exported']}/{v['planned']}" + (f" (-{len(v['not_complete'])})" if v["not_complete"] else "") for l, v in info.items()}
        print(f"{scen:8s} {len(rows):5d} rows  {short}")
    with open(os.path.join(OUT, "RUNS_INDEX.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True, ensure_ascii=False)


if __name__ == "__main__":
    main()
