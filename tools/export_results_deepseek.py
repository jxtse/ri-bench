#!/usr/bin/env python3
"""Export the DeepSeek V4.1 Flash supplementary run (official API, 2026-09-24) into results/supplement/deepseek/.

Same row schema and filters as tools/export_results.py (which stays frozen at the seven-model plan counts), but
the four DeepSeek lanes are exported into a separate directory with their own RUNS_INDEX.json. The frozen
seven-model files in results/*.jsonl are never touched. Unlike the frozen exporter, a shortfall against the
planned counts does not abort: it is recorded per scenario (planned, exported, excluded by reason) so that the
paper can disclose it. Deterministic; no model calls.
Usage: .venv/bin/python tools/export_results_deepseek.py
"""
import glob, hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_results as base  # noqa: E402

R = base.R
RUNS = base.RUNS
OUT = os.path.join(R, "results", "supplement", "deepseek")
MODEL = "deepseek/deepseek-flash"
BATCHES = {
    "court": "full-court-deepseek-20260911",
    "open_review_v2": "full-openreview-deepseek-20260911",
    "open_review_v3": "full-openreview-v3-deepseek-20260913",
    "parliament": "full-parliament-deepseek-20260911",
}
PLANNED = {"court": 480, "open_review_v2": 476, "open_review_v3": 472, "parliament": 240}


def classify_exclusion(ed, scenario, roles):
    """Why an episode directory is not exported: workspace escape, empty output, guard, or missing role."""
    single = len(base.ROLES[scenario]) == 1
    reasons = []
    for role in base.ROLES[scenario]:
        p = os.path.join(ed if single else os.path.join(ed, role), "episode.json")
        if not os.path.exists(p):
            # a downstream role never ran because an upstream role failed, or the guard refused to build its
            # workspace (Open Review: AC text contained the historical decision string)
            if not reasons:
                reasons.append("guard_blocked")
            continue
        e = base.read(p)
        if e.get("status") == "complete":
            continue
        errs = " ".join(str(x) for x in (e.get("errors") or [])) + " " + str(e.get("integrity_error") or "")
        if "outside case workspace" in errs:
            reasons.append("workspace_escape")
        elif "outside read-only allowlist" in errs:
            reasons.append("non_readonly_tool")
        elif "empty" in errs.lower() or "no output" in errs.lower():
            reasons.append("empty_output")
        else:
            reasons.append("other_error")
    return ",".join(sorted(set(reasons))) or "unknown"


def export(scenario, bname):
    rows, excluded = [], {}
    b = os.path.join(RUNS, bname)
    for ed in sorted(glob.glob(os.path.join(b, "episodes", "*"))):
        if not os.path.exists(ed + "/context.json"):
            continue
        c = base.read(ed + "/context.json")
        if c["model"] != MODEL:
            continue
        single = len(base.ROLES[scenario]) == 1
        roles = {r: base.role_stats(ed, r, single, scenario, c["case_id"], c.get("arm")) for r in base.ROLES[scenario]}
        ok = os.path.exists(ed + "/extraction.json") and all(v is not None and v["status"] == "complete" for v in roles.values())
        if not ok:
            k = classify_exclusion(ed, scenario, roles)
            excluded[k] = excluded.get(k, 0) + 1
            continue
        x = base.read(ed + "/extraction.json")
        row = {
            "scenario": scenario,
            "episode_id": os.path.basename(ed),
            "batch": bname,
            "case_id": c["case_id"],
            "arm": c.get("arm"),
            "condition": c["condition"],
            "model": "deepseek-v4.1-flash",
            "model_requested": c["model"],
            "final": x["final"],
            "pattern_label": x.get("regex") or x.get("classification"),
            "scorer": base.slim_scorer(x.get("scorer")),
            "agreement": x.get("agreement"),
            "episode_status": x.get("episode_status", "complete"),
            "roles": roles,
        }
        if scenario == "open_review_v2":
            row["final"]["iclr_decision"] = base.ICLR.get(c["case_id"])
        rows.append(row)
    rows.sort(key=lambda r: (r["model"], r["condition"], r["arm"] or "", r["case_id"]))
    return rows, excluded


def main():
    os.makedirs(OUT, exist_ok=True)
    index = {"model": "deepseek-v4.1-flash", "model_requested": MODEL, "note":
             "Supplementary run after the seven-model tables were frozen; official DeepSeek API, medium effort, same "
             "frozen plans, same pi harness and scoring pipeline. Not part of results/*.jsonl."}
    for scen, bname in BATCHES.items():
        rows, excluded = export(scen, bname)
        path = os.path.join(OUT, f"{scen}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        index[scen] = {"file": os.path.basename(path), "batch": bname, "planned": PLANNED[scen], "rows": len(rows),
                       "excluded": excluded, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
        print(f"{scen:16s} planned {PLANNED[scen]:4d}  exported {len(rows):4d}  excluded {excluded}")
        if len(rows) + sum(excluded.values()) != PLANNED[scen]:
            print(f"  WARNING: exported+excluded != planned for {scen}", file=sys.stderr)
    with open(os.path.join(OUT, "RUNS_INDEX.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
