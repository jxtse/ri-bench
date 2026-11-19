#!/usr/bin/env python3
"""Export the frozen per-episode results of the three RI-Bench scenarios from runs/agentic into results/.

One row per completed episode (all role statuses complete, model identity matches the plan, not a retired
or duplicate lane). Columns are documented in results/README.md. Also writes results/summary.json with the
numbers used in the manuscript tables and figures, and results/RUNS_INDEX.json mapping each row back to
its source batch. Deterministic; no model calls. Usage: .venv/bin/python tools/export_results.py
"""
import glob, hashlib, json, os, sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(R, "runs", "agentic")
OUT = os.path.join(R, "results")
SOL_DUP = "foundry-a/gpt-5.6-terra"  # route foundry-a served Sol under the Terra name; excluded everywhere
SCENARIOS = {
    "court": ("full-court-", None),
    "open_review_v2": ("full-openreview-", "v3"),  # prefix, exclude substring
    "open_review_v3": ("full-openreview-v3-", None),
    "parliament": ("full-parliament-", None),
}
ROLES = {"court": ["judge"], "open_review_v2": ["area_chair", "program_chair"],
         "open_review_v3": ["area_chair", "program_chair"], "parliament": ["chair"]}


def short_model(m):
    return m.split("/")[-1]


def read(p):
    with open(p) as f:
        return json.load(f)


def paper_coverage(e, scenario, case_id, arm):
    """Legacy paper_coverage: serialized-tool-result-byte / paper-byte proxy, capped at 1.

    Successful paper-path tool results include JSON wrapper overhead and may
    repeat chunks; this is not unique document coverage.
    """
    if not scenario.startswith("open_review"):
        return None
    root = "open_review_v3" if scenario == "open_review_v3" else "open_review"
    ft = os.path.join(R, "data", "agentic", root, f"{case_id}__{arm}", "docs", "paper", "fulltext.md")
    if not os.path.exists(ft):
        return None
    got = sum(int(t.get("result_size") or 0) for t in e.get("tools") or []
              if (t.get("args") or {}).get("path") == "docs/paper/fulltext.md" and not t.get("isError"))
    return round(min(1.0, got / os.path.getsize(ft)), 4)


def slim_scorer(sc):
    """Drop raw model output and token usage from the scorer record; keep label, quote, reason, status."""
    if not isinstance(sc, dict):
        return sc
    if "status" in sc or "classification" in sc:
        return {k: v for k, v in sc.items() if k not in ("raw_output", "usage")}
    return {k: slim_scorer(v) for k, v in sc.items()}


def role_stats(ed, role, single, scenario, case_id, arm):
    """Tool/usage stats for one role episode; single=True when the episode dir is itself the role dir."""
    base = ed if single else os.path.join(ed, role)
    ep = os.path.join(base, "episode.json")
    if not os.path.exists(ep):
        return None
    e = read(ep)
    u = e.get("usage") or {}
    tools = e.get("tools") or []
    files = sorted(set(e.get("files_read") or []))
    out = {
        "status": e.get("status"),
        "model_returned": e.get("returned_model"),
        "tool_calls": len(tools),
        "files_read": len(files),
        "input_tokens": u.get("input"),
        "output_tokens": u.get("output"),
        "reasoning_tokens": u.get("reasoning"),
        "cache_read_tokens": u.get("cacheRead") or u.get("cache_read"),
        "wall_seconds": e.get("wall_seconds"),
        "integrity_error": e.get("integrity_error"),
    }
    cov = paper_coverage(e, scenario, case_id, arm)
    if cov is not None:
        out["paper_coverage"] = cov
    return out


def load_iclr_decisions():
    """paper_id -> 'accept'/'reject' from data/cases/open_review; empty decision -> absent."""
    out = {}
    for p in glob.glob(os.path.join(R, "data", "cases", "open_review", "iclr*.json")):
        try:
            c = read(p)
        except Exception:
            continue
        d = str(c.get("decision") or "").strip().lower()
        if d:
            out[c["paper_id"]] = "accept" if d.startswith("accept") else "reject"
    return out


ICLR = load_iclr_decisions()


def export(scenario, prefix, exclude):
    rows, batches = [], {}
    for b in sorted(glob.glob(os.path.join(RUNS, prefix + "*"))):
        if exclude and exclude in os.path.basename(b):
            continue
        bname = os.path.basename(b)
        for ed in sorted(glob.glob(os.path.join(b, "episodes", "*"))):
            if not (os.path.exists(ed + "/extraction.json") and os.path.exists(ed + "/context.json")):
                continue
            c = read(ed + "/context.json")
            if c["model"] == SOL_DUP:
                continue
            x = read(ed + "/extraction.json")
            single = len(ROLES[scenario]) == 1
            roles = {r: role_stats(ed, r, single, scenario, c["case_id"], c.get("arm")) for r in ROLES[scenario]}
            if any(v is None or v["status"] != "complete" for v in roles.values()):
                continue
            sc = read(ed + "/score.json") if os.path.exists(ed + "/score.json") else None
            row = {
                "scenario": scenario,
                "episode_id": os.path.basename(ed),
                "batch": bname,
                "case_id": c["case_id"],
                "arm": c.get("arm"),
                "condition": c["condition"],
                "model": short_model(c["model"]),
                "model_requested": c["model"],
                "final": x["final"],
                "pattern_label": x.get("regex") or x.get("classification"),
                "scorer": slim_scorer(x.get("scorer")),
                "agreement": x.get("agreement"),
                "episode_status": x.get("episode_status", "complete"),
                "roles": roles,
            }
            if scenario == "open_review_v2":
                # freeze the real ICLR decision (evaluator-only) so agreement can be computed from results/ alone
                row["final"]["iclr_decision"] = ICLR.get(c["case_id"])
            rows.append(row)
            batches.setdefault(bname, 0)
            batches[bname] += 1
    return rows, batches


def main():
    os.makedirs(OUT, exist_ok=True)
    expected = {"court": 3360, "open_review_v2": 2380, "open_review_v3": 3304, "parliament": 1680}
    if not os.path.isdir(RUNS):
        print(f"no {RUNS}; nothing exported and results/ left untouched", file=sys.stderr)
        sys.exit(1)
    exported, index = {}, {}
    for scen, (prefix, exclude) in SCENARIOS.items():
        rows, batches = export(scen, prefix, exclude)
        rows.sort(key=lambda r: (r["model"], r["condition"], r["arm"] or "", r["case_id"]))
        exported[scen] = (rows, batches)
        print(f"{scen:16s} {len(rows):6d} rows  <- {len(batches)} batches")
    bad = {k: (len(exported[k][0]), v) for k, v in expected.items() if len(exported[k][0]) != v}
    if bad:
        print("COUNT MISMATCH vs frozen plans; results/ left untouched:", bad, file=sys.stderr)
        sys.exit(1)
    # all counts verified: write each file to a temp path and rename atomically
    for scen, (rows, batches) in exported.items():
        path = os.path.join(OUT, f"{scen}.jsonl"); tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, path)
        index[scen] = {"file": os.path.basename(path), "rows": len(rows), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "batches": batches}
    tmp = os.path.join(OUT, "RUNS_INDEX.json.tmp")
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    os.replace(tmp, os.path.join(OUT, "RUNS_INDEX.json"))
    print("all four scenarios match the frozen plan counts; results/ updated")


if __name__ == "__main__":
    main()
