#!/usr/bin/env python3
"""Freeze 120-case samples per scenario for the agentic three-scenario study. Zero LLM calls.

Open Review: dev = the 30 batch-1 base cases (first 30 in the frozen Open Review order) kept as
development set; holdout = 90 new tier-1 (4 reviewers, 3:1 split) cases sampled from the
remaining eligible pool. Court: 120 from 249 with ground_truth_verdict in {petitioner,
respondent} and non-empty evidence on both sides, stratified by difficulty; batch-1's 20 are
excluded from the pilot subset but eligible for the full set. Parliament: 150 candidates
(>=5 witnesses, >=5 submissions with text >= 2000 chars, non-empty ground_truth_conclusion);
120 are kept after A/B direction labeling in a later step. Seed 20260910.
"""
import json, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "cases"
FROZEN_OLD = DATA / "provenance" / "open_review_frozen_night1.json"  # the pre-agentic frozen Open Review list this sample was seeded from
OUT = REPO / "configs" / "agentic" / "cases_frozen_120.json"
SEED = 20260910


def open_review():
    old = json.load(open(FROZEN_OLD))["open_review"]
    dev = [c["case_id"] for c in old[:30]]
    pool = []
    for fp in sorted((DATA / "open_review").glob("*.json")):
        d = json.load(open(fp))
        scores = (d.get("initial_scores") or [])[:4]
        if len(scores) != 4 or not (d.get("rebuttal") or "").strip():
            continue
        if not (DATA / "open_review" / f"{fp.stem}.md").exists():
            continue
        acc = sum(1 for s in scores if s >= 6)
        if {acc, 4 - acc} != {3, 1}:
            continue
        cid = d.get("paper_id") or fp.stem
        if cid in dev:
            continue
        pool.append({"case_id": cid, "initial_scores": scores,
                     "majority_stance": "accept" if acc == 3 else "reject"})
    rng = random.Random(SEED)
    rng.shuffle(pool)
    hold = pool[:90]
    return {"dev": dev, "holdout": [c["case_id"] for c in hold], "holdout_meta": hold, "pool_size": len(pool)}


def court():
    old = {c["case_id"] for c in json.load(open(FROZEN_OLD))["court"][:20]}
    elig = []
    for fp in sorted((DATA / "court").glob("*.json")):
        d = json.load(open(fp))
        if d.get("ground_truth_verdict") not in {"petitioner", "respondent"}:
            continue
        if not d.get("evidence_pro") or not d.get("evidence_def"):
            continue
        if len(d.get("case_summary") or "") < 400:
            continue
        elig.append({"case_id": d["case_id"], "difficulty": d.get("difficulty") or "unknown",
                     "batch1": d["case_id"] in old})
    rng = random.Random(SEED)
    by = {}
    for c in elig:
        by.setdefault(c["difficulty"], []).append(c)
    picked = []
    total = 120
    for k in sorted(by):
        rng.shuffle(by[k])
        n = round(total * len(by[k]) / len(elig))
        picked += by[k][:n]
    picked = picked[:total]
    if len(picked) < total:
        rest = [c for c in elig if c not in picked]
        rng.shuffle(rest)
        picked += rest[: total - len(picked)]
    pilot = [c["case_id"] for c in picked if not c["batch1"]][:20]
    return {"full": [c["case_id"] for c in picked], "pilot": pilot, "eligible": len(elig),
            "difficulty_counts": {k: len(v) for k, v in by.items()}}


def parliament():
    elig = []
    for fp in sorted((DATA / "parliament").glob("parl_*.json")):
        d = json.load(open(fp))
        subs = [s for s in d.get("written_submissions", []) if len(s.get("text") or "") >= 2000]
        if len(subs) < 5 or d.get("n_witnesses", 0) < 5:
            continue
        if len(d.get("ground_truth_conclusion") or "") < 500:
            continue
        elig.append({"case_id": d["inquiry_id"], "committee": d["committee"], "n_subs": len(subs)})
    rng = random.Random(SEED)
    rng.shuffle(elig)
    cand = elig[:150]
    return {"candidates": [c["case_id"] for c in cand], "pilot_candidates": [c["case_id"] for c in cand[:30]],
            "eligible_all": [c["case_id"] for c in elig], "eligible": len(elig)}


if __name__ == "__main__":
    out = {"seed": SEED, "frozen_at": "2026-09-10", "open_review": open_review(), "court": court(),
           "parliament": parliament()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    o = out
    print(f"open_review dev={len(o['open_review']['dev'])} holdout={len(o['open_review']['holdout'])} pool={o['open_review']['pool_size']}")
    print(f"court full={len(o['court']['full'])} pilot={len(o['court']['pilot'])} eligible={o['court']['eligible']} {o['court']['difficulty_counts']}")
    print(f"parliament candidates={len(o['parliament']['candidates'])} eligible={o['parliament']['eligible']}")
    print("->", OUT)
