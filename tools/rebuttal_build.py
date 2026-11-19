#!/usr/bin/env python3
"""follow-up: build de-identified paper texts, bundles and skeletons for the OpenReview designs.

  rebuttal_build.py deid                      # data/cases/open_review_deid/<case>.md + abstract + MANIFEST.json
  rebuttal_build.py bundles                   # or_r (C,X), or_e1 (P), or_e2a (C,X), or_e3c (X) from frozen v3 reviews
  rebuttal_build.py skeletons                 # or_e2b (SP,SM), or_e3a (X), or_e3b (X) -> data/agentic/_skeletons_rebuttal
  rebuttal_build.py generate --scenario S [--shard i/n] [--limit N]
  rebuttal_build.py audit                     # residual-cue audit of every built bundle (no model calls)

Design: docs/prereg/REBUTTAL_OPENREVIEW_DESIGNS.md. Nothing under data/agentic/open_review_v3 is modified.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ribench_agentic import sanitize_paper  # noqa: E402
from ribench_agentic.build import rebuttal as rb  # noqa: E402
from ribench_agentic.casefs import write_json  # noqa: E402

FROZEN = json.load(open(REPO / "configs/agentic/cases_frozen_v3.json"))
CASES = FROZEN["good"] + FROZEN["bad"]
E3_CASES = REPO / "configs/agentic/rebuttal_e3_cases.json"
OUT = REPO / "data/agentic"
SKEL = REPO / "data/agentic/_skeletons_rebuttal"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_deid(_):
    rb.DEID.mkdir(parents=True, exist_ok=True)
    report = {}
    for cid in CASES:
        raw = (REPO / "data/cases/open_review" / f"{cid}.md").read_text(encoding="utf-8")
        text, counts = sanitize_paper.sanitize(raw)
        (rb.DEID / f"{cid}.md").write_text(text, encoding="utf-8")
        report[cid] = {"raw_sha256": hashlib.sha256(raw.encode()).hexdigest(), "deid_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "raw_chars": len(raw), "deid_chars": len(text), "removed": counts, "residual": sanitize_paper.residual_cues(text)}
    write_json(rb.DEID / "MANIFEST.json", {"sanitizer": "src/ribench_agentic/sanitize_paper.py",
                                           "sanitizer_sha256": sha(REPO / "src/ribench_agentic/sanitize_paper.py"),
                                           "cases": report})
    print(f"deid: {len(report)} papers -> {rb.DEID}")


def _abstract_fix():
    """Abstracts come from OpenReview metadata (camera-ready for accepted papers); strip URLs/emails there too."""
    return lambda text: sanitize_paper.URL.sub("[URL removed]", sanitize_paper.EMAIL.sub("[email removed]", text))


def cmd_bundles(a):
    original = rb._paper_documents
    fix = _abstract_fix()

    def paper_documents(case_id):
        docs = original(case_id)
        return [(p, d, r, fix(t) if p.endswith("abstract.md") else t) for p, d, r, t in docs]

    rb._paper_documents = paper_documents
    e3 = json.load(open(E3_CASES))["cases"]
    built = skipped = 0
    jobs = []
    for cid in CASES:
        jobs += [("or_r", cid, arm, lambda c=cid, x=arm: rb.build_or_r(c, x, OUT)) for arm in ("C", "X")]
        jobs += [("or_e1", cid, "P", lambda c=cid: rb.build_or_e1(c, OUT))]
        jobs += [("or_e2a", cid, arm, lambda c=cid, x=arm: rb.build_or_e2a(c, x, OUT)) for arm in ("C", "X")]
    for cid in e3:
        jobs += [("or_e3c", cid, "X", lambda c=cid: rb.build_or_e3c(c, OUT))]
    for scenario, cid, arm, fn in jobs:
        if a.only and scenario not in a.only:
            continue
        if (OUT / scenario / f"{cid}__{arm}").exists():
            skipped += 1
            continue
        fn()
        built += 1
    print(f"bundles built={built} skipped={skipped}")


def cmd_skeletons(a):
    original = rb._paper_documents
    fix = _abstract_fix()
    rb._paper_documents = lambda case_id: [(p, d, r, fix(t) if p.endswith("abstract.md") else t) for p, d, r, t in original(case_id)]
    e3 = json.load(open(E3_CASES))["cases"]
    built = skipped = 0
    jobs = [("or_e2b", cid, arm, lambda c=cid, x=arm: rb.build_or_e2b_skeleton(c, x, SKEL)) for cid in CASES for arm in ("SP", "SM")]
    jobs += [("or_e3a", cid, "X", lambda c=cid: rb.build_or_e3a_skeleton(c, SKEL)) for cid in e3]
    jobs += [("or_e3b", cid, "X", lambda c=cid: rb.build_or_e3b_skeleton(c, SKEL)) for cid in e3]
    for scenario, cid, arm, fn in jobs:
        if a.only and scenario not in a.only:
            continue
        if (SKEL / scenario / f"{cid}__{arm}").exists():
            skipped += 1
            continue
        fn()
        built += 1
    print(f"skeletons built={built} skipped={skipped} -> {SKEL}")


def _seed_reused_initial(skeleton, work):
    """E3a/E3b reuse v3 initial reviews verbatim: pre-seed completed generation records so generate_reviews keeps them."""
    manifest = json.loads((skeleton / "MANIFEST.json").read_text())
    source = manifest.get("reuse_initial_from")
    if not source:
        return 0
    seeded = 0
    for a in manifest["reviewer_assignment"]:
        src = OUT / source / "docs/reviews" / f"reviewer_{a['reviewer_id']}.md"
        if not src.exists():
            continue  # new reviewer (E3b slot 5) is generated
        target = work / f"reviewer_{a['reviewer_id']}"
        if (target / "episode.json").exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "episode.json", {"status": "complete", "final_text": src.read_text(encoding="utf-8"),
                                             "reused_from": str(src.relative_to(REPO)), "source_sha256": sha(src)})
        seeded += 1
    return seeded


def cmd_generate(a):
    from ribench_agentic.generate_reviews import generate_reviews
    root = SKEL / a.scenario
    targets = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        targets = targets[i::n]
    done = 0
    for sk in targets:
        if done >= a.limit:
            break
        if (OUT / a.scenario / sk.name).exists():
            continue
        work_root = REPO / "runs/agentic" / f"review-generation-{a.scenario}"
        seeded = _seed_reused_initial(sk, work_root / sk.name)
        t = time.time()
        try:
            res = generate_reviews(sk, output_root=OUT, work_root=work_root, timeout_s=900)
            print(f"  {sk.name} ok {time.time() - t:.0f}s seeded={seeded} -> {res}", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep the shard going; failures are listed and retried on rerun
            print(f"  {sk.name} FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        done += 1


def cmd_audit(_):
    rows = {}
    for scenario in ("or_r", "or_e1", "or_e2a", "or_e2b", "or_e3a", "or_e3b", "or_e3c"):
        root = OUT / scenario
        if not root.exists():
            continue
        agg = {"bundles": 0, "paper_residual": {}, "review_flags": {}, "score_residual": 0}
        for b in sorted(root.iterdir()):
            if not (b / "MANIFEST.json").exists():
                continue
            agg["bundles"] += 1
            paper = (b / "docs/paper/fulltext.md").read_text(encoding="utf-8") + (b / "docs/paper/abstract.md").read_text(encoding="utf-8")
            for key, value in sanitize_paper.residual_cues(paper).items():
                if value and not (key == "line_numbers" and value < 30):
                    agg["paper_residual"][key] = agg["paper_residual"].get(key, 0) + 1
            for f in sorted((b / "docs/reviews").glob("*.md")) if (b / "docs/reviews").exists() else []:
                text = f.read_text(encoding="utf-8")
                for key, pattern in {"published_status": r"published as a conference|camera[- ]ready (?:version|paper) (?:of|shows)|this (?:camera|published) version",
                                     "url": r"https?://|github\.com", "email": sanitize_paper.EMAIL.pattern}.items():
                    if re.search(pattern, text, re.I):
                        agg["review_flags"][key] = agg["review_flags"].get(key, 0) + 1
                if scenario == "or_e2a" and f.name != "rebuttal.md" and rb.residual_scores(text):
                    agg["score_residual"] += 1
        rows[scenario] = agg
    write_json(REPO / "build/qa/rebuttal-20261107/bundle_audit.json", rows)
    print(json.dumps(rows, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("deid")
    b = sub.add_parser("bundles"); b.add_argument("--only", nargs="*")
    s = sub.add_parser("skeletons"); s.add_argument("--only", nargs="*")
    g = sub.add_parser("generate"); g.add_argument("--scenario", required=True); g.add_argument("--shard"); g.add_argument("--limit", type=int, default=10**6)
    sub.add_parser("audit")
    a = p.parse_args()
    (REPO / "build/qa/rebuttal-20261107").mkdir(parents=True, exist_ok=True)
    {"deid": cmd_deid, "bundles": cmd_bundles, "skeletons": cmd_skeletons, "generate": cmd_generate, "audit": cmd_audit}[a.cmd](a)


if __name__ == "__main__":
    main()
