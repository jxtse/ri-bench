#!/usr/bin/env python3
"""Build Open Review skeletons (paper + rebuttal + private reviewer assignment) for the frozen 120 cases,
then optionally run the Sonnet-5 pi-harness review generation for N of them. Zero subject-model calls.

  agentic_or_build.py skeletons [--set dev|holdout|all]
  agentic_or_build.py generate --limit N [--set dev|holdout|all] [--dry-run]
"""
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ribench_agentic.build.open_review import build_open_review_skeleton  # noqa: E402
from ribench_agentic.generate_reviews import generate_reviews  # noqa: E402

DATA = REPO / "data" / "cases"
FROZEN = json.load(open(REPO / "configs" / "agentic" / "cases_frozen_120.json"))["open_review"]
SKELETONS = REPO / "data" / "agentic" / "_skeletons"
OUT = REPO / "data" / "agentic"


def case_ids(which):
    return FROZEN["dev"] + FROZEN["holdout"] if which == "all" else FROZEN[which]


def load_case(cid):
    d = json.load(open(DATA / "open_review" / f"{cid}.json"))
    d["fulltext"] = (DATA / "open_review" / f"{cid}.md").read_text(encoding="utf-8")
    return d


def cmd_skeletons(a):
    built = skipped = 0
    for cid in case_ids(a.set):
        case = load_case(cid)
        for arm in ("A", "B"):
            if (SKELETONS / "open_review" / f"{cid}__{arm}").exists():
                skipped += 1; continue
            build_open_review_skeleton(case, arm, output_root=SKELETONS); built += 1
    print(f"skeletons built={built} skipped={skipped} -> {SKELETONS}/open_review")


def cmd_generate(a):
    done = 0
    targets = [(cid, arm) for cid in case_ids(a.set) for arm in ("A", "B")]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        targets = targets[i::n]
    for cid, arm in targets:
        if True:
            if done >= a.limit:
                return
            if (OUT / "open_review" / f"{cid}__{arm}").exists():
                continue
            sk = SKELETONS / "open_review" / f"{cid}__{arm}"
            if not sk.exists():
                print("missing skeleton", sk); continue
            t = time.time()
            try:
                res = generate_reviews(sk, output_root=OUT, work_root=REPO / "runs" / "agentic" / "review-generation",
                                       dry_run=a.dry_run, timeout_s=900)
                print(f"  {cid}__{arm} {'dry' if a.dry_run else 'ok'} {time.time()-t:.0f}s {'' if a.dry_run else res}", flush=True)
            except Exception as e:
                print(f"  {cid}__{arm} FAILED {time.time()-t:.0f}s {str(e)[:200]}", flush=True)
            done += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("skeletons"); s.add_argument("--set", default="all")
    g = sub.add_parser("generate"); g.add_argument("--set", default="all"); g.add_argument("--limit", type=int, default=1); g.add_argument("--dry-run", action="store_true"); g.add_argument("--shard", default=None)
    a = ap.parse_args()
    {"skeletons": cmd_skeletons, "generate": cmd_generate}[a.cmd](a)
