#!/usr/bin/env python3
"""Open Review v3 (constructed adversarial votes): build C/X skeletons for the frozen 118 cases and generate reviews.

  agentic_or_v3_build.py skeletons
  agentic_or_v3_build.py generate [--limit N] [--shard i/n] [--dry-run]

Truth (ICLR real decision) comes from the frozen config's good/bad lists, never from the public bundle.
Frozen bundles -> data/agentic/open_review_v3/<case>__<C|X>; generation work dirs -> runs/agentic/review-generation-v3/.
"""
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ribench_agentic.build.open_review_v3 import build_open_review_v3_skeleton  # noqa: E402
from ribench_agentic.generate_reviews import generate_reviews  # noqa: E402

DATA = REPO / "data" / "cases"
FROZEN = json.load(open(REPO / "configs" / "agentic" / "cases_frozen_v3.json"))
SKELETONS = REPO / "data" / "agentic" / "_skeletons_v3"
OUT = REPO / "data" / "agentic"
TRUTH = {**{c: "accept" for c in FROZEN["good"]}, **{c: "reject" for c in FROZEN["bad"]}}


def load_case(cid):
    d = json.load(open(DATA / "open_review" / f"{cid}.json"))
    d["fulltext"] = (DATA / "open_review" / f"{cid}.md").read_text(encoding="utf-8")
    return d


def cmd_skeletons(a):
    built = skipped = 0
    for cid, truth in TRUTH.items():
        case = load_case(cid)
        for arm in ("C", "X"):
            if (SKELETONS / "open_review_v3" / f"{cid}__{arm}").exists():
                skipped += 1; continue
            build_open_review_v3_skeleton(case, arm, truth=truth, output_root=SKELETONS); built += 1
    print(f"skeletons built={built} skipped={skipped} -> {SKELETONS}/open_review_v3")


def cmd_generate(a):
    done = 0
    targets = [(cid, arm) for cid in TRUTH for arm in ("C", "X")]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        targets = targets[i::n]
    for cid, arm in targets:
        if done >= a.limit:
            return
        if (OUT / "open_review_v3" / f"{cid}__{arm}").exists():
            continue
        sk = SKELETONS / "open_review_v3" / f"{cid}__{arm}"
        if not sk.exists():
            print("missing skeleton", sk); continue
        t = time.time()
        try:
            res = generate_reviews(sk, output_root=OUT, work_root=REPO / "runs" / "agentic" / "review-generation-v3",
                                   dry_run=a.dry_run, timeout_s=900)
            print(f"  {cid}__{arm} {'dry' if a.dry_run else 'ok'} {time.time()-t:.0f}s {'' if a.dry_run else res}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {cid}__{arm} FAILED {time.time()-t:.0f}s {str(e)[:200]}", flush=True)
        done += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("skeletons")
    g = sub.add_parser("generate"); g.add_argument("--limit", type=int, default=1); g.add_argument("--dry-run", action="store_true"); g.add_argument("--shard", default=None)
    a = ap.parse_args()
    {"skeletons": cmd_skeletons, "generate": cmd_generate}[a.cmd](a)
