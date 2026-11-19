#!/usr/bin/env python3
"""Download the RI-Bench case corpus from Hugging Face into data/cases/.

The case files (Open Review, Parliament, Court and provenance) are published at
https://huggingface.co/datasets/jxtse/RI-Bench under CC BY-NC 4.0. The raw per-case files there have the same
layout as data/cases/ in this repository, so the bundle builders and exporters read them unchanged.

Usage: make cases   (or: uv run --no-project --with huggingface_hub python tools/fetch_cases.py [--revision <commit>])
"""
import argparse
from pathlib import Path

REPO_ID = "jxtse/RI-Bench"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revision", default=None, help="dataset commit or tag (default: main)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "cases")
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub is required: run `make cases` or `pip install huggingface_hub`")
    path = snapshot_download(REPO_ID, repo_type="dataset", revision=args.revision, local_dir=args.out,
                             allow_patterns=["open_review/*", "parliament/*", "court/*", "provenance/*", "LICENSE"])
    n = {d: len(list((Path(path) / d).glob("*.json"))) for d in ("open_review", "parliament", "court")}
    print(f"data/cases ready: {n}")


if __name__ == "__main__":
    main()
