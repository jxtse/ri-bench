import argparse
import json
from pathlib import Path

from .batch import build_runs, run_plan
from .build.court import build_court_bundle
from .build.open_review import build_open_review_skeleton
from .build.parliament import build_parliament_bundle
from .build.from_generator import build_bundles
from .extract import reextract
from .score import score_all
from .summary import summarize


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ribench-agentic")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["build-runs", "run", "controller"]:
        child = commands.add_parser(name)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--runs-root", default="runs/agentic")
        if name in {"run", "controller"}:
            child.add_argument("--concurrency", type=int, default=1)
            child.add_argument("--max-episodes", type=int)
        if name == "run":
            child.add_argument("--dry-run", action="store_true")
        if name == "controller":
            child.add_argument("--resume", action="store_true")
            child.add_argument("--order", default="forward", help="forward, reverse, middle or shard:i/n")
            child.add_argument("--progress-name", default="progress.json")
    builds = commands.add_parser("build-bundles")
    builds.add_argument("--scenario", choices=["court", "parliament"], required=True)
    builds.add_argument("--injections", type=Path, required=True)
    builds.add_argument("--cases-root", type=Path, default="data/cases")
    builds.add_argument("--output-root", default="data/agentic")
    build = commands.add_parser("build-bundle")
    build.add_argument("--scenario", choices=["court", "parliament", "open_review"], required=True)
    build.add_argument("--case", type=Path, required=True)
    build.add_argument("--injection", type=Path)
    build.add_argument("--arm", choices=["A", "B", "P"], required=True)
    build.add_argument("--fulltext", type=Path)
    build.add_argument("--output-root", default="data/agentic")
    extract = commands.add_parser("extract")
    extract.add_argument("--runs-root", default="runs/agentic")
    score = commands.add_parser("score")
    score.add_argument("--runs-root", required=True)
    score.add_argument("--only-missing", action="store_true",
                       help="Skip existing score.json files (also the default)")
    summary = commands.add_parser("summary")
    summary.add_argument("--ledger", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    action = args.pop("command")
    if action == "controller":
        from .controller import run_controller
        raise SystemExit(run_controller(**args))
    if action in {"run", "build-runs"}:
        result = (run_plan if action == "run" else build_runs)(**args)
    elif action == "build-bundles":
        result = build_bundles(**args)
    elif action == "extract":
        result = {"reextracted": reextract(args["runs_root"])}
    elif action == "score":
        result = score_all(**args)
    elif action == "summary":
        result = summarize(args["ledger"])
    else:
        case = json.loads(args["case"].read_text(encoding="utf-8"))
        if args["scenario"] == "open_review":
            if args["fulltext"]:
                case["fulltext"] = args["fulltext"].read_text(encoding="utf-8")
            result = build_open_review_skeleton(case, args["arm"], output_root=args["output_root"])
        else:
            if args["injection"] is None:
                parser.error("--injection is required for court/parliament")
            injection = json.loads(args["injection"].read_text(encoding="utf-8"))
            builder = build_court_bundle if args["scenario"] == "court" else build_parliament_bundle
            result = builder(case, injection, args["arm"], output_root=args["output_root"])
    print(json.dumps(result, default=str, ensure_ascii=False, indent=2))
    if action == "build-bundles" and result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
