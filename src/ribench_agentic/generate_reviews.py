"""Separate lower-tier generation job; dry runs never invent review text."""

import argparse
import json
import time
import shutil
from pathlib import Path
import shlex

from .build.common import publish_bundle
from .casefs import sha256, verify_sha256
from .ledger import append_invocation
from .pi_runner import DEFAULT_AGENT_DIR, command, run_pi
from .prompts import REBUTTAL_SCENARIOS, reviewer_prompt


GENERATOR_MODEL = "foundry-a/claude-sonnet-5"


def generate_reviews(skeleton, *, max_attempts=3, output_root="data/agentic", work_root="runs/agentic/review-generation",
                     dry_run=False, pi_binary="pi", pi_agent_dir=DEFAULT_AGENT_DIR, timeout_s=900):
    skeleton = Path(skeleton).resolve()
    state = verify_sha256(skeleton)
    manifest = json.loads((skeleton / "MANIFEST.json").read_text())
    if manifest["scenario"] not in {"open_review", "open_review_v3", *REBUTTAL_SCENARIOS} or manifest.get("reviews_complete"):
        raise ValueError("Expected an unfinished Open Review skeleton")
    destination = Path(output_root) / manifest["scenario"] / f"{manifest['case_id']}__{manifest['arm']}"
    if not dry_run and destination.exists():
        raise FileExistsError("Frozen output must be new; use a separate root for skeletons")
    work = Path(work_root).resolve() / f"{manifest['case_id']}__{manifest['arm']}"
    work.mkdir(parents=True, exist_ok=True)  # resume-safe: completed roles inside are reused
    context = {"scenario": manifest["scenario"], "case_id": manifest["case_id"], "arm": manifest["arm"],
               "condition": "generation", "model": GENERATOR_MODEL,
               "bundle_sha256": sha256(json.dumps(state, sort_keys=True).encode())}
    generated, commands = {}, []
    reviewer_ids = [a["reviewer_id"] for a in manifest["reviewer_assignment"]]
    rebuttal_design = manifest["scenario"] in REBUTTAL_SCENARIOS
    for post in [False, True]:
        prefix = "post_rebuttal" if post else "reviewer"
        for assignment in manifest["reviewer_assignment"]:
            reviewer_id = assignment["reviewer_id"]
            output = work / f"{prefix}_{reviewer_id}"
            prior = output / "episode.json"
            if prior.exists() and not dry_run:
                # resume: a completed role from an earlier attempt is reused verbatim (never regenerated)
                previous = json.loads(prior.read_text(encoding="utf-8"))
                if previous.get("status") == "complete" and previous.get("final_text"):
                    generated[f"{prefix}_{reviewer_id}"] = previous["final_text"]
                    continue
            if output.exists() and not dry_run:
                shutil.rmtree(output)  # partial/failed earlier attempt (no or incomplete episode.json)
            workspace = output / "workspace"
            workspace.mkdir(parents=True)
            documents = {}
            for name in ["abstract.md", "fulltext.md"]:
                documents[f"paper/{name}"] = (skeleton / "docs/paper" / name).read_text()
            if post:
                documents["rebuttal.md"] = (skeleton / "docs/reviews/rebuttal.md").read_text()
                for index in reviewer_ids:
                    if not dry_run:
                        documents[f"reviews/reviewer_{index}.md"] = generated[f"reviewer_{index}"]
            for relative, text in documents.items():
                target = workspace / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            listing = [f"- {relative}: {'Author rebuttal' if relative == 'rebuttal.md' else 'Paper document' if relative.startswith('paper/') else 'Initial reviewer report'}" for relative in documents]
            if post and dry_run:
                listing += [f"- reviews/reviewer_{index}.md: Initial reviewer report" for index in reviewer_ids]
            system, task = reviewer_prompt(assignment, post, submission_frame=rebuttal_design)
            prompt = task + "\n\nFiles:\n" + "\n".join(listing) + "\n\nOutput: My score: X/10, followed by 3–4 numbered points, 180–260 words."
            argv = command(prompt, system, model=GENERATOR_MODEL, thinking="medium", pi_binary=pi_binary)
            if dry_run:
                print(f"cd {shlex.quote(str(workspace))} && PI_CODING_AGENT_DIR={shlex.quote(str(Path(pi_agent_dir).resolve()))} {shlex.join(argv)}")
                commands.append(argv)
                continue
            result = None
            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    # keep the failed attempt auditable, then give run_pi a clean output dir with the same workspace
                    failed = output.parent / f"{output.name}.failed{attempt - 1}"
                    if failed.exists():
                        shutil.rmtree(failed)
                    failed.mkdir()
                    for item in list(output.iterdir()):
                        if item.name != "workspace":
                            shutil.move(str(item), str(failed / item.name))
                    time.sleep(min(60 * attempt, 180))
                result = run_pi(workspace, prompt, system, model=GENERATOR_MODEL, thinking="medium", timeout_s=timeout_s,
                                pi_agent_dir=pi_agent_dir, output_dir=output, pi_binary=pi_binary)
                result["attempt"] = attempt
                append_invocation(work / "ledger.jsonl", context, result, system, prompt, f"{prefix}_{reviewer_id}")
                if result["status"] == "complete":
                    break
            if result["status"] != "complete":
                raise ValueError(f"Review generation failed at {prefix}_{reviewer_id} after {max_attempts} attempts")
            text = result["final_text"]
            if manifest["scenario"] == "open_review_v3" or rebuttal_design:
                # drop any assistant preamble before the review proper (v2 bundles are frozen and left as generated)
                idx = text.find("My score:")
                text = text[idx:] if idx > 0 else text
            generated[f"{prefix}_{reviewer_id}"] = text
    verify_sha256(skeleton, state)
    if dry_run:
        return commands
    documents = [(entry["path"], entry["description"], entry["role"], (skeleton / entry["path"]).read_text()) for entry in manifest["files"] if entry["path"] != "README.md"]
    documents += [(f"docs/reviews/{name}.md", f"{'Post-rebuttal response' if name.startswith('post') else 'Initial review'} from Reviewer {name.rsplit('_', 1)[1]}", "review", text) for name, text in generated.items()]
    private = {key: value for key, value in manifest.items() if key not in {"files", "scenario", "case_id", "arm"}}
    private.update(reviews_complete=True, generator_model=GENERATOR_MODEL, generator_thinking="medium", skeleton_sha256=context["bundle_sha256"])
    return publish_bundle(manifest["scenario"], manifest["case_id"], manifest["arm"], documents, private, output_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("skeleton", type=Path)
    parser.add_argument("--output-root", default="data/agentic")
    parser.add_argument("--work-root", default="runs/agentic/review-generation")
    parser.add_argument("--pi-agent-dir", default=str(DEFAULT_AGENT_DIR))
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args())
    result = generate_reviews(**args)
    if not args["dry_run"]:
        print(result)


if __name__ == "__main__":
    main()
