"""Materialize and run bounded plans; each Open Review episode has AC then PC."""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import itertools
import json
from pathlib import Path
import shlex

from .build.common import allowed_arms, identifier
from .casefs import build_run_dir, sha256, snapshot, verify_sha256, write_json
from .extract import extract_all
from .ledger import append_invocation
from .pi_runner import command, run_pi
from .harness_runner import HARNESSES, run_harness
from .prompts import DEFAULT_ROLE, is_review_scenario, render
from . import prompts


# Subject model identifiers as "<route>/<model>". Route names are the provider keys of the pi model configuration
# (configs/agentic/pi-agent/models.example.json): foundry-a / foundry-b = Microsoft Foundry deployments, mint = MinT,
# ark = Volcengine Ark, openrouter = OpenRouter pinned to one upstream, deepseek = DeepSeek official API,
# vllm = open weights served with vLLM. Several models were served by more than one route over the project; every
# episode records the identifier the provider returned, and the paper's appendix lists the route of each table.
SUBJECT_MODELS = {"foundry-a/claude-opus-5", "foundry-a/gpt-5.6-sol", "foundry-a/gpt-5.6-terra", "foundry-b/gpt-5.6-luna",
                  "foundry-b/gpt-5.6-sol", "foundry-b/gpt-5.6-terra", "mint/glm-5.3", "mint/deepseek-flash",
                  "ark/glm-5.3-flash", "tokendance/qwen3.5-35b-a3b", "openrouter/qwen/qwen3.5-35b-a3b",
                  "vllm/qwen3.5-35b-a3b", "deepseek/deepseek-flash"}


def load_plan(plan):
    if isinstance(plan, (str, Path)):
        return json.loads(Path(plan).read_text(encoding="utf-8"))
    return plan


def episodes(plan, runs_root):
    required = {"batch", "scenario", "cases", "arms", "conditions", "models", "thinking", "timeout_s", "pi_agent_dir"}
    if not required <= plan.keys():
        raise ValueError("Missing required plan fields")
    identifier(plan["batch"])
    if plan["scenario"] not in DEFAULT_ROLE or plan["timeout_s"] <= 0:
        raise ValueError("Invalid scenario or timeout")
    if plan["thinking"] not in {"off", "minimal", "low", "medium", "high", "xhigh"}:
        raise ValueError("Invalid thinking level")
    # harness-variance experiment (2026-09-23): optional plan key; absent/"pi" keeps the paper's pi lane byte-identical
    harness = plan.get("harness", "pi")
    if harness != "pi" and harness not in HARNESSES:
        raise ValueError("Unknown harness")
    for key in ["cases", "arms", "conditions", "models"]:
        if not isinstance(plan[key], list) or not plan[key]:
            raise ValueError(f"Empty or invalid plan {key}")
    if not set(plan["arms"]) <= allowed_arms(plan["scenario"]) or not set(plan["conditions"]) <= {"vanilla", "obligation"} or not set(plan["models"]) <= SUBJECT_MODELS:
        raise ValueError("Invalid plan arm, condition or subject model")
    batch_root = Path(runs_root).resolve() / plan["batch"]
    result, identities = [], set()
    # order: case-major, then arm, condition, model — so the first N episodes (kill-check window) cover
    # every model/arm/condition on the first few cases rather than one model on all cases
    for case, arm, condition, model in itertools.product(plan["cases"], plan["arms"], plan["conditions"], plan["models"]):
        case_path = Path(case)
        if case_path.is_dir():
            bundle = case_path.resolve()
        else:
            identifier(case)
            bundle = Path("data/agentic") / plan["scenario"] / f"{case}__{arm}"
            bundle = bundle.resolve()
        state = verify_sha256(bundle)
        manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
        if manifest["scenario"] != plan["scenario"] or manifest["arm"] != arm:
            raise ValueError("Plan does not match bundle scenario/arm")
        if is_review_scenario(plan["scenario"]):
            if plan["scenario"] in {"open_review", "open_review_v3"}:
                reviewer_ids = range(1, 5)
            else:  # rebuttal designs carry 0 (paper only), 3, 4 or 5 reviewers
                reviewer_ids = [a["reviewer_id"] for a in manifest.get("reviewer_assignment", [])]
            post = [] if manifest.get("no_post_rebuttal") else ["post_rebuttal"]
            required_reviews = {f"docs/reviews/{prefix}_{index}.md" for prefix in ["reviewer", *post] for index in reviewer_ids}
            if not manifest.get("reviews_complete") or not required_reviews <= state.keys():
                raise ValueError("Frozen reviews are incomplete; generate reviews first")
        case_id = identifier(manifest["case_id"])
        provider, model_id = model.split("/", 1)
        identifier(provider)
        # model ids may be namespaced (OpenRouter: qwen/qwen3.5-35b-a3b); each segment must still be a safe identifier
        for segment in model_id.split("/"):
            identifier(segment)
        episode_id = sha256(json.dumps([case_id, arm, condition, model] + ([harness] if harness != "pi" else [])).encode())
        destination = batch_root / "episodes" / episode_id
        if destination in identities:
            raise ValueError("Duplicate episode in plan")
        identities.add(destination)
        context = {"scenario": plan["scenario"], "case_id": case_id, "arm": arm, "condition": condition,
                   "model": model, "bundle_sha256": sha256(json.dumps(state, sort_keys=True).encode()),
                   "manifest": manifest, "thinking": plan["thinking"], "timeout_s": plan["timeout_s"],
                   "pi_agent_dir": str(Path(plan["pi_agent_dir"]).resolve()),
                   "templates_sha256": sha256(Path(prompts.__file__).read_bytes())}
        if harness != "pi":
            context["harness"] = harness
        result.append((bundle, destination, context))
    return result


def prepare_workspace(bundle, output, context, role, area_chair_text=None):
    workspace = output / "workspace"
    if workspace.exists():
        expected = json.loads((output / "input_sha256.json").read_text())
        verify_sha256(workspace, expected)
        if area_chair_text is not None and (workspace / "docs/meta_review/area_chair.md").read_text() != area_chair_text:
            raise ValueError("Existing PC input does not match AC output")
    else:
        build_run_dir(bundle, workspace, context["condition"], role, area_chair_text)
        write_json(output / "input_sha256.json", snapshot(workspace))
    return workspace


def print_command(workspace, prompt, system, context, pi_binary):
    argv = command(prompt, system, model=context["model"], thinking=context["thinking"], pi_binary=pi_binary)
    print(f"cd {shlex.quote(str(workspace))} && PI_CODING_AGENT_DIR={shlex.quote(context['pi_agent_dir'])} {shlex.join(argv)}")
    return argv


def reconcile_ledger(destination, context, ledger):
    with (destination / ".episode.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _reconcile_ledger(destination, context, ledger)


def _reconcile_ledger(destination, context, ledger):
    for intent in sorted(destination.rglob("invocation.json")):
        recorded = intent.parent / "episode.json"
        if recorded.exists():
            invocation = json.loads(intent.read_text())
            result = json.loads(recorded.read_text())
            append_invocation(ledger, context, result, invocation["system_prompt"], invocation["prompt"],
                              invocation["role"], invocation_id=sha256(str(intent.parent).encode()))


def _context_core(context):
    """Fields that define what the agent saw. The private manifest and its hash may be repaired between
    phases (e.g. leakage-scan list fixes) without changing the agent-visible input, which input_sha256.json guards."""
    return {k: v for k, v in context.items() if k not in {"bundle_sha256", "manifest"}}


def run_episode(bundle, destination, context, ledger, dry_run, pi_binary):
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".episode.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        context_path = destination / "context.json"
        if context_path.exists() and _context_core(json.loads(context_path.read_text())) != _context_core(context):
            raise ValueError("Existing episode context differs; use a new batch")
        if not context_path.exists():
            write_json(context_path, context)
        _reconcile_ledger(destination, context, ledger)
        if (destination / "episode.json").exists():
            extract_all(destination)
            return {"status": "skipped", "episode_dir": str(destination)}
        roles = ["area_chair", "program_chair"] if is_review_scenario(context["scenario"]) else [DEFAULT_ROLE[context["scenario"]]]
        ac_text, last = None, None
        for role in roles:
            output = destination / role if len(roles) == 2 else destination
            output.mkdir(parents=True, exist_ok=True)
            recorded = output / "episode.json"
            if recorded.exists():
                last = json.loads(recorded.read_text())
            elif dry_run and role == "program_chair":
                public = {"scenario": context["scenario"], "files": context["manifest"]["files"] + [{"path": "docs/meta_review/area_chair.md", "description": "Area Chair meta-review from this episode"}]}
                system, prompt = render(public, context["condition"], role)
                print_command(output / "workspace", prompt, system, context, pi_binary)
                continue
            else:
                workspace = prepare_workspace(bundle, output, context, role, ac_text)
                public = json.loads((workspace / "MANIFEST.public.json").read_text())
                system, prompt = render(public, context["condition"], role)
                if dry_run:
                    print_command(workspace, prompt, system, context, pi_binary)
                    continue
                intent = output / "invocation.json"
                if intent.exists():
                    raise ValueError("Unsettled invocation exists; reconcile it before any new call")
                write_json(intent, {"attempt": 1, "role": role, "model": context["model"], "system_prompt": system, "prompt": prompt})
                if context.get("harness", "pi") == "pi":
                    last = run_pi(workspace, prompt, system, model=context["model"], thinking=context["thinking"],
                                  timeout_s=context["timeout_s"], pi_agent_dir=context["pi_agent_dir"], output_dir=output, pi_binary=pi_binary)
                else:
                    last = run_harness(context["harness"], workspace, prompt, system, model=context["model"].split("/", 1)[1],
                                       thinking=context["thinking"], timeout_s=context["timeout_s"], output_dir=output)
                append_invocation(ledger, context, last, system, prompt, role, invocation_id=sha256(str(output).encode()))
            if last["status"] != "complete":
                break
            if role == "area_chair":
                ac_text = last["final_text"]
        if dry_run:
            return {"status": "dry_run", "episode_dir": str(destination)}
        if len(roles) == 2:
            write_json(destination / "episode.json", {"status": last["status"], "final_text": last["final_text"], "roles": roles})
        extract_all(destination)
        return {"status": last["status"], "episode_dir": str(destination)}


def run_plan(plan, *, runs_root="runs/agentic", concurrency=1, max_episodes=None, dry_run=False, pi_binary="pi"):
    plan = load_plan(plan)
    if concurrency < 1 or (max_episodes is not None and max_episodes < 0):
        raise ValueError("Invalid concurrency or episode limit")
    entries = episodes(plan, runs_root)
    ledger = Path(runs_root).resolve() / plan["batch"] / "ledger.jsonl"
    selected, skipped = [], []
    for entry in entries:
        if (entry[1] / "episode.json").exists():
            previous = json.loads((entry[1] / "context.json").read_text())
            if _context_core(previous) != _context_core(entry[2]):
                raise ValueError("Existing episode context differs; use a new batch")
            reconcile_ledger(entry[1], entry[2], ledger)
            extract_all(entry[1])
            skipped.append({"status": "skipped", "episode_dir": str(entry[1])})
        elif max_episodes is None or len(selected) < max_episodes:
            selected.append(entry)

    def execute(entry):
        return run_episode(*entry, ledger, dry_run, pi_binary)

    if concurrency == 1:
        return skipped + [execute(entry) for entry in selected]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return skipped + list(executor.map(execute, selected))


def build_runs(plan, *, runs_root="runs/agentic", pi_binary="pi"):
    return run_plan(plan, runs_root=runs_root, dry_run=True, pi_binary=pi_binary)
