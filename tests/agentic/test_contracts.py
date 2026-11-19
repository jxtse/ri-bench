import importlib.util
import json
from pathlib import Path
import subprocess
import tomllib
import zipfile

import pytest

from ribench_agentic.batch import run_plan
from ribench_agentic.build.court import build_court_bundle
from ribench_agentic.casefs import build_run_dir, leakage_scan, verify_sha256
from ribench_agentic.cli import main
from ribench_agentic.prompts import DUTIES, OUTPUTS, render
from ribench_agentic.pi_runner import parse_trace, run_pi
from ribench_agentic.summary import summarize


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("role,scenario", [("area_chair", "open_review"), ("program_chair", "open_review"), ("judge", "court"), ("chair", "parliament")])
def test_prompts_only_role_manifest_and_contract(role, scenario):
    manifest = {"scenario": scenario, "arm": "B", "injection": {"truth": "SECRET"}, "files": [{"path": "docs/input.md", "description": "Source document", "role": "evidence", "sha256": "PRIVATE_HASH"}]}
    vanilla, task = render(manifest, "vanilla", role)
    obligation, other_task = render(manifest, "obligation", role)
    assert task == other_task
    assert DUTIES[role] in obligation and DUTIES[role] not in vanilla
    assert OUTPUTS[role] in task
    assert "docs/input.md: Source document" in task
    assert "SECRET" not in task + obligation and "PRIVATE_HASH" not in task


def test_readme_hash_verified_in_bundle_and_run(tmp_path, cases, injections):
    bundle = build_court_bundle(cases["court"], injections["court"], "A", output_root=tmp_path / "bundles")
    run = build_run_dir(bundle, tmp_path / "run", "vanilla")
    for root in [bundle, run]:
        (root / "README.md").write_text("replaced task")
        with pytest.raises(ValueError, match="integrity"):
            verify_sha256(root)


def test_symlink_file_and_private_value_leak(tmp_path, cases, injections):
    bundle = build_court_bundle(cases["court"], injections["court"], "A", output_root=tmp_path / "bundles")
    target = bundle / "docs/case/facts.md"
    target.unlink()
    target.symlink_to(ROOT / "README.md")
    with pytest.raises(ValueError):
        build_run_dir(bundle, tmp_path / "bad", "vanilla")
    private = tmp_path / "private"
    private.mkdir()
    (private / "note.md").write_text(cases["court"]["case_name"])
    assert leakage_scan(private, "court", [cases["court"]["case_name"]])


def test_invalid_tool_and_path_fail_closed(trace):
    for name, path in [("bash", "docs"), ("read", "../MANIFEST.json"), ("read", "/etc/passwd")]:
        events = [json.loads(line) for line in trace.splitlines()]
        for event in events:
            if event["type"] == "tool_execution_start":
                event["toolName"] = name
                event["args"]["path"] = path
        result = parse_trace("\n".join(json.dumps(event) for event in events))
        assert result["status"] == "error"


def test_fail_before_process_on_bad_manifest(tmp_path, cases, injections, monkeypatch):
    bundle = build_court_bundle(cases["court"], injections["court"], "A", output_root=tmp_path / "bundles")
    run = build_run_dir(bundle, tmp_path / "run", "vanilla")
    (run / "docs/case/facts.md").write_text("tampered")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("Process must not launch"))
    with pytest.raises(ValueError):
        run_pi(run, "task", "role", model="foundry-a/gpt-5.6-sol", output_dir=tmp_path / "outputs")


def test_cli_dry_and_build_runs(tmp_path, cases, injections, monkeypatch, capsys):
    case_file = tmp_path / "case.json"
    injection_file = tmp_path / "injection.json"
    case_file.write_text(json.dumps(cases["court"]))
    injection_file.write_text(json.dumps(injections["court"]))
    main(["build-bundle", "--scenario", "court", "--case", str(case_file), "--injection", str(injection_file), "--arm", "A", "--output-root", str(tmp_path / "bundles")])
    bundle = tmp_path / "bundles/court/court_fixture__A"
    plan = {"batch": "cli", "scenario": "court", "cases": [str(bundle)], "arms": ["A"], "conditions": ["vanilla"], "models": ["foundry-a/gpt-5.6-sol"], "thinking": "medium", "timeout_s": 900, "pi_agent_dir": "configs/agentic/pi-agent"}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("No subprocess in dry run"))
    for command in [["build-runs"], ["run", "--dry-run", "--concurrency", "1", "--max-episodes", "1"]]:
        main(command + ["--plan", str(plan_path), "--runs-root", str(tmp_path / "runs")])
    assert "--model foundry-a/gpt-5.6-sol" in capsys.readouterr().out


def test_smoke_plans():
    paths = [ROOT / "configs/agentic/plans" / f"smoke_{s}.json" for s in ("court", "open_review", "parliament")]
    assert len(paths) == 3
    scenarios = set()
    for path in paths:
        plan = json.loads(path.read_text())
        scenarios.add(plan["scenario"])
        assert all(len(plan[key]) == 1 for key in ["cases", "conditions", "models"])
        assert plan["arms"] == {"court": ["A", "B"], "parliament": ["P"], "open_review": ["A"]}[plan["scenario"]]
        assert plan["models"] == ["foundry-a/gpt-5.6-sol"]
    assert scenarios == {"court", "parliament", "open_review"}


def test_packaged_entrypoint_and_schema(tmp_path, monkeypatch):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["scripts"]["ribench-agentic"] == "ribench_agentic.cli:main"
    spec = importlib.util.spec_from_file_location("wp1_build_backend", ROOT / "build_backend.py")
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    monkeypatch.chdir(ROOT)
    wheel = backend.build_wheel(tmp_path)
    with zipfile.ZipFile(tmp_path / wheel) as archive:
        assert "ribench_agentic/cli.py" in archive.namelist()
        assert "ribench_agentic/schemas/injection.schema.json" in archive.namelist()


def test_unknown_usage_accounting(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"model": "provider/id", "status": "error", "usage": {"input": None, "output": 12, "reasoning": None, "cacheRead": None}}) + "\n")
    result = summarize(ledger)["provider/id"]
    assert result["missing_usage"]["input"] == 1
    assert result["usage"]["input"] is None
    assert result["usage"]["output"] == 12


def test_interrupted_ledger_is_recovered_without_second_call(tmp_path, cases, injections, fake_pi, monkeypatch):
    from ribench_agentic import batch
    bundle = build_court_bundle(cases["court"], injections["court"], "A", output_root=tmp_path / "bundles")
    plan = {"batch": "recover", "scenario": "court", "cases": [str(bundle)], "arms": ["A"], "conditions": ["vanilla"], "models": ["foundry-a/gpt-5.6-sol"], "thinking": "medium", "timeout_s": 900, "pi_agent_dir": "configs/agentic/pi-agent"}
    append = batch.append_invocation

    def interrupted(*args, **kwargs):
        raise OSError("Interrupted before ledger append")

    monkeypatch.setattr(batch, "append_invocation", interrupted)
    with pytest.raises(OSError):
        run_plan(plan, runs_root=tmp_path / "runs", pi_binary=fake_pi)
    monkeypatch.setattr(batch, "append_invocation", append)
    run_plan(plan, runs_root=tmp_path / "runs", pi_binary="DO-NOT-LAUNCH")
    run_plan(plan, runs_root=tmp_path / "runs", pi_binary="DO-NOT-LAUNCH")
    ledger = tmp_path / "runs/recover/ledger.jsonl"
    assert len(ledger.read_text().splitlines()) == 1
