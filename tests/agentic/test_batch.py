import json
from pathlib import Path

import pytest

from ribench_agentic.batch import run_plan
from ribench_agentic.build.court import build_court_bundle
from ribench_agentic.build.open_review import build_open_review_skeleton
from ribench_agentic.extract import extract_all, reextract
from ribench_agentic.generate_reviews import generate_reviews
from ribench_agentic.summary import summarize


def plan_for(bundle, scenario="court"):
    return {"batch": "offline", "scenario": scenario, "cases": [str(bundle)], "arms": ["A"], "conditions": ["obligation"], "models": ["foundry-a/gpt-5.6-sol"], "thinking": "medium", "timeout_s": 900, "pi_agent_dir": "configs/agentic/pi-agent"}


@pytest.mark.parametrize("arms", [["A"], ["B"], ["A", "B"], ["P", "A"]])
def test_parliament_build_runs_rejects_old_arms(tmp_path, arms):
    from ribench_agentic.batch import build_runs
    plan = plan_for("unused", "parliament")
    plan["arms"] = arms
    with pytest.raises(ValueError, match="arm"):
        build_runs(plan, runs_root=tmp_path)
    assert not list(tmp_path.iterdir())


def test_parliament_p_build_runs_and_extract(tmp_path, cases, injections):
    from ribench_agentic.batch import build_runs
    from ribench_agentic.build.parliament import build_parliament_bundle
    bundle = build_parliament_bundle(cases["parliament"], injections["parliament"], "P", output_root=tmp_path / "bundles")
    plan = plan_for(bundle, "parliament")
    plan["arms"] = ["P"]
    build_runs(plan, runs_root=tmp_path / "runs")
    context = next((tmp_path / "runs").rglob("context.json"))
    (context.parent / "episode.json").write_text(json.dumps({"status": "ok", "final_text": "62% support (n=41)."}))
    result = extract_all(context.parent)
    assert result["aligned"]["classification"] == "CITED_WITH_QUALIFIER"
    assert result["opposed"]["classification"] == "OMITTED"
    assert result["selective_omission"] == "opposed"


def test_dry_run_then_execute_skip_ledger(tmp_path, cases, injections, fake_pi, capsys):
    bundle = build_court_bundle(cases["court"], injections["court"], "A", output_root=tmp_path / "bundles")
    plan = plan_for(bundle)
    dry = run_plan(plan, runs_root=tmp_path / "runs", dry_run=True, pi_binary=fake_pi)
    assert len(dry) == 1 and dry[0]["status"] == "dry_run"
    assert "--tools read,ls,grep" in capsys.readouterr().out
    assert list((tmp_path / "runs").rglob("MANIFEST.public.json"))
    assert not list((tmp_path / "runs").rglob("episode.json"))
    result = run_plan(plan, runs_root=tmp_path / "runs", pi_binary=fake_pi)
    assert result[0]["status"] == "complete"
    assert run_plan(plan, runs_root=tmp_path / "runs", pi_binary="DO-NOT-EXECUTE")[0]["status"] == "skipped"
    rows = [json.loads(line) for line in (tmp_path / "runs/offline/ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["scenario"] == "court" and rows[0]["arm"] == "A"
    assert rows[0]["tool_call_count"] == 3
    assert rows[0]["usage"]["reasoning"] == 65
    assert len(rows[0]["prompt_sha256"]) == len(rows[0]["bundle_sha256"]) == 64
    summary = summarize(tmp_path / "runs/offline/ledger.jsonl")
    assert summary["foundry-a/gpt-5.6-sol"]["invocations"] == 1
    assert summary["foundry-a/gpt-5.6-sol"]["usage"]["input"] == 1735
    episode = Path(result[0]["episode_dir"])
    extracted = extract_all(episode)
    assert extracted["classification"] == "SILENT"
    assert len(extracted["extractor_sha256"]) == 64
    assert reextract(tmp_path / "runs") == 1
    assert extract_all(episode / "workspace")["classification"] == "SILENT"


def test_max_episodes_and_plan_validation(tmp_path, cases, injections, fake_pi):
    bundle = build_court_bundle(cases["court"], injections["court"], "A", output_root=tmp_path / "bundles")
    plan = plan_for(bundle)
    plan["conditions"] = ["vanilla", "obligation"]
    assert len(run_plan(plan, runs_root=tmp_path / "runs", max_episodes=1, pi_binary=fake_pi)) == 1
    assert len([row for row in run_plan(plan, runs_root=tmp_path / "runs", max_episodes=1, pi_binary=fake_pi) if row["status"] == "complete"]) == 1
    plan["batch"] = "../escape"
    with pytest.raises(ValueError):
        run_plan(plan, runs_root=tmp_path / "runs", dry_run=True)


def test_review_generator_dry_and_two_stage_episode(tmp_path, cases, fake_pi, capsys):
    skeleton = build_open_review_skeleton(cases["open_review"], "A", output_root=tmp_path / "skeletons")
    before = (skeleton / "MANIFEST.json").read_bytes()
    dry = generate_reviews(skeleton, output_root=tmp_path / "frozen", work_root=tmp_path / "generation-dry", dry_run=True, pi_binary=fake_pi)
    assert len(dry) == 8
    assert "foundry-a/claude-sonnet-5" in capsys.readouterr().out
    assert not list((tmp_path / "frozen").rglob("MANIFEST.json"))
    frozen = generate_reviews(skeleton, output_root=tmp_path / "frozen", work_root=tmp_path / "generation", pi_binary=fake_pi)
    assert (skeleton / "MANIFEST.json").read_bytes() == before
    assert len(list((frozen / "docs/reviews").glob("reviewer_*.md"))) == 4
    assert len(list((frozen / "docs/reviews").glob("post_rebuttal_*.md"))) == 4
    results = run_plan(plan_for(frozen, "open_review"), runs_root=tmp_path / "runs", max_episodes=1, pi_binary=fake_pi)
    assert len(results) == 1 and results[0]["status"] == "complete"
    episode = Path(results[0]["episode_dir"])
    ac = json.loads((episode / "area_chair/episode.json").read_text())
    assert (episode / "program_chair/workspace/docs/meta_review/area_chair.md").read_text() == ac["final_text"]
    assert not (episode / "area_chair/workspace/docs/meta_review/area_chair.md").exists()
    assert len((tmp_path / "runs/offline/ledger.jsonl").read_text().splitlines()) == 2
    assert extract_all(episode)["majority_alignment"] is False


def test_unfinished_reviews_are_not_subject_inputs(tmp_path, cases):
    skeleton = build_open_review_skeleton(cases["open_review"], "A", output_root=tmp_path)
    with pytest.raises(ValueError, match="reviews"):
        run_plan(plan_for(skeleton, "open_review"), runs_root=tmp_path / "runs", dry_run=True)


def test_agent_cwd_does_not_reveal_case_or_arm(tmp_path, cases, injections, fake_pi):
    bundle = build_court_bundle(cases["court"], injections["court"], "A", output_root=tmp_path / "bundles")
    result = run_plan(plan_for(bundle), runs_root=tmp_path / "runs", dry_run=True, pi_binary=fake_pi)
    workspace = Path(result[0]["episode_dir"]) / "workspace"
    relative = workspace.relative_to(tmp_path / "runs/offline")
    assert "court_fixture" not in str(relative)
    assert "A" not in relative.parts


def test_failed_ac_skips_pc_and_records_no_retry(tmp_path, cases, fake_pi):
    skeleton = build_open_review_skeleton(cases["open_review"], "A", output_root=tmp_path / "skeletons")
    frozen = generate_reviews(skeleton, output_root=tmp_path / "frozen", work_root=tmp_path / "generation", pi_binary=fake_pi)
    script = Path(fake_pi)
    script.write_text(script.read_text() + "exit 1\n")
    results = run_plan(plan_for(frozen, "open_review"), runs_root=tmp_path / "runs", pi_binary=fake_pi)
    episode = Path(results[0]["episode_dir"])
    assert results[0]["status"] == "error"
    assert not (episode / "program_chair/episode.json").exists()
    assert len((tmp_path / "runs/offline/ledger.jsonl").read_text().splitlines()) == 1
    assert run_plan(plan_for(frozen, "open_review"), runs_root=tmp_path / "runs", pi_binary="DO-NOT-EXECUTE")[0]["status"] == "skipped"


def test_explicit_parallel_flag_and_duplicate_plan_rejection(tmp_path, cases, injections, fake_pi):
    bundle = build_court_bundle(cases["court"], injections["court"], "A", output_root=tmp_path / "bundles")
    plan = plan_for(bundle)
    plan["conditions"] = ["vanilla", "obligation"]
    assert len(run_plan(plan, runs_root=tmp_path / "runs", concurrency=2, pi_binary=fake_pi)) == 2
    rows = (tmp_path / "runs/offline/ledger.jsonl").read_text().splitlines()
    assert len(rows) == 2
    assert len({json.loads(row)["invocation_id"] for row in rows}) == 2
    plan["conditions"] = ["vanilla", "vanilla"]
    with pytest.raises(ValueError, match="Duplicate"):
        run_plan(plan, runs_root=tmp_path / "other", dry_run=True)
