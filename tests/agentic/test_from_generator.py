import copy
import json
from pathlib import Path

import pytest

from ribench_agentic.build.court import build_court_bundle
from ribench_agentic.build.parliament import build_parliament_bundle
from ribench_agentic.build.common import validate_injection
from ribench_agentic.casefs import build_run_dir, leakage_scan
from ribench_agentic.cli import main


FIXTURES = Path(__file__).parent / "fixtures"
CASES = Path(__file__).resolve().parents[2] / "data/cases"


def record_for(scenario):
    return json.loads((FIXTURES / f"generator_{scenario}.jsonl").read_text())


def convert(record):
    from ribench_agentic.build.from_generator import to_injection
    return to_injection(record)


@pytest.mark.parametrize("scenario", ["court", "parliament"])
def test_real_generator_scenario_arms(tmp_path, scenario):
    record = record_for(scenario)
    original = copy.deepcopy(record)
    injection = convert(record)
    assert record == original
    validate_injection(injection)
    case = json.loads(next(CASES.rglob(record["case_id"] + ".json")).read_text())
    builder = build_court_bundle if scenario == "court" else build_parliament_bundle
    for arm in (["P"] if scenario == "parliament" else ["A", "B"]):
        bundle = builder(case, injection, arm, output_root=tmp_path / "bundles")
        run = build_run_dir(bundle, tmp_path / arm, "obligation")
        assert leakage_scan(run, scenario) == []
        if scenario == "court":
            side = record["arm_side"][arm]
            assert injection["arms"][side]["replacement_text"] in (bundle / f"docs/briefs/{side}.md").read_text()
            other = record["arm_side"]["B" if arm == "A" else "A"]
            assert injection["arms"][other]["anchor_text"] in (bundle / f"docs/briefs/{other}.md").read_text()
        else:
            for witness in injection["witnesses"].values():
                assert any(witness["qualified_text"] in path.read_text() for path in (bundle / "docs/submissions").glob("*.md"))


@pytest.mark.parametrize("source", ["record", "verdict", "both"])
def test_court_winner_sources(tmp_path, cases, injections, source):
    if source == "record":
        del cases["court"]["ground_truth_verdict"]
    if source == "verdict":
        del injections["court"]["arm_side"]
    build_court_bundle(cases["court"], injections["court"], "A", output_root=tmp_path)


def test_court_duplicate_cleaned_anchor(tmp_path, cases, injections):
    cases["court"]["evidence_pro"].append("Delivery occurred on Monday.")
    with pytest.raises(ValueError, match="exactly one excerpt"):
        build_court_bundle(cases["court"], injections["court"], "A", output_root=tmp_path)


@pytest.mark.parametrize("use_id", [True, False])
def test_parliament_submission_identity_and_whitespace(tmp_path, cases, injections, use_id):
    case, injection = cases["parliament"], injections["parliament"]
    witness = injection["witnesses"]["aligned"]
    case["written_submissions"].reverse()
    case["written_submissions"][0]["submitter"] = "Other witness"
    original = "Before.\nA survey\n  was conducted.\t After."
    case["written_submissions"][1]["text"] = original
    witness["anchor_text"] = "A survey was\nconducted."
    if not use_id:
        del witness["submission_id"]
        witness["submitter"] = " transit   GROUP "
    bundle = build_parliament_bundle(case, injection, "P", output_root=tmp_path)
    expected = original.replace("A survey\n  was conducted.", witness["qualified_text"])
    assert expected in next((bundle / "docs/submissions").glob("002-*.md")).read_text()


@pytest.mark.parametrize("problem", ["missing", "ambiguous", "witness", "duplicate_anchor"])
def test_parliament_rejects_invalid_target(tmp_path, cases, injections, problem):
    case, injection = cases["parliament"], injections["parliament"]
    witness = injection["witnesses"]["aligned"]
    if problem == "missing":
        witness["submission_id"] = -1
    elif problem == "ambiguous":
        del witness["submission_id"]
        case["written_submissions"][1]["submitter"] = witness["submitter"]
    elif problem == "witness":
        witness["witness_index"] = 2
    else:
        case["written_submissions"][0]["text"] += " A survey\nwas conducted."
    with pytest.raises(ValueError):
        build_parliament_bundle(case, injection, "P", output_root=tmp_path)


@pytest.mark.parametrize("scenario", ["court", "parliament"])
def test_cli_builds_and_skips(tmp_path, capsys, scenario):
    record = record_for(scenario)
    inputs = tmp_path / "records.jsonl"
    inputs.write_text(json.dumps({"status": "error"}) + "\n" + json.dumps(record) + "\n")
    args = ["build-bundles", "--scenario", scenario, "--injections", str(inputs), "--cases-root", str(CASES), "--output-root", str(tmp_path / "out")]
    main(args)
    result = json.loads(capsys.readouterr().out)
    assert result["built"] == (1 if scenario == "parliament" else 2) and result["skipped_records"] == 1
    assert result["errors"] == []
    main(args)
    result = json.loads(capsys.readouterr().out)
    assert result["built"] == 0 and result["skipped_existing"] == (1 if scenario == "parliament" else 2)


@pytest.mark.parametrize("stance", ["aligned", "opposed"])
@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_each_anchor_failure_is_atomic(tmp_path, cases, injections, stance, problem):
    injection = injections["parliament"]
    witness = injection["witnesses"][stance]
    if problem == "missing":
        witness["anchor_text"] = "No such sentence."
    else:
        cases["parliament"]["written_submissions"][witness["witness_index"]]["text"] += " " + witness["anchor_text"]
    with pytest.raises(ValueError, match="anchor"):
        build_parliament_bundle(cases["parliament"], injection, "P", output_root=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("arm", ["A", "B"])
def test_parliament_builder_rejects_retired_arms(tmp_path, cases, injections, arm):
    with pytest.raises(ValueError, match="arm"):
        build_parliament_bundle(cases["parliament"], injections["parliament"], arm, output_root=tmp_path)


@pytest.mark.parametrize("problem", ["missing_design", "old_design", "missing_witness", "old_shape"])
def test_parliament_rejects_nonpaired_records(problem):
    record = record_for("parliament")
    if problem == "missing_design":
        del record["design"]
    elif problem == "old_design":
        record["design"] = "single_injection"
    elif problem == "missing_witness":
        del record["spec"]["witnesses"]["opposed"]
    else:
        record["spec"] = {"witness_index": 0, "variants": {"A": "Old.", "B": "Old."}}
    with pytest.raises((ValueError, KeyError)):
        convert(record)
