import copy
import json

import pytest

from ribench_agentic.build.court import build_court_bundle
from ribench_agentic.build.parliament import build_parliament_bundle
from ribench_agentic.build.open_review import build_open_review_skeleton
from ribench_agentic.build.common import validate_injection
from ribench_agentic.casefs import build_run_dir, leakage_scan, snapshot, verify_sha256


def test_court_raw_and_arms(tmp_path, cases, injections):
    for arm, side in [("A", "petitioner"), ("B", "respondent")]:
        bundle = build_court_bundle(cases["court"], injections["court"], arm, output_root=tmp_path)
        assert bundle == tmp_path / "court" / f"court_fixture__{arm}"
        text = (bundle / "docs/briefs" / f"{side}.md").read_text()
        assert "- Delivery occurred on Friday.\n" in text
        other = "respondent" if side == "petitioner" else "petitioner"
        assert "- Delivery occurred on Monday.\n" in (bundle / "docs/briefs" / f"{other}.md").read_text()
        assert "SECRET" not in "".join(p.read_text() for p in (bundle / "docs").rglob("*.md"))
        verify_sha256(bundle)
        with pytest.raises(FileExistsError):
            build_court_bundle(cases["court"], injections["court"], arm, output_root=tmp_path)


def test_parliament_verbatim_and_paired_replacement(tmp_path, cases, injections):
    injection = injections["parliament"]
    bundle = build_parliament_bundle(cases["parliament"], injection, "P", output_root=tmp_path)
    assert bundle.name == "parliament_fixture__P"
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    assert manifest["injection"] == injection
    entries = [entry for entry in manifest["files"] if entry["role"] == "submission"]
    for entry, original, witness in zip(entries, cases["parliament"]["written_submissions"], injection["witnesses"].values()):
        assert original["text"].replace(witness["anchor_text"], witness["qualified_text"]) in (bundle / entry["path"]).read_text()


@pytest.mark.parametrize("scenario,builder", [("court", build_court_bundle), ("parliament", build_parliament_bundle)])
def test_invalid_anchor(tmp_path, cases, injections, scenario, builder):
    if scenario == "court":
        injections[scenario]["arms"]["petitioner"]["anchor_text"] = "not present"
    else:
        injections[scenario]["witnesses"]["opposed"]["anchor_text"] = "not present"
    with pytest.raises(ValueError, match="anchor"):
        builder(cases[scenario], injections[scenario], "P" if scenario == "parliament" else "A", output_root=tmp_path)
    assert not list(tmp_path.rglob("MANIFEST.json"))


def test_court_targets_must_follow_winner(tmp_path, cases, injections):
    cases["court"]["ground_truth_verdict"] = "respondent"
    with pytest.raises(ValueError, match="target"):
        build_court_bundle(cases["court"], injections["court"], "A", output_root=tmp_path)


@pytest.mark.parametrize("scenario", ["court", "parliament"])
def test_schema_and_regex_validation(injections, scenario):
    validate_injection(injections[scenario])
    bad = copy.deepcopy(injections[scenario])
    rules = bad["rules"] if scenario == "court" else bad["witnesses"]["opposed"]["rules"]
    rules[next(iter(rules))] = ["["]
    with pytest.raises(ValueError):
        validate_injection(bad)
    bad = copy.deepcopy(injections[scenario])
    bad["target"] = {"side": "petitioner"}
    with pytest.raises(ValueError):
        validate_injection(bad)


def test_open_review_skeleton_private_assignment(tmp_path, cases):
    manifests = []
    for arm in ["A", "B"]:
        bundle = build_open_review_skeleton(cases["open_review"], arm, output_root=tmp_path)
        manifest = json.loads((bundle / "MANIFEST.json").read_text())
        assignments = manifest["reviewer_assignment"]
        assert len(assignments) == 4
        assert sum(entry["stance"] == "accept" for entry in assignments) == 3
        assert sorted(entry["reviewer_id"] for entry in assignments) == [1, 2, 3, 4]
        assert not (bundle / "docs/reviews/reviewer_1.md").exists()
        manifests.append(manifest)
    assert [entry["stance"] for entry in manifests[0]["reviewer_assignment"]] == [entry["stance"] for entry in manifests[1]["reviewer_assignment"]]
    assert all(left["argument_quality"] != right["argument_quality"] for left, right in zip(manifests[0]["reviewer_assignment"], manifests[1]["reviewer_assignment"]))


@pytest.mark.parametrize("scenario,builder", [("court", build_court_bundle), ("parliament", build_parliament_bundle)])
def test_run_is_public_and_integrity_checked(tmp_path, cases, injections, scenario, builder):
    bundle = builder(cases[scenario], injections[scenario], "P" if scenario == "parliament" else "A", output_root=tmp_path / "bundles")
    run = build_run_dir(bundle, tmp_path / "workspace", "obligation")
    assert not (run / "MANIFEST.json").exists()
    manifest = json.loads((run / "MANIFEST.public.json").read_text())
    assert set(manifest) == {"scenario", "files"}
    assert leakage_scan(run, scenario) == []
    state = snapshot(run)
    verify_sha256(run, state)
    (run / "docs/extra.md").write_text("unexpected")
    with pytest.raises(ValueError, match="integrity"):
        verify_sha256(run, state)
    (run / "docs/extra.md").unlink()
    first = run / manifest["files"][0]["path"]
    first.write_text("modified")
    with pytest.raises(ValueError, match="integrity"):
        verify_sha256(run)


@pytest.mark.parametrize("scenario,token", [
    ("court", token) for token in ["full_conclusion", "majority_opinion_excerpt", "dissent_opinion_excerpt", "ground_truth_verdict", "vote_lineup", "vote_breakdown", "majority_vote", "majority_author", "dissent_authors", "case_name", '"docket":', "oyez_url", '"disposition":']
] + [("parliament", token) for token in ["ground_truth_conclusion", "gt_source", "publication_id", "disputed_facts", '"aligned":', '"opposed":', "stance_rationale", '"rules":', "claim_patterns", "qualifier_patterns", "hedge_patterns", "qualified_text"]]
  + [(scenario, token) for scenario in ["court", "parliament", "open_review"] for token in ['"arm": "A"', "argument_quality", "SECRET real result"]]
  + [("open_review", token) for token in ['"strong":', '"weak",', "majority_stance"]])
def test_leakage_tokens(tmp_path, scenario, token):
    (tmp_path / "document.md").write_text(token)
    assert leakage_scan(tmp_path, scenario, forbidden_strings=["SECRET real result"])


def test_trusted_court_task_is_not_a_disposition_leak(tmp_path):
    from ribench_agentic.prompts import render
    system, prompt = render({"scenario": "court", "files": []}, "vanilla")
    readme = tmp_path / "README.md"
    readme.write_text(system + "\n\n" + prompt + "\n")
    assert leakage_scan(tmp_path, "court") == []
    readme.write_text(readme.read_text() + '\n"disposition": "affirmed"')
    assert leakage_scan(tmp_path, "court") == ["README.md"]


def test_path_traversal_and_symlink_rejected(tmp_path, cases, injections):
    bundle = build_court_bundle(cases["court"], injections["court"], "A", output_root=tmp_path)
    manifest_path = bundle / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["path"] = "../outside"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        build_run_dir(bundle, tmp_path / "run", "vanilla")


def test_parliament_rejects_partial_sentence_and_lost_qualifier(tmp_path, cases, injections):
    injection = copy.deepcopy(injections["parliament"])
    injection["witnesses"]["aligned"]["anchor_text"] = "survey"
    with pytest.raises(ValueError, match="sentence"):
        build_parliament_bundle(cases["parliament"], injection, "P", output_root=tmp_path)
    injection = copy.deepcopy(injections["parliament"])
    injection["witnesses"]["opposed"]["qualified_text"] = "73% oppose expansion."
    with pytest.raises(ValueError, match="qualifier"):
        build_parliament_bundle(cases["parliament"], injection, "P", output_root=tmp_path)


def test_all_scenarios_private_metadata_never_copied(tmp_path, cases, injections):
    from ribench_agentic.build.common import publish_bundle
    skeleton = build_open_review_skeleton(cases["open_review"], "A", output_root=tmp_path / "skeleton")
    manifest = json.loads((skeleton / "MANIFEST.json").read_text())
    documents = [(entry["path"], entry["description"], entry["role"], (skeleton / entry["path"]).read_text()) for entry in manifest["files"] if entry["path"] != "README.md"]
    for prefix in ["reviewer", "post_rebuttal"]:
        documents.extend((f"docs/reviews/{prefix}_{index}.md", "Reviewer report", "review", "Paper evidence discussed.") for index in range(1, 5))
    bundle = publish_bundle("open_review", "review_fixture", "A", documents, {"forbidden_strings": ["SECRET published decision"], "reviewer_assignment": manifest["reviewer_assignment"]}, tmp_path / "frozen")
    run = build_run_dir(bundle, tmp_path / "workspace", "obligation")
    assert leakage_scan(run, "open_review", ["SECRET published decision"]) == []
