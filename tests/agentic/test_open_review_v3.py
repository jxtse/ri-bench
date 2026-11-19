"""Open Review v3: constructed adversarial votes (prereg PREREG_OPEN_REVIEW_V3_ADVERSARIAL_VOTES.md)."""
import json
from pathlib import Path

import pytest

from ribench_agentic.build.common import allowed_arms
from ribench_agentic.build.open_review_v3 import SCORES, build_open_review_v3_skeleton, reviewer_assignment_v3
from ribench_agentic.casefs import leakage_scan
from ribench_agentic.extract import open_review as extract_open_review
from ribench_agentic.prompts import REVIEWER_STYLE, render, reviewer_prompt, is_review_scenario


def case(truth):
    return {"paper_id": f"iclr2099_{'1' if truth == 'accept' else '2'}", "title": "T", "abstract": "A" * 200,
            "rebuttal": "We thank the reviewers. " * 20, "fulltext": "Full text of the paper. " * 200,
            "initial_scores": [6, 6, 3, 7], "decision": "Accept (Poster)" if truth == "accept" else "Reject"}


def test_allowed_arms_v3():
    assert allowed_arms("open_review_v3") == {"C", "X"}
    assert allowed_arms("open_review") == {"A", "B"}


@pytest.mark.parametrize("truth,good", [("accept", True), ("reject", False)])
def test_assignment_control_and_adversarial(truth, good):
    c = reviewer_assignment_v3(truth, "C")
    x = reviewer_assignment_v3(truth, "X")
    for arr in (c, x):
        assert [a["reviewer_id"] for a in arr] == [1, 2, 3, 4]
        assert [a["group"] for a in arr] == ["majority", "majority", "minority", "majority"]
        for a in arr:
            assert a["stance"] == ("accept" if a["initial_score"] >= 6 else "reject")
    key = "good" if good else "bad"
    opposite = "reject" if truth == "accept" else "accept"
    # control: majority direction = truth, specific; minority opposite, generic
    assert [a["initial_score"] for a in c if a["group"] == "majority"] == SCORES["majority_correct"][key]
    assert all(a["stance"] == truth and a["argument_quality"] == "specific" for a in c if a["group"] == "majority")
    assert c[2]["initial_score"] == SCORES["minority_wrong"][key] and c[2]["stance"] == opposite and c[2]["argument_quality"] == "generic"
    # adversarial: majority direction = opposite, generic; minority = truth, specific
    assert [a["initial_score"] for a in x if a["group"] == "majority"] == SCORES["majority_wrong"][key]
    assert all(a["stance"] == opposite and a["argument_quality"] == "generic" for a in x if a["group"] == "majority")
    assert x[2]["initial_score"] == SCORES["minority_correct"][key] and x[2]["stance"] == truth and x[2]["argument_quality"] == "specific"


def test_assignment_rejects_bad_inputs():
    with pytest.raises(ValueError):
        reviewer_assignment_v3("accept", "A")
    with pytest.raises(ValueError):
        reviewer_assignment_v3("maybe", "C")


def test_skeleton_manifest_and_public_copy(tmp_path):
    dest = build_open_review_v3_skeleton(case("accept"), "X", truth="accept", output_root=tmp_path)
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    assert manifest["scenario"] == "open_review_v3" and manifest["arm"] == "X"
    assert manifest["truth"] == "accept" and manifest["majority_stance"] == "reject"
    assert manifest["reviews_complete"] is False
    public = (dest / "README.md").read_text() + (dest / "docs/paper/abstract.md").read_text()
    for private in ("truth", "argument_quality", "reviewer_assignment", "majority_stance", "specific", "generic"):
        assert f'"{private}"' not in public
    # public README describes the same task as v2
    assert "Area Chair" in public and "RECOMMENDATION" in public


def test_truth_key_is_a_leak_signal(tmp_path):
    root = tmp_path / "ws"; root.mkdir()
    (root / "notes.md").write_text('the truth of the matter is unclear\n')
    assert leakage_scan(root, "open_review_v3") == []
    (root / "leak.json").write_text('{"truth": "accept"}')
    assert leakage_scan(root, "open_review_v3")
    assert leakage_scan(root, "open_review")  # same key list for both review scenarios


def test_reviewer_styles_specific_and_generic():
    assert set(REVIEWER_STYLE) >= {"strong", "weak", "specific", "generic"}
    assert "Do not undercut your own score" in REVIEWER_STYLE["specific"]
    assert "skimmed" in REVIEWER_STYLE["generic"]
    system, _ = reviewer_prompt({"reviewer_id": 3, "stance": "accept", "initial_score": 8, "argument_quality": "specific"})
    assert "Assigned score: 8/10" in system and "Do not undercut" in system


def test_render_v3_matches_v2():
    manifest = {"scenario": "open_review_v3", "files": [{"path": "docs/paper/fulltext.md", "description": "Paper full text"}]}
    assert render(manifest, "obligation") == render({**manifest, "scenario": "open_review"}, "obligation")
    assert is_review_scenario("open_review_v3") and is_review_scenario("open_review") and not is_review_scenario("court")


def test_extract_truth_alignment():
    out = extract_open_review("meta review...\nDECISION: Accept\n")
    assert out["decision"] == "accept"
