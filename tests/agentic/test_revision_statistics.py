"""Offline regression tests for revision statistics and complete per-case exports."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("revision_statistics", ROOT / "tools" / "revision_statistics.py")
statistics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(statistics)
sys.modules.setdefault("revision_statistics", statistics)
APPENDIX_SPEC = importlib.util.spec_from_file_location("make_percase_appendix", ROOT / "tools" / "make_percase_appendix.py")
appendix = importlib.util.module_from_spec(APPENDIX_SPEC)
APPENDIX_SPEC.loader.exec_module(appendix)


def test_repeated_cluster_width_does_not_shrink_with_replication():
    frame = [f"case-{index:02d}" for index in range(30)]
    records = [(case, int(index < 15), 1, 1, 1) for index, case in enumerate(frame)]
    original = statistics.cluster_ratio(frame, records)
    repeated = statistics.cluster_ratio(frame, records * 20)
    independent = statistics.cluster_ratio([f"episode-{index}" for index in range(600)],
                                          [(f"episode-{index}", record[1], 1, 1, 1) for index, record in enumerate(records * 20)])
    assert original["ci95_case_cluster"] == repeated["ci95_case_cluster"]
    assert np.ptp(repeated["ci95_case_cluster"]) > 2 * np.ptp(independent["ci95_case_cluster"])


def test_local_rng_calls_and_row_order_are_stable():
    frame = ["alpha", "beta", "gamma"]
    records = [("beta", 1, 2, 2, 2), ("alpha", 0, 0, 1, 1), ("gamma", 2, 3, 3, 3)]
    np.random.seed(123)
    state = np.random.get_state()
    first = statistics.cluster_ratio(frame, records)
    statistics.cluster_ratio(["unrelated"], [("unrelated", 0, 1, 1, 1)], seed=6)
    second = statistics.cluster_ratio(frame[::-1], records[::-1])
    assert first == second
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_empty_ratios_and_undefined_draws_are_not_zero():
    empty = statistics.cluster_ratio([], [])
    zero = statistics.cluster_ratio(["zero"], [("zero", 0, 0, 1, 1)])
    for value in (empty, zero):
        assert value["rate"] is None
        assert value["ci95_case_cluster"] == [None, None]
        assert value["draws_undefined"] == 10000
    mixed = statistics.cluster_ratio(["zero", "cited"], [("zero", 0, 0, 2, 2), ("cited", 1, 1, 2, 2)])
    assert mixed["n_cases"] == 2 and mixed["n_cases_zero_denominator"] == 1
    assert mixed["draws_undefined"] > 2000
    assert mixed["ci95_case_cluster"] == [1, 1]
    assert mixed["n_valid"] == 4 and mixed["n"] == 1


def test_conditional_contrast_retains_all_cited_zero_cases():
    frame = ["both-zero", "left-only", "right-only"]
    left = [("both-zero", 0, 0, 2, 2), ("left-only", 1, 1, 2, 2), ("right-only", 0, 0, 2, 2)]
    right = [("both-zero", 0, 0, 2, 2), ("left-only", 0, 0, 2, 2), ("right-only", 0, 1, 2, 2)]
    effect = statistics.cluster_difference(frame, left, right)
    assert effect["n_cases"] == 3 and effect["draws_undefined"] > 0
    assert effect["estimate"] == 1 and effect["ci95_case_cluster"] == [1, 1]
    same = statistics.cluster_difference(frame, left, left)
    assert same["estimate"] == 0 and same["ci95_case_cluster"] == [0, 0]


def decision(case, value, reference="accept"):
    return dict(case_id=case, final=dict(decision=value, truth=reference, majority_stance=reference))


def test_valid_pair_accounting_and_signed_movement():
    left = {("positive", "model"): decision("positive", "accept"),
            ("negative", "model"): decision("negative", "reject"),
            ("unclear", "model"): decision("unclear", "unclear"),
            ("absent", "model"): decision("absent", "accept")}
    right = {("positive", "model"): decision("positive", "reject"),
             ("negative", "model"): decision("negative", "accept"),
             ("unclear", "model"): decision("unclear", "reject")}
    records, transitions = statistics.paired_records(left, right, "away_historical")
    value = statistics.cluster_ratio([key[0] for key in left], records)
    assert transitions == dict(positive=1, negative=1, unchanged=0, missing=2)
    assert (value["n_valid"], value["n_missing"], value["n_planned"]) == (2, 2, 4)
    assert value["rate"] == 0


def test_trace_missingness_not_zero_fill():
    value = appendix.describe([None, 2, 6, None], 4)
    assert value == dict(n_valid=2, n_missing=2, n_planned=4, q1=3, median=4, q3=5)
    assert appendix.describe([None], 1)["median"] is None


def test_reference_mismatch_invalidates_a_pair():
    left = {("paper", "model"): decision("paper", "accept", "accept")}
    right = {("paper", "model"): decision("paper", "reject", "reject")}
    for metric in ("minority", "away_historical"):
        records, transitions = statistics.paired_records(left, right, metric)
        assert records == [("paper", 0, 0, 0, 1)]
        assert transitions["missing"] == 1


def test_all_scenario_input_permutations_are_stable():
    rows = [dict(case_id=f"case-{case}", model=model, arm=arm, condition=condition,
                 final=dict(decision="unclear" if case == 0 and arm == "X" else "accept", truth="accept",
                            majority_stance="accept" if arm == "C" else "reject"))
            for case in range(3) for model in ("model-1", "model-2") for arm in ("C", "X") for condition in statistics.CONDS]
    first = statistics.build_case_cluster({"open_review_v3": rows})
    second = statistics.build_case_cluster({"open_review_v3": rows[::-1]})
    assert first == second


def test_percase_counts_reconcile_to_summary_and_role_raw_export():
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    cases = json.loads((ROOT / "results" / "per_case.json").read_text())
    for condition in statistics.CONDS:
        court = [entry for entry in cases if entry["stratum"] == "court"]
        for category in statistics.COURT:
            numerator = sum(cell["categories"][category]["k"] for entry in court for cell in entry["conditions"][condition].values())
            assert numerator == summary["case_cluster"]["court"]["pooled"]["rates"][condition][f"both/{category}"]["k"]
        parliament = [entry for entry in cases if entry["stratum"] == "parliament"]
        for key in ("k", "n"):
            total = sum(cell["ql"][key] for entry in parliament for cell in entry["conditions"][condition].values())
            assert total == summary["case_cluster"]["parliament"]["pooled"]["rates"][condition]["both/ql"][key]
    traces = json.loads((ROOT / "results" / "trace_statistics_raw.json").read_text())
    groups = json.loads((ROOT / "results" / "trace_statistics.json").read_text())
    assert len(traces) == 16408 and len(groups) == 38
    for group in groups:
        selected = [row for row in traces if all(row[key] == group[key] for key in ("scenario", "model", "role"))]
        for metric, expected in group["metrics"].items():
            assert appendix.describe([row[metric] for row in selected], len(selected)) == expected


def test_case_unknown_never_counts_as_resistance():
    rows = [dict(case_id="synthetic", model=model, arm=arm, condition=condition,
                 final=dict(decision="unclear" if model == "unknown" else "reject", majority_stance="reject", truth="accept"))
            for model in ("known", "unknown") for arm in ("C", "X") for condition in statistics.CONDS]
    entry = appendix.case_export("open_review_v3", "synthetic", rows, ["known", "unknown"])
    cell = entry["conditions"]["vanilla"]["X"]
    assert cell["following"]["k"] == cell["following"]["n"] == 1
    assert (cell["n_valid"], cell["n_unclear_or_missing"], cell["n_planned"]) == (1, 1, 2)


def test_frozen_valid_pair_counts_and_all_477_cases():
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    v2 = summary["case_cluster"]["open_review_v2"]["pooled"]["effects"]["vanilla/A_to_B/minority"]
    assert (v2["n_valid"], v2["n_missing"], v2["n_planned"]) == (594, 1, 595)
    cases = json.loads((ROOT / "results" / "per_case.json").read_text())
    assert len(cases) == len({(entry["stratum"], entry["case_id"]) for entry in cases}) == 477
    for scenario, expected in (("court", 120), ("open_review_v2", 119), ("open_review_v3", 118), ("parliament", 120)):
        assert sum(entry["stratum"] == scenario for entry in cases) == expected
    for model, entry in summary["case_cluster"]["parliament"].items():
        for condition in statistics.CONDS:
            assert entry["rates"][condition]["both/ql"]["n_cases"] == 120


def test_figures_use_summary_intervals_and_exact_numerators():
    # the paper's figures are drawn from the eight-model revision view (tools/build_revision8.sh)
    summary = json.loads((ROOT / "results" / "revision8" / "summary.json").read_text())
    figures = json.loads((ROOT / "manuscript" / "figures" / "fig_numbers.json").read_text())
    for name, scenario, metric in (("follow_adversarial_majority", "open_review_v3", "X/following"),
                                   ("court_adopted", "court", "both/ADOPTED"),
                                   ("parliament_limitation_loss", "parliament", "both/ql")):
        for key, value in figures["overview_case_cluster"][name].items():
            model, condition = key.split("|")
            assert value == summary["case_cluster"][scenario][model]["rates"][condition][metric]
            assert value["rate"] == pytest.approx(value["k"] / value["n"])
