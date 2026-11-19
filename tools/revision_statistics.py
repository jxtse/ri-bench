"""Deterministic whole-case inference from frozen episode exports; no model calls."""
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

SEED = 20260914
DRAWS = 10000
CONDS = ("vanilla", "obligation")
COURT = ("ADOPTED", "CORRECTED", "TRUTH_STATED", "SILENT")
PARLIAMENT = ("CITED_WITH_QUALIFIER", "CITED_WITHOUT_QUALIFIER", "OMITTED")


@lru_cache(maxsize=16)
def weights(size, draws=DRAWS, seed=SEED):
    result = np.random.default_rng(seed).multinomial(size, np.full(size, 1 / size), size=draws)
    result.flags.writeable = False
    return result


def aggregate(frame, records):
    totals = defaultdict(lambda: [0, 0, 0, 0])
    for case_id, numerator, denominator, valid, planned in records:
        for index, value in enumerate((numerator, denominator, valid, planned)):
            totals[case_id][index] += value
    if set(totals) - set(frame):
        raise ValueError("Records outside the case sampling frame")
    return np.array([totals[case_id] for case_id in frame], dtype=np.int64).reshape((-1, 4))


def counts(values):
    numerator, denominator, valid, planned = map(int, values.sum(axis=0))
    return dict(k=numerator, n=denominator, rate=numerator / denominator if denominator else None,
                n_valid=valid, n_missing=planned - valid, n_planned=planned)


def interval(samples, draws):
    defined = samples[np.isfinite(samples)]
    return dict(ci95_case_cluster=np.quantile(defined, [0.025, 0.975]).tolist() if len(defined) else [None, None],
                draws_defined=len(defined), draws_undefined=draws - len(defined))


def sample_ratios(values, draws, seed):
    if not len(values):
        return np.full(draws, np.nan)
    sampled = weights(len(values), draws, seed) @ values[:, :2]
    return np.divide(sampled[:, 0], sampled[:, 1], out=np.full(draws, np.nan), where=sampled[:, 1] != 0)


def cluster_ratio(case_ids, records, draws=DRAWS, seed=SEED):
    frame = sorted(set(case_ids))
    values = aggregate(frame, records)
    return dict(**counts(values), **interval(sample_ratios(values, draws, seed), draws),
                n_cases=len(frame), n_cases_zero_denominator=int(np.sum(values[:, 1] == 0)))


def cluster_difference(case_ids, left, right, draws=DRAWS, seed=SEED):
    frame = sorted(set(case_ids))
    left_values, right_values = aggregate(frame, left), aggregate(frame, right)
    left_counts, right_counts = counts(left_values), counts(right_values)
    estimate = (left_counts["rate"] - right_counts["rate"]
                if left_counts["rate"] is not None and right_counts["rate"] is not None else None)
    return dict(estimate=estimate, left=left_counts, right=right_counts, n_cases=len(frame),
                **interval(sample_ratios(left_values, draws, seed) - sample_ratios(right_values, draws, seed), draws))


def outcome(row, metric):
    final = row.get("final")
    if metric in COURT:
        return int(final == metric) if final in COURT else None
    if not isinstance(final, dict) or final.get("decision") not in ("accept", "reject"):
        return None
    if metric == "accept":
        return int(final["decision"] == "accept")
    if metric in ("historical_agreement", "away_historical"):
        reference = final.get("truth", final.get("iclr_decision"))
        if reference not in ("accept", "reject"):
            return None
        agreement = final["decision"] == reference
        return int(not agreement if metric == "away_historical" else agreement)
    stance = final.get("majority_stance")
    if stance not in ("accept", "reject"):
        return None
    following = final["decision"] == stance
    return int(not following if metric == "minority" else following)


def binary_records(rows, metric):
    result = []
    for row in rows:
        value = outcome(row, metric)
        result.append((row["case_id"], value if value is not None else 0, int(value is not None), int(value is not None), 1))
    return result


def witness_records(rows, sides, metric):
    result = []
    for row in rows:
        for side in sides:
            label = (row.get("final") or {}).get(side)
            valid = label in PARLIAMENT
            cited = valid and label != "OMITTED"
            numerator = cited if metric == "citation" else label == "CITED_WITHOUT_QUALIFIER"
            denominator = valid if metric == "citation" else cited
            result.append((row["case_id"], int(numerator), int(denominator), int(valid), 1))
    return result


def paired_records(left, right, metric, right_metric=None):
    records, transitions = [], Counter()
    for key in sorted(set(left) | set(right)):
        before = outcome(left[key], metric) if key in left else None
        after = outcome(right[key], right_metric or metric) if key in right else None
        valid = before is not None and after is not None
        if valid and metric in ("minority", "away_historical", "historical_agreement", "following"):
            left_final, right_final = left[key]["final"], right[key]["final"]
            if metric in ("minority", "following"):
                valid = left_final.get("majority_stance") == right_final.get("majority_stance")
            else:
                valid = left_final.get("truth", left_final.get("iclr_decision")) == right_final.get("truth", right_final.get("iclr_decision"))
        change = after - before if valid else 0
        records.append((key[0], change, int(valid), int(valid), 1))
        transitions["positive" if change > 0 else "negative" if change < 0 else "unchanged" if valid else "missing"] += 1
    return records, {key: transitions[key] for key in ("positive", "negative", "unchanged", "missing")}


def build_case_cluster(datasets):
    result = {"method": {"unit": "entire case across models, arms, prompts and witnesses", "draws": DRAWS,
                         "seed": SEED, "interval": "percentile 95%; numpy linear quantiles",
                         "undefined_draws": "excluded, never zero; counts disclosed",
                         "count_fields": "k/n is the unrounded ratio; n_valid counts classified units (including omitted witnesses), n_missing counts unclassified units, n_planned includes both; paired k is the signed change sum",
                         "contrasts": "left ratio minus right ratio with shared case weights and separately resampled denominators; not a mean of case ratios",
                         "legacy": "Outside case_cluster, point counts are legacy; McNemar p values are descriptive, not independent-case inference."}}
    for scenario, rows in datasets.items():
        frame = sorted({row["case_id"] for row in rows})
        models = sorted({row["model"] for row in rows})
        result[scenario] = {}
        for model in models + ["pooled"]:
            selected = [row for row in rows if model == "pooled" or row["model"] == model]
            entry = {"rates": {}, "effects": {}}
            stored = {}

            def add_rate(condition, metric, records):
                stored[condition, metric] = records
                entry["rates"].setdefault(condition, {})[metric] = cluster_ratio(frame, records)

            def add_pair(name, left, right, metric):
                records, transitions = paired_records(left, right, metric)
                estimate = cluster_ratio(frame, records)
                entry["effects"][name] = dict(**estimate, estimate=estimate["rate"], signed_sum=estimate["k"], transitions=transitions,
                                               direction="right minus left; valid decision pairs only")

            for condition in (*CONDS, "all"):
                current = [row for row in selected if condition == "all" or row["condition"] == condition]
                if scenario == "court":
                    for arm in ("both", "A", "B"):
                        arm_rows = [row for row in current if arm == "both" or row["arm"] == arm]
                        for metric in COURT:
                            add_rate(condition, f"{arm}/{metric}", binary_records(arm_rows, metric))
                    for metric in ("ADOPTED", "CORRECTED"):
                        left = {(row["case_id"], row["model"], row["condition"]): row for row in current if row["arm"] == "A"}
                        right = {(row["case_id"], row["model"], row["condition"]): row for row in current if row["arm"] == "B"}
                        add_pair(f"{condition}/B_minus_A/{metric}", left, right, metric)
                elif scenario.startswith("open_review"):
                    arms = ("A", "B") if scenario.endswith("v2") else ("C", "X")
                    for arm in arms:
                        arm_rows = [row for row in current if row["arm"] == arm]
                        for metric in ("accept", "following", "historical_agreement"):
                            add_rate(condition, f"{arm}/{metric}", binary_records(arm_rows, metric))
                        if arm == "X":
                            for reference in ("accept", "reject"):
                                split = [row for row in arm_rows if row["final"].get("truth") == reference]
                                split_frame = sorted({row["case_id"] for row in split})
                                entry["rates"][condition][f"X/following/reference_{reference}"] = cluster_ratio(split_frame, binary_records(split, "following"))
                    left = {(row["case_id"], row["model"], row["condition"]): row for row in current if row["arm"] == arms[0]}
                    right = {(row["case_id"], row["model"], row["condition"]): row for row in current if row["arm"] == arms[1]}
                    metric = "minority" if scenario.endswith("v2") else "away_historical"
                    add_pair(f"{condition}/{arms[0]}_to_{arms[1]}/{metric}", left, right, metric)
                else:
                    for side in ("aligned", "opposed", "both"):
                        sides = ("aligned", "opposed") if side == "both" else (side,)
                        for metric in ("citation", "ql"):
                            add_rate(condition, f"{side}/{metric}", witness_records(current, sides, metric))
                    for metric in ("citation", "ql"):
                        entry["effects"][f"{condition}/aligned_minus_opposed/{metric}"] = cluster_difference(
                            frame, stored[condition, f"aligned/{metric}"], stored[condition, f"opposed/{metric}"])
            for metric in entry["rates"]["vanilla"]:
                if ("vanilla", metric) in stored:
                    entry["effects"][f"obligation_minus_vanilla/{metric}"] = cluster_difference(
                        frame, stored["obligation", metric], stored["vanilla", metric])
            if scenario != "parliament":
                arms = ("both", "A", "B") if scenario == "court" else (("A", "B") if scenario.endswith("v2") else ("C", "X"))
                metrics = ("ADOPTED", "CORRECTED") if scenario == "court" else ("following", "historical_agreement")
                for arm in arms:
                    current = [row for row in selected if arm == "both" or row["arm"] == arm]
                    left = {(row["case_id"], row["model"], row["arm"]): row for row in current if row["condition"] == "vanilla"}
                    right = {(row["case_id"], row["model"], row["arm"]): row for row in current if row["condition"] == "obligation"}
                    for metric in metrics:
                        add_pair(f"paired_obligation_minus_vanilla/{arm}/{metric}", left, right, metric)
            result[scenario][model] = entry
    return result
