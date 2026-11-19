#!/usr/bin/env python3
"""Compute the manuscript's headline numbers from results/*.jsonl and write results/summary.json.
Every number in the paper's tables should be derivable from this file. Deterministic. Usage:
.venv/bin/python tools/summarize_results.py"""
import collections, json, math, os
from math import comb
from revision_statistics import build_case_cluster, cluster_ratio

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# RIBENCH_RESULTS_DIR points the same computation at a supplementary export (e.g. results/supplement/deepseek,
# written by tools/export_results_deepseek.py); the model list is then read from that directory's RUNS_INDEX.json.
# Without it the script is unchanged and must stay byte-identical on the frozen seven-model export.
RES = os.environ.get("RIBENCH_RESULTS_DIR") or os.path.join(R, "results")
ORDER = ["claude-opus-5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "glm-5.3", "glm-5.3-flash", "qwen3.5-35b-a3b"]
if os.environ.get("RIBENCH_RESULTS_DIR"):
    with open(os.path.join(RES, "RUNS_INDEX.json")) as f:
        _index = json.load(f)
    # a single-model supplement records "model"; the eight-model pooled view (results/pooled8) records "models"
    ORDER = _index["models"] if "models" in _index else [_index["model"]]
CONDS = ["vanilla", "obligation"]


def rows(scen):
    with open(os.path.join(RES, f"{scen}.jsonl")) as f:
        return [json.loads(l) for l in f]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def cluster_ci(items, B=10000):
    """Compatibility wrapper; corrected conditional-ratio frames live under case_cluster."""
    return cluster_ratio([case for case, value in items],
                         [(case, value, 1, 1, 1) for case, value in items], draws=B)["ci95_case_cluster"]


def pooled_rate(items):
    k = sum(v for _, v in items); n = len(items)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95_wilson": wilson(k, n), "ci95_case_cluster": cluster_ci(items)}


out: dict = {"models": ORDER, "conditions": CONDS}

# ---- Court: adoption of the planted error; correction; truth stated; silent
court = rows("court")
labels = ["ADOPTED", "CORRECTED", "TRUTH_STATED", "SILENT"]
out["court"] = {"n_episodes": len(court), "per_model": {}, "pooled": {}}
for m in ORDER + ["pooled"]:
    for cond in CONDS:
        sel = [r for r in court if (m == "pooled" or r["model"] == m) and r["condition"] == cond]
        c = collections.Counter(r["final"] for r in sel)
        ver = sum(1 for r in sel if (r.get("scorer") or {}).get("status") == "verified")
        entry = {lab: rate(c.get(lab, 0), len(sel)) for lab in labels}
        if m == "pooled":
            entry = {lab: pooled_rate([(r["case_id"], int(r["final"] == lab)) for r in sel]) for lab in labels}
        entry["scorer_verified"] = rate(ver, len(sel))
        entry["by_arm"] = {arm: dict(collections.Counter(r["final"] for r in sel if r["arm"] == arm)) for arm in ("A", "B")}
        (out["court"]["pooled"] if m == "pooled" else out["court"]["per_model"].setdefault(m, {}))[cond] = entry
    if m != "pooled":
        # paired Vanilla->Obligation change in CORRECTED, by case x arm
        van = {(r["case_id"], r["arm"]): r["final"] for r in court if r["model"] == m and r["condition"] == "vanilla"}
        obl = {(r["case_id"], r["arm"]): r["final"] for r in court if r["model"] == m and r["condition"] == "obligation"}
        up = sum(1 for k in van if van[k] != "CORRECTED" and obl.get(k) == "CORRECTED")
        down = sum(1 for k in van if van[k] == "CORRECTED" and obl.get(k) not in (None, "CORRECTED"))
        out["court"]["per_model"][m]["corrected_vanilla_to_obligation"] = {"gained": up, "lost": down, "mcnemar_exact_p": round(mcnemar_exact(up, down), 4)}
adopt_total = sum(1 for r in court if r["final"] == "ADOPTED")
out["court"]["adoptions_total"] = adopt_total
out["court"]["adoptions_by_arm"] = dict(collections.Counter(r["arm"] for r in court if r["final"] == "ADOPTED"))
out["court"]["adoptions_small_models"] = sum(1 for r in court if r["final"] == "ADOPTED" and r["model"] in ("glm-5.3", "glm-5.3-flash", "qwen3.5-35b-a3b"))

# ---- Open Review v2 (real votes; arm A = majority well-argued, arm B = minority well-argued)
v2 = rows("open_review_v2")
# real ICLR decision is frozen into each v2 row by the exporter (final.iclr_decision); no external file needed
for r in v2:
    t_ = r["final"].get("iclr_decision")
    r["final"]["iclr_alignment"] = (r["final"].get("decision") == t_) if (t_ and r["final"].get("decision") in ("accept", "reject")) else None
missing_iclr = sum(1 for r in v2 if "iclr_decision" not in r["final"])
if missing_iclr:
    raise SystemExit(f"{missing_iclr} v2 rows lack final.iclr_decision; re-run tools/export_results.py")
out["open_review_v2"] = {"n_episodes": len(v2), "per_model": {}, "pooled": {}}
for m in sorted({r["model"] for r in v2}) + ["pooled"]:
    for cond in CONDS:
        sel = [r for r in v2 if (m == "pooled" or r["model"] == m) and r["condition"] == cond]
        A = {(r["model"], r["case_id"]): r["final"] for r in sel if r["arm"] == "A"}
        B = {(r["model"], r["case_id"]): r["final"] for r in sel if r["arm"] == "B"}
        pairs = [(A[c], B[c]) for c in A if c in B and A[c].get("decision") in ("accept", "reject") and B[c].get("decision") in ("accept", "reject")]
        to_min = sum(1 for a, b in pairs if a["decision"] != b["decision"] and b.get("majority_alignment") is False)
        to_maj = sum(1 for a, b in pairs if a["decision"] != b["decision"] and b.get("majority_alignment") is True)
        same = sum(1 for a, b in pairs if a["decision"] == b["decision"])
        icl = [r["final"].get("iclr_alignment") for r in sel if r["final"].get("iclr_alignment") is not None]
        entry = {"pairs": len(pairs), "toward_minority": to_min, "toward_majority": to_maj, "unchanged": same,
                 "mcnemar_exact_p": round(mcnemar_exact(to_min, to_maj), 4),
                 "agreement_with_iclr_outcome": rate(sum(icl), len(icl))}
        (out["open_review_v2"]["pooled"] if m == "pooled" else out["open_review_v2"]["per_model"].setdefault(m, {}))[cond] = entry

# ---- Open Review v3 (constructed votes; arm C control, arm X adversarial)
v3 = rows("open_review_v3")
out["open_review_v3"] = {"n_episodes": len(v3), "per_model": {}, "pooled": {}}
for m in ORDER + ["pooled"]:
    for cond in CONDS:
        sel = [r for r in v3 if (m == "pooled" or r["model"] == m) and r["condition"] == cond]
        C = {(r["model"], r["case_id"]): r["final"] for r in sel if r["arm"] == "C"}
        X = {(r["model"], r["case_id"]): r["final"] for r in sel if r["arm"] == "X"}
        cf = [f["majority_alignment"] for f in C.values() if f.get("majority_alignment") is not None]
        xf = [f["majority_alignment"] for f in X.values() if f.get("majority_alignment") is not None]
        pairs = [(C[c], X[c]) for c in C if c in X and C[c].get("majority_alignment") is not None and X[c].get("majority_alignment") is not None]
        flip = sum(1 for c_, x_ in pairs if c_["majority_alignment"] and x_["majority_alignment"])      # correct under C, wrong under X
        rev = sum(1 for c_, x_ in pairs if not c_["majority_alignment"] and not x_["majority_alignment"])
        # H3 asymmetry: X-arm following split by true outcome
        acc = [f["majority_alignment"] for f in X.values() if f.get("majority_alignment") is not None and f.get("truth") == "accept"]
        rej = [f["majority_alignment"] for f in X.values() if f.get("majority_alignment") is not None and f.get("truth") == "reject"]
        entry = {"control_agreement": rate(sum(cf), len(cf)), "adversarial_following": rate(sum(xf), len(xf)),
                 **({"control_agreement": pooled_rate([(k[1], int(f["majority_alignment"])) for k, f in C.items() if f.get("majority_alignment") is not None]),
                     "adversarial_following": pooled_rate([(k[1], int(f["majority_alignment"])) for k, f in X.items() if f.get("majority_alignment") is not None])} if m == "pooled" else {}),
                 "pairs": len(pairs), "flipped_to_wrong": flip, "reversed_to_right": rev,
                 "following_by_true_outcome": {"accept": rate(sum(acc), len(acc)), "reject": rate(sum(rej), len(rej))}}
        (out["open_review_v3"]["pooled"] if m == "pooled" else out["open_review_v3"]["per_model"].setdefault(m, {}))[cond] = entry
    if m != "pooled":
        van = {r["case_id"]: r["final"].get("majority_alignment") for r in v3 if r["model"] == m and r["condition"] == "vanilla" and r["arm"] == "X"}
        obl = {r["case_id"]: r["final"].get("majority_alignment") for r in v3 if r["model"] == m and r["condition"] == "obligation" and r["arm"] == "X"}
        better = sum(1 for c in van if van[c] is True and obl.get(c) is False)
        worse = sum(1 for c in van if van[c] is False and obl.get(c) is True)
        out["open_review_v3"]["per_model"][m]["obligation_effect_adversarial"] = {"improved": better, "worsened": worse, "mcnemar_exact_p": round(mcnemar_exact(better, worse), 4)}

# ---- Parliament: limitation loss conditional on citation, by witness alignment
parl = rows("parliament")
out["parliament"] = {"n_episodes": len(parl), "per_model": {}, "pooled": {}}
for m in ORDER + ["pooled"]:
    for cond in CONDS:
        sel = [r for r in parl if (m == "pooled" or r["model"] == m) and r["condition"] == cond]
        e = {}
        for side in ("aligned", "opposed", "both"):
            sides = ("aligned", "opposed") if side == "both" else (side,)
            labs = [r["final"][s] for r in sel for s in sides]
            cited = [l for l in labs if l != "OMITTED"]
            lost = [l for l in cited if l == "CITED_WITHOUT_QUALIFIER"]
            e[side] = {"cited": rate(len(cited), len(labs)), "limitation_loss_given_cited": rate(len(lost), len(cited))}
            if m == "pooled":
                items = [(r["case_id"], int(r["final"][s_] == "CITED_WITHOUT_QUALIFIER")) for r in sel for s_ in sides if r["final"][s_] != "OMITTED"]
                e[side]["limitation_loss_given_cited"] = pooled_rate(items)
        e["omitted_only_aligned"] = sum(1 for r in sel if r["final"]["aligned"] == "OMITTED" and r["final"]["opposed"] != "OMITTED")
        e["omitted_only_opposed"] = sum(1 for r in sel if r["final"]["opposed"] == "OMITTED" and r["final"]["aligned"] != "OMITTED")
        ver = [((r.get("scorer") or {}).get(sd) or {}).get("status") for r in sel for sd in ("aligned", "opposed")]
        e["scorer_verified_per_witness"] = rate(sum(1 for v in ver if v == "verified"), len(ver))
        (out["parliament"]["pooled"] if m == "pooled" else out["parliament"]["per_model"].setdefault(m, {}))[cond] = e

# ---- Process (v3 area chair)
out["process_v3_area_chair"] = {}
for m in ORDER:
    ac = [r["roles"]["area_chair"] for r in v3 if r["model"] == m]
    cov = [a["paper_coverage"] for a in ac if a.get("paper_coverage") is not None]
    outt = sorted((a.get("output_tokens") or 0) + (a.get("reasoning_tokens") or 0) for a in ac)
    out["process_v3_area_chair"][m] = {"n": len(ac), "paper_coverage_mean": round(sum(cov) / len(cov), 4), "paper_coverage_frac_ge_0.9": round(sum(c >= 0.9 for c in cov) / len(cov), 4),
                                       "files_read_median": sorted(a["files_read"] for a in ac)[len(ac) // 2], "tool_calls_median": sorted(a["tool_calls"] for a in ac)[len(ac) // 2],
                                       "output_plus_reasoning_tokens_median": outt[len(outt) // 2], "output_plus_reasoning_tokens_iqr": [outt[len(outt) // 4], outt[3 * len(outt) // 4]]}

# all-model, all-condition pooled headline rates with case-cluster intervals (as quoted in the manuscript)
out["headline"] = {
    "court_adoption_all": pooled_rate([(r["case_id"], int(r["final"] == "ADOPTED")) for r in court]),
    "parliament_limitation_loss_all": pooled_rate([(r["case_id"], int(r["final"][s_] == "CITED_WITHOUT_QUALIFIER")) for r in parl for s_ in ("aligned", "opposed") if r["final"][s_] != "OMITTED"]),
    "or_v3_adversarial_following_vanilla": pooled_rate([(r["case_id"], int(r["final"]["majority_alignment"])) for r in v3 if r["arm"] == "X" and r["condition"] == "vanilla" and r["final"].get("majority_alignment") is not None]),
    "or_v3_adversarial_following_obligation": pooled_rate([(r["case_id"], int(r["final"]["majority_alignment"])) for r in v3 if r["arm"] == "X" and r["condition"] == "obligation" and r["final"].get("majority_alignment") is not None]),
    "or_v3_following_true_accept_vanilla": pooled_rate([(r["case_id"], int(r["final"]["majority_alignment"])) for r in v3 if r["arm"] == "X" and r["condition"] == "vanilla" and r["final"].get("truth") == "accept" and r["final"].get("majority_alignment") is not None]),
    "or_v3_following_true_reject_vanilla": pooled_rate([(r["case_id"], int(r["final"]["majority_alignment"])) for r in v3 if r["arm"] == "X" and r["condition"] == "vanilla" and r["final"].get("truth") == "reject" and r["final"].get("majority_alignment") is not None]),
}
out["totals"] = {"episodes": len(court) + len(v2) + len(v3) + len(parl), "court": len(court), "open_review_v2": len(v2), "open_review_v3": len(v3), "parliament": len(parl)}
out["case_cluster"] = build_case_cluster({"court": court, "open_review_v2": v2, "open_review_v3": v3, "parliament": parl})
for condition in CONDS:
    for side in ("aligned", "opposed", "both"):
        corrected = out["case_cluster"]["parliament"]["pooled"]["rates"][condition][f"{side}/ql"]
        out["parliament"]["pooled"][condition][side]["limitation_loss_given_cited"]["ci95_case_cluster"] = corrected["ci95_case_cluster"]
out["headline"]["parliament_limitation_loss_all"]["ci95_case_cluster"] = out["case_cluster"]["parliament"]["pooled"]["rates"]["all"]["both/ql"]["ci95_case_cluster"]
with open(os.path.join(RES, "summary.json"), "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
print("totals", out["totals"])
print("court pooled adoption", out["court"]["pooled"]["vanilla"]["ADOPTED"]["rate"], out["court"]["pooled"]["obligation"]["ADOPTED"]["rate"], "adoptions", adopt_total)
print("v3 pooled following", out["open_review_v3"]["pooled"]["vanilla"]["adversarial_following"]["rate"], out["open_review_v3"]["pooled"]["obligation"]["adversarial_following"]["rate"])
print("parliament pooled QL", out["parliament"]["pooled"]["vanilla"]["both"]["limitation_loss_given_cited"]["rate"], out["parliament"]["pooled"]["obligation"]["both"]["limitation_loss_given_cited"]["rate"])
