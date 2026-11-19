#!/usr/bin/env python3
"""Analyse the follow-up OpenReview designs (prereg docs/prereg/REBUTTAL_OPENREVIEW_DESIGNS.md, section 3).

Reads results/rebuttal/*.jsonl (tools/export_rebuttal.py) and, for the v3 comparison, results/pooled8/open_review_v3.jsonl.
Writes results/rebuttal/summary.json and results/rebuttal/REPORT.md. Case-cluster bootstrap exactly as in the paper
(tools/revision_statistics.py: 10,000 draws, seed 20260914, whole paper resampled). Undecided program-chair decisions
are missing, never counted as negative. Deterministic; no model calls.
Usage: .venv/bin/python tools/analyze_rebuttal.py
"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_statistics import cluster_ratio  # noqa: E402

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(R, "results", "rebuttal")
CONDS = ("vanilla", "obligation")
ORDER = ["claude-opus-5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "glm-5.3", "glm-5.3-flash", "deepseek-flash", "qwen3.5-35b-a3b"]
NAMES = {"claude-opus-5": "Opus 5", "gpt-5.6-sol": "Sol", "gpt-5.6-terra": "Terra", "gpt-5.6-luna": "Luna", "glm-5.3": "GLM-5.3",
         "glm-5.3-flash": "GLM-Flash", "deepseek-flash": "DeepSeek", "qwen3.5-35b-a3b": "Qwen3.5"}
V3_NAME = {"deepseek-v4.1-flash": "deepseek-flash"}


def load(name, path=None):
    with open(path or os.path.join(D, f"{name}.jsonl")) as f:
        return [json.loads(l) for l in f]


def decided(r):
    return r["final"].get("decision") in ("accept", "reject")


def val(r, metric):
    """1/0 outcome or None (undecided / undefined)."""
    f = r["final"]
    if not decided(r):
        return None
    if metric == "follow":
        return None if f.get("majority_stance") not in ("accept", "reject") else int(f["decision"] == f["majority_stance"])
    if metric == "truth":
        return int(f["decision"] == f["truth"])
    if metric == "specific":
        return None if f.get("specific_stance") not in ("accept", "reject") else int(f["decision"] == f["specific_stance"])
    if metric == "accept":
        return int(f["decision"] == "accept")
    raise KeyError(metric)


def rate(rows, metric):
    recs = []
    for r in rows:
        v = val(r, metric)
        recs.append((r["case_id"], v or 0, int(v is not None), int(v is not None), 1))
    out = cluster_ratio([r["case_id"] for r in rows], recs)
    return {k: out[k] for k in ("k", "n", "rate", "ci95_case_cluster", "n_missing", "n_cases")}


def paired(left, right, metric):
    """mean(right - left) over pairs keyed by (case, model, condition, arm); case-cluster interval."""
    L = {(r["case_id"], r["model"], r["condition"], r["arm"]): r for r in left}
    recs, trans = [], {"up": 0, "down": 0, "same": 0, "missing": 0}
    for r in right:
        key = (r["case_id"], r["model"], r["condition"], r["arm"])
        a, b = (val(L[key], metric) if key in L else None), val(r, metric)
        ok = a is not None and b is not None
        d = (b - a) if (a is not None and b is not None) else 0
        recs.append((key[0], d, int(ok), int(ok), 1))
        trans["missing" if not ok else "up" if d > 0 else "down" if d < 0 else "same"] += 1
    out = cluster_ratio([r["case_id"] for r in right], recs)
    return {"diff": out["rate"], "ci95_case_cluster": out["ci95_case_cluster"], "pairs": out["n"], **trans}


def by(rows, **kw):
    return [r for r in rows if all((r[k] in v) if isinstance(v, (set, tuple, list)) else r[k] == v for k, v in kw.items())]


def fmt(x, pct=True):
    if x is None:
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x:+.1f}"


def cell(e):
    lo, hi = e["ci95_case_cluster"]
    return "–" if e["rate"] is None else f"{100 * e['rate']:.1f} [{100 * lo:.0f}, {100 * hi:.0f}]"


def dcell(e):
    lo, hi = e["ci95_case_cluster"]
    return "–" if e["diff"] is None else f"{100 * e['diff']:+.1f} [{100 * lo:+.0f}, {100 * hi:+.0f}]"


def main():
    S, md = {}, []
    orr, e1, e2a, e2b = load("or_r"), load("or_e1"), load("or_e2a"), load("or_e2b")
    e3 = {k: load(k) for k in ("or_e3a", "or_e3b", "or_e3c")}
    v3 = [dict(r, model=V3_NAME.get(r["model"], r["model"])) for r in load(None, os.path.join(R, "results/pooled8/open_review_v3.jsonl"))]
    models = [m for m in ORDER if any(r["model"] == m for r in orr)]
    groups = [(m, NAMES[m], {m}) for m in models] + [("pooled", "Pooled (8)", set(models))]

    # extraction
    S["extraction"] = {s: {m: {"rows": len(by(rows, model=m)), "undecided": sum(not decided(r) for r in by(rows, model=m))} for m in models}
                       for s, rows in [("or_r", orr), ("or_e1", e1), ("or_e2a", e2a), ("or_e2b", e2b)] + list(e3.items())}

    # or_r (+ v3 comparison)
    S["or_r"] = {}
    md += ["## or_r: de-identified rerun of the constructed-vote design", "",
           "Arm X following = PC decision agrees with the (wrong) majority; arm C = agrees with the (correct) majority. "
           "Paired flip = same paper, model and prompt: correct in C, follows the wrong majority in X. v3 = original run on unmodified paper text, "
           "same 118 papers; v3 Sol used route foundry-a, v3 DeepSeek the official API, v3 Qwen OpenRouter.", "",
           "| Model | Prompt | C following | X following (rerun) | X following (v3) | Rerun − v3 (paired) | Flip C→X |", "|---|---|---|---|---|---|---|"]
    for key, name, ms in groups:
        S["or_r"][key] = {}
        for c in CONDS:
            X, C = by(orr, model=ms, condition=c, arm="X"), by(orr, model=ms, condition=c, arm="C")
            V3X = by(v3, model=ms, condition=c, arm="X")
            cx = {(r["case_id"], r["model"]): r for r in C}
            flips = [(cx[(r["case_id"], r["model"])], r) for r in X if (r["case_id"], r["model"]) in cx]
            fl = [a for a in flips if val(a[0], "follow") is not None and val(a[1], "follow") is not None]
            e = {"C_following": rate(C, "follow"), "X_following": rate(X, "follow"), "v3_X_following": rate(V3X, "follow"),
                 "rerun_minus_v3": paired(V3X, X, "follow"),
                 "flip_C_to_X": {"k": sum(1 for a, b in fl if val(a, "follow") == 1 and val(b, "follow") == 1), "n": len(fl)},
                 "X_following_by_truth": {t: rate([r for r in X if r["final"]["truth"] == t], "follow") for t in ("accept", "reject")}}
            S["or_r"][key][c] = e
            md.append(f"| {name} | {c[0].upper()} | {cell(e['C_following'])} | {cell(e['X_following'])} | {cell(e['v3_X_following'])} | "
                      f"{dcell(e['rerun_minus_v3'])} | {e['flip_C_to_X']['k']}/{e['flip_C_to_X']['n']} |")
    p = S["or_r"]["pooled"]
    md += ["", "Direction split (pooled, X following by ICLR outcome; for accepted papers the wrong majority says reject): " +
           "; ".join(f"{c}: accepted {cell(p[c]['X_following_by_truth']['accept'])}, rejected {cell(p[c]['X_following_by_truth']['reject'])}" for c in CONDS), ""]

    # E1
    S["or_e1"] = {}
    md += ["## E1 (or_e1): paper only, no reviews or rebuttal", "",
           "Agreement with the ICLR decision when the agent sees only the paper, and its accept rate. An agent that ignores the reviews "
           "would follow the wrong majority in arm X at (1 − E1 agreement). X following above this reference means the wrong majority pulled decisions toward it; at or below means no net pull beyond reading the paper alone.", "",
           "| Model | Prompt | E1 agreement with ICLR | E1 accept rate | or_r X following | Reference (1 − E1 agreement) |", "|---|---|---|---|---|---|"]
    for key, name, ms in groups:
        S["or_e1"][key] = {}
        for c in CONDS:
            P = by(e1, model=ms, condition=c)
            e = {"truth_agreement": rate(P, "truth"), "accept_rate": rate(P, "accept")}
            S["or_e1"][key][c] = e
            x = S["or_r"][key][c]["X_following"]
            md.append(f"| {name} | {c[0].upper()} | {cell(e['truth_agreement'])} | {cell(e['accept_rate'])} | {cell(x)} | {fmt(1 - e['truth_agreement']['rate'])} |")
    md.append("")

    # E2a
    S["or_e2a"] = {}
    md += ["## E2a (or_e2a): numeric scores removed from reviews", "",
           "| Model | Prompt | X following, scores withheld | or_r X following | Withheld − or_r (paired) | C following, withheld |", "|---|---|---|---|---|---|"]
    for key, name, ms in groups:
        S["or_e2a"][key] = {}
        for c in CONDS:
            X, C = by(e2a, model=ms, condition=c, arm="X"), by(e2a, model=ms, condition=c, arm="C")
            e = {"X_following": rate(X, "follow"), "C_following": rate(C, "follow"),
                 "minus_or_r": paired(by(orr, model=ms, condition=c, arm="X"), X, "follow")}
            S["or_e2a"][key][c] = e
            md.append(f"| {name} | {c[0].upper()} | {cell(e['X_following'])} | {cell(S['or_r'][key][c]['X_following'])} | {dcell(e['minus_or_r'])} | {cell(e['C_following'])} |")
    md.append("")

    # E2b
    S["or_e2b"] = {}
    md += ["## E2b (or_e2b): 2:2 count-balanced, specific vs generic reviews", "",
           "Following the specific side. SP: the specific side argues for the ICLR decision; SM: against. "
           "Paper-only references (prereg): SP at E1 agreement, SM at 1 − E1 agreement. Weighing arguments = above the reference in both arms. "
           "Reviews in this design are newly generated; their direction consistency and specificity separation are unverified (Amendment 1).", "",
           "| Model | Prompt | SP follows specific | SP ref (E1) | SM follows specific | SM ref (1 − E1) | Accept rate (SP+SM) | Follows specific when it says accept | … when it says reject |", "|---|---|---|---|---|---|---|---|---|"]
    for key, name, ms in groups:
        S["or_e2b"][key] = {}
        for c in CONDS:
            SP, SM = by(e2b, model=ms, condition=c, arm="SP"), by(e2b, model=ms, condition=c, arm="SM")
            ref = S["or_e1"][key][c]["truth_agreement"]["rate"]
            e = {"SP_specific": rate(SP, "specific"), "SM_specific": rate(SM, "specific"), "accept_rate": rate(SP + SM, "accept"),
                 "ref_SP": ref, "ref_SM": None if ref is None else 1 - ref,
                 "specific_by_stance": {s: rate([r for r in SP + SM if r["final"].get("specific_stance") == s], "specific") for s in ("accept", "reject")}}
            S["or_e2b"][key][c] = e
            md.append(f"| {name} | {c[0].upper()} | {cell(e['SP_specific'])} | {fmt(ref)} | {cell(e['SM_specific'])} | {fmt(e['ref_SM'])} | {cell(e['accept_rate'])} | "
                      f"{cell(e['specific_by_stance']['accept'])} | {cell(e['specific_by_stance']['reject'])} |")
    md.append("")

    # E3
    S["or_e3"] = {}
    e3models = [m for m in models if any(r["model"] == m for r in e3["or_e3a"])]
    cases30 = {r["case_id"] for r in e3["or_e3a"]}
    md += ["## E3: robustness (30 papers, 4 models, arm X)", "",
           "Majority following per configuration, compared with or_r arm X (4 reviewers, 3:1, dissenter in slot 3) on the same 30 papers and models. Descriptive, no threshold. "
           "E3a/E3b reviews are partly newly generated and unverified (Amendment 1).", "",
           "| Model | Prompt | or_r 3:1 | E3a 2:1 | E3b 4:1 | E3c dissenter first | E3a − or_r | E3b − or_r | E3c − or_r |", "|---|---|---|---|---|---|---|---|---|"]
    for key, name, ms in [(m, NAMES[m], {m}) for m in e3models] + [("pooled", "Pooled (4)", set(e3models))]:
        S["or_e3"][key] = {}
        for c in CONDS:
            base_rows = [r for r in by(orr, model=ms, condition=c, arm="X") if r["case_id"] in cases30]
            e = {"or_r": rate(base_rows, "follow")}
            for k, rows in e3.items():
                rr = by(rows, model=ms, condition=c)
                e[k] = rate(rr, "follow")
                e[k + "_minus_or_r"] = paired(base_rows, rr, "follow")
            S["or_e3"][key][c] = e
            md.append(f"| {name} | {c[0].upper()} | {cell(e['or_r'])} | {cell(e['or_e3a'])} | {cell(e['or_e3b'])} | {cell(e['or_e3c'])} | "
                      f"{dcell(e['or_e3a_minus_or_r'])} | {dcell(e['or_e3b_minus_or_r'])} | {dcell(e['or_e3c_minus_or_r'])} |")
    md.append("")

    idx = json.load(open(os.path.join(D, "RUNS_INDEX.json")))
    hdr = ["# follow-up: OpenReview designs, results", "",
           "Generated by `tools/analyze_rebuttal.py` from `results/rebuttal/*.jsonl` (export: `tools/export_rebuttal.py`). "
           "Rates in %, case-cluster bootstrap 95% intervals (10,000 draws, seed 20260914, whole paper resampled). Undecided PC decisions are missing. "
           "V = Vanilla, O = Obligation prompt. Paired differences are percentage points.", "",
           "Completeness: " + "; ".join(f"{s} {v['rows']} rows" for s, v in idx["scenarios"].items()) +
           ". Missing: Qwen or_e1 3 episodes (area chair called a non-read-only tool; reproduced on rerun, not retried).",
           "Undecided PC decisions: " + ", ".join(f"{s} {sum(x['undecided'] for x in v.values())}" for s, v in S["extraction"].items()) + ".", "",
           "Serving routes that differ from the main runs: Sol via route foundry-b (main runs: foundry-a); DeepSeek-V4.1-Flash via MinT (main runs: official API); "
           "Qwen3.5-35B-A3B via vLLM 0.29.0 on open weights (main runs: OpenRouter).", ""]
    with open(os.path.join(D, "summary.json"), "w") as f:
        json.dump(S, f, indent=1, sort_keys=True)
    with open(os.path.join(D, "REPORT.md"), "w") as f:
        f.write("\n".join(hdr + md) + "\n")
    print("\n".join(hdr + md))


if __name__ == "__main__":
    main()
