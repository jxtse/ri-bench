#!/usr/bin/env python3
"""Revision tables for the OpenReview review-package controls (paper revision).

Reads results/revision8/open_review_v3.jsonl (constructed-vote design, eight models) and
results/rebuttal/{or_e1,or_e2a,or_e2b,or_e3a,or_e3b,or_e3c}.jsonl, and writes
manuscript/sections/table_or_controls.tex (pooled, main text) and
manuscript/sections/table_or_controls_models.tex (per model, appendix).
Intervals: whole-paper case-cluster bootstrap, 10,000 draws, seed 20260914 (tools/revision_statistics.py).
Deterministic; no model calls. Usage: .venv/bin/python tools/make_rebuttal_tables.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from revision_statistics import cluster_difference, cluster_ratio  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "results" / "revision8" / "open_review_v3.jsonl"
REB = ROOT / "results" / "rebuttal"
OUT = ROOT / "manuscript" / "sections"
MODELS = [("claude-opus-5", "Claude Opus 5"), ("gpt-5.6-sol", "GPT-5.6 Sol"), ("gpt-5.6-terra", "GPT-5.6 Terra"),
          ("gpt-5.6-luna", "GPT-5.6 Luna"), ("glm-5.3", "GLM-5.3"), ("glm-5.3-flash", "GLM-5.3-Flash"),
          ("qwen3.5-35b-a3b", "Qwen3.5-35B-A3B"), ("deepseek-v4.1-flash", "DeepSeek-V4.1-Flash")]
E3_MODELS = {"claude-opus-5", "gpt-5.6-sol", "glm-5.3", "qwen3.5-35b-a3b"}
CONDS = ("vanilla", "obligation")


def load(path):
    rows = [json.loads(line) for line in open(path)]
    for r in rows:
        if r["model"] == "deepseek-flash":
            r["model"] = "deepseek-v4.1-flash"
    return rows


def decision(r):
    d = (r.get("final") or {}).get("decision")
    return d if d in ("accept", "reject") else None


def away(r):
    """1 if the program chair's decision differs from the historical ICLR decision."""
    return int(decision(r) != r["final"]["truth"])


def keyed(rows, arm=None):
    return {(r["case_id"], r["model"], r["condition"]): r for r in rows
            if (arm is None or r["arm"] == arm) and decision(r)}


main = load(MAIN)
X, C = keyed(main, "X"), keyed(main, "C")
E1 = keyed(load(REB / "or_e1.jsonl"))
E2A = keyed(load(REB / "or_e2a.jsonl"), "X")
e2b = load(REB / "or_e2b.jsonl")
SP, SM = keyed(e2b, "SP"), keyed(e2b, "SM")
E3 = {s: keyed(load(REB / f"or_{s}.jsonl"), "X") for s in ("e3a", "e3b", "e3c")}


def follows_specific(r):
    """E2b: 1 if the decision goes with the side argued by the two specific reviews."""
    return int(decision(r) == r["final"]["specific_stance"])


def rate(table, fn, models, cond):
    recs = [(k[0], fn(r), 1, 1, 1) for k, r in table.items() if k[1] in models and k[2] == cond]
    return cluster_ratio([k[0] for k in table], recs)


def paired(left, right, lf, rf, models, cond):
    keys = [k for k in left if k in right and k[1] in models and k[2] == cond]
    frame = sorted({k[0] for k in left} | {k[0] for k in right})
    out = cluster_difference(frame, [(k[0], lf(left[k]), 1, 1, 1) for k in keys],
                             [(k[0], rf(right[k]), 1, 1, 1) for k in keys])
    out["pairs"] = len(keys)
    return out


def ref_e1(models, cond, arm):
    """Paper-only reference for E2b: SP follows specific at E1 agreement, SM at 1 - E1 agreement."""
    disagreement = float(rate(E1, away, models, cond)["rate"])
    return 1 - disagreement if arm == "SP" else disagreement


def f2(x):
    return f"{x:.2f}".replace("-0.00", "0.00").replace("-", "$-$")


def ci(d, key="rate"):
    lo, hi = d["ci95_case_cluster"]
    return f"{f2(d[key])} [{f2(lo)}, {f2(hi)}]"


def sci(d):
    lo, hi = d["ci95_case_cluster"]
    s = lambda v: ("$+$" if round(v, 2) > 0 else "") + f2(v)  # noqa: E731
    return f"{s(d['estimate'])} [{s(lo)}, {s(hi)}]"


def sgn(d):
    """signed point estimate without interval (main-text Table 2)"""
    v = d["estimate"]
    return ("$+$" if round(v, 2) > 0 else "") + f2(v)


ALL = {m for m, _ in MODELS}
cells = {}
for cond in CONDS:
    c = cells[cond] = {}
    c["adv"] = rate(X, away, ALL, cond)
    c["e1"] = rate(E1, away, ALL, cond)
    c["excess"] = paired(X, E1, away, away, ALL, cond)
    c["e2a"] = paired(E2A, X, away, away, ALL, cond)
    c["sp"] = rate(SP, follows_specific, ALL, cond)
    c["sm"] = rate(SM, follows_specific, ALL, cond)
    c["sp_ref"], c["sm_ref"] = ref_e1(ALL, cond, "SP"), ref_e1(ALL, cond, "SM")
    c["e3_base"] = rate({k: v for k, v in X.items() if k[0] in {kk[0] for kk in E3["e3a"]}}, away, E3_MODELS, cond)
    for s in ("e3a", "e3b", "e3c"):
        c[s] = rate(E3[s], away, E3_MODELS, cond)

V, O = cells["vanilla"], cells["obligation"]
t = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}",
     r"\caption{\rev{Controls for the review package (OpenReview, constructed votes; 118 papers and eight models unless noted). "
     r"Rates of program-chair decisions. Paper only: the agent sees the paper but no reviews or "
     r"rebuttal; an agent that ignores the reviews would follow the adversarial majority at the rate of its paper-only disagreement "
     r"with the historical decision (reference). Excess: same paper, model and prompt, adversarial following minus that reference. "
     r"No scores: every numeric score removed from reviews and responses, change against the original adversarial arm. "
     r"2:2: two specific reviews against two generic reviews on the other side, with the specific side for (SP) or against (SM) "
     r"the historical decision; parentheses give the paper-only reference. Ratio and order: 30 papers, four models.}}",
     r"\label{tab:or-controls}",
     r"\begin{tabular}{l l cc}\toprule",
     r"Design & Quantity & Vanilla & Obligation \\ \midrule",
     rf"Adversarial 3:1 & follows majority & {f2(V['adv']['rate'])} & {f2(O['adv']['rate'])} \\",
     rf"Paper only & reference & {f2(V['e1']['rate'])} & {f2(O['e1']['rate'])} \\",
     rf" & excess over reference & {sgn(V['excess'])} & {sgn(O['excess'])} \\",
     rf"No scores & change in following & {sgn(V['e2a'])} & {sgn(O['e2a'])} \\",
     rf"2:2, SP & follows specific side & {f2(V['sp']['rate'])} ({f2(V['sp_ref'])}) & {f2(O['sp']['rate'])} ({f2(O['sp_ref'])}) \\",
     rf"2:2, SM & follows specific side & {f2(V['sm']['rate'])} ({f2(V['sm_ref'])}) & {f2(O['sm']['rate'])} ({f2(O['sm_ref'])}) \\ \midrule",
     rf"Ratio 2:1 / 3:1 / 4:1 & follows majority & {f2(V['e3a']['rate'])} / {f2(V['e3_base']['rate'])} / {f2(V['e3b']['rate'])} & "
     rf"{f2(O['e3a']['rate'])} / {f2(O['e3_base']['rate'])} / {f2(O['e3b']['rate'])} \\",
     rf"3:1, dissenter first & follows majority & {f2(V['e3c']['rate'])} & {f2(O['e3c']['rate'])} \\",
     r"\bottomrule\end{tabular}\end{table}"]
(OUT / "table_or_controls.tex").write_text("\n".join(t) + "\n")

rows = []
for mid, name in MODELS:
    vals = []
    for cond in CONDS:
        ex = paired(X, E1, away, away, {mid}, cond)
        sp, sm = rate(SP, follows_specific, {mid}, cond), rate(SM, follows_specific, {mid}, cond)
        vals += [ci(rate(X, away, {mid}, cond)), f2(rate(E1, away, {mid}, cond)["rate"]), sci(ex),
                 f"{f2(sp['rate'])} / {f2(sm['rate'])}"]
    rows.append(name + " & " + " & ".join(vals) + r" \\")
m = [r"\begin{table}[h]\centering\small\setlength{\tabcolsep}{3pt}",
     r"\caption{\rev{Review-package controls per model (constructed votes, 118 papers). Adv: adversarial-arm majority following. "
     r"Ref: paper-only reference (disagreement with the historical decision when no reviews are shown). Excess: paired difference with "
     r"case-cluster 95\% interval. SP / SM: following of the specific side in the 2:2 design when it argues for / against the "
     r"historical decision.}}",
     r"\label{tab:or-controls-models}",
     r"\resizebox{\linewidth}{!}{\begin{tabular}{l cccc cccc}\toprule",
     r" & \multicolumn{4}{c}{Vanilla} & \multicolumn{4}{c}{Obligation} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
     r"Model & Adv & Ref & Excess & SP / SM & Adv & Ref & Excess & SP / SM \\ \midrule"] + rows + \
    [r"\bottomrule\end{tabular}}\end{table}"]
(OUT / "table_or_controls_models.tex").write_text("\n".join(m) + "\n")

summary = {cond: {k: (v if isinstance(v, float) else {kk: vv for kk, vv in v.items() if kk in ("rate", "estimate", "k", "n", "pairs", "ci95_case_cluster")})
                  for k, v in cells[cond].items()} for cond in CONDS}
(ROOT / "results" / "revision8" / "or_controls.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
