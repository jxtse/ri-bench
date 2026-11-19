#!/usr/bin/env python3
"""Human calibration statistics for the semantic readout layer.

Reads every annotator return under data/human_annotation/returns/
(one CSV per scenario per annotator, exported from the HTML annotators),
joins them to machine_labels.json, and prints
per-annotator agreement with the semantic layer, inter-annotator
agreement, and agreement of the annotator majority with the semantic
layer. Pure stdlib.

Usage:
    python tools/human_calibration_stats.py            # text report
    python tools/human_calibration_stats.py --json     # machine-readable
    python tools/human_calibration_stats.py --tex      # appendix tables
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import itertools
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data" / "human_annotation"
KEY = QA / "machine_labels.json"
RETURNS = QA / "returns"

COURT_LABELS = ["ADOPTED", "CORRECTED", "TRUTH_STATED", "SILENT"]
PARL_LABELS = ["CITED_WITH_QUALIFIER", "CITED_WITHOUT_QUALIFIER", "OMITTED"]
COURT_SC = {"ADOPTED": r"\textsc{adopted}", "CORRECTED": r"\textsc{corrected}",
            "TRUTH_STATED": r"\textsc{truth stated}", "SILENT": r"\textsc{silent}"}
PARL_SC = {"CITED_WITH_QUALIFIER": r"\textsc{with qual.}",
           "CITED_WITHOUT_QUALIFIER": r"\textsc{without qual.}",
           "OMITTED": r"\textsc{omitted}"}


# ----------------------------------------------------------------- helpers
def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))


def cohen_kappa(a: list, b: list) -> float:
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(rows: list[list]) -> float:
    """rows: one list of rater labels per item (equal number of raters)."""
    cats = sorted({c for r in rows for c in r})
    n_r = len(rows[0])
    N = len(rows)
    P_i = []
    p_j = collections.Counter()
    for r in rows:
        c = collections.Counter(r)
        for k, v in c.items():
            p_j[k] += v
        P_i.append((sum(v * v for v in c.values()) - n_r) / (n_r * (n_r - 1)))
    P_bar = sum(P_i) / N
    p_j = {k: v / (N * n_r) for k, v in p_j.items()}
    P_e = sum(v * v for v in p_j.values())
    return 1.0 if P_e == 1 else (P_bar - P_e) / (1 - P_e)


def confusion(rows_a: list, rows_b: list, labels: list) -> dict:
    m = {x: {y: 0 for y in labels} for x in labels}
    for x, y in zip(rows_a, rows_b):
        m[x][y] += 1
    return m


def majority(labels: list, tiebreak: str | None = None):
    """Strict majority (>= 2 of 3 agreeing). Returns None when all differ."""
    c = collections.Counter(labels).most_common()
    if c[0][1] >= 2:
        return c[0][0]
    return None


def frac(a: int, b: int) -> str:
    return f"{a}/{b}" if b else "0/0"


# ------------------------------------------------------------------ loading
def order(anns):
    """Annotators A1, A2, A3 (the same three people in every scenario)."""
    fixed = [a for a in ("A1", "A2") if a in anns]
    return fixed + sorted(a for a in anns if a not in fixed)


def load_returns():
    key = json.loads(KEY.read_text())
    court_key = {r["id"]: r for r in key["court"]}
    parl_key = {r["id"]: r for r in key["parliament"]}
    rev_key = {r["id"]: r for r in key["openreview_reviews30"]}

    court, parl, rev = {}, {}, {}
    for f in sorted(RETURNS.glob("*.csv")):
        rows = read_csv(f)
        ann = {r["annotator"] for r in rows}
        assert len(ann) == 1, f"{f}: mixed annotator column {ann}"
        ann = ann.pop()
        if "court" in f.name:
            court[ann] = {r["id"]: r for r in rows}
        elif "parliament" in f.name:
            parl[ann] = {r["id"]: r for r in rows}
        elif "review" in f.name:
            rev[ann] = {r["id"]: r for r in rows}
    return court_key, parl_key, rev_key, court, parl, rev


# -------------------------------------------------------------------- court
def court_stats(key: dict, ret: dict):
    anns = order(ret)
    ids = sorted(key)
    out = {"n_items": len(ids), "annotators": anns, "missing": {}}
    # per-annotator vs semantic / regex
    per = {}
    for a in anns:
        lab = {i: ret[a][i]["label"].strip() for i in ids}
        miss = [i for i in ids if lab[i] not in COURT_LABELS]
        out["missing"][a] = miss
        keep = [i for i in ids if i not in miss]
        sem = [key[i]["gemini"] for i in keep]
        rgx = [key[i]["regex"] for i in keep]
        hum = [lab[i] for i in keep]
        adopt_sem = {i for i in keep if key[i]["gemini"] == "ADOPTED"}
        adopt_hum = {i for i in keep if lab[i] == "ADOPTED"}
        per[a] = {
            "n": len(keep),
            "agree_semantic": sum(x == y for x, y in zip(sem, hum)),
            "kappa_semantic": round(cohen_kappa(sem, hum), 3),
            "agree_regex": sum(x == y for x, y in zip(rgx, hum)),
            "kappa_regex": round(cohen_kappa(rgx, hum), 3),
            "adopted_tp": len(adopt_sem & adopt_hum),
            "adopted_sem": len(adopt_sem),
            "adopted_hum": len(adopt_hum),
            "binary_kappa_adopted": round(cohen_kappa(
                ["A" if i in adopt_sem else "-" for i in keep],
                ["A" if i in adopt_hum else "-" for i in keep]), 3),
            "label_dist": dict(collections.Counter(hum)),
            "confusion_semantic_rows_human_cols": confusion(sem, hum, COURT_LABELS),
        }
    out["per_annotator"] = per
    # inter-annotator on items labeled by all
    common = [i for i in ids if all(ret[a][i]["label"].strip() in COURT_LABELS for a in anns)]
    out["n_complete"] = len(common)
    pair = {}
    for a, b in itertools.combinations(anns, 2):
        la = [ret[a][i]["label"].strip() for i in common]
        lb = [ret[b][i]["label"].strip() for i in common]
        pair[f"{a}|{b}"] = {"agree": sum(x == y for x, y in zip(la, lb)),
                            "kappa": round(cohen_kappa(la, lb), 3)}
    out["pairwise"] = pair
    rows = [[ret[a][i]["label"].strip() for a in anns] for i in common]
    out["fleiss_kappa"] = round(fleiss_kappa(rows), 3)
    # binary adopted inter-annotator
    rows_b = [["A" if x == "ADOPTED" else "-" for x in r] for r in rows]
    out["fleiss_kappa_adopted_binary"] = round(fleiss_kappa(rows_b), 3)
    # majority vs semantic
    maj = {i: majority([ret[a][i]["label"].strip() for a in anns]) for i in common}
    dec = [i for i in common if maj[i] is not None]
    out["n_unanimous"] = sum(len(set(ret[a][i]["label"].strip() for a in anns)) == 1 for i in common)
    out["n_majority"] = len(dec)
    out["n_no_majority"] = len(common) - len(dec)
    sem = [key[i]["gemini"] for i in dec]
    rgx = [key[i]["regex"] for i in dec]
    hm = [maj[i] for i in dec]
    adopt_sem = {i for i in dec if key[i]["gemini"] == "ADOPTED"}
    adopt_maj = {i for i in dec if maj[i] == "ADOPTED"}
    out["majority"] = {
        "n": len(dec),
        "agree_semantic": sum(x == y for x, y in zip(sem, hm)),
        "kappa_semantic": round(cohen_kappa(sem, hm), 3),
        "agree_regex": sum(x == y for x, y in zip(rgx, hm)),
        "kappa_regex": round(cohen_kappa(rgx, hm), 3),
        "adopted_tp": len(adopt_sem & adopt_maj),
        "adopted_sem": len(adopt_sem),
        "adopted_maj": len(adopt_maj),
        "binary_kappa_adopted": round(cohen_kappa(
            ["A" if i in adopt_sem else "-" for i in dec],
            ["A" if i in adopt_maj else "-" for i in dec]), 3),
        "label_dist": dict(collections.Counter(hm)),
        "confusion_semantic_rows_majority_cols": confusion(sem, hm, COURT_LABELS),
        "false_adoptions_by_majority_label": dict(collections.Counter(
            maj[i] for i in adopt_sem - adopt_maj)),
        "missed_adoptions_semantic_label": dict(collections.Counter(
            key[i]["gemini"] for i in adopt_maj - adopt_sem)),
    }
    # any-annotator adopted (loose recall check)
    any_ad = {i for i in common if any(ret[a][i]["label"].strip() == "ADOPTED" for a in anns)}
    sem_ad = {i for i in common if key[i]["gemini"] == "ADOPTED"}
    out["adopted_any_annotator"] = len(any_ad)
    out["adopted_any_recovered_by_semantic"] = len(any_ad & sem_ad)
    out["semantic_adopted_confirmed_by_any_annotator"] = len(sem_ad & any_ad)
    out["semantic_adopted_complete"] = len(sem_ad)
    out["semantic_adopted_confirmed_by_k_annotators"] = dict(collections.Counter(
        sum(ret[a][i]["label"].strip() == "ADOPTED" for a in anns) for i in sem_ad))
    return out


# --------------------------------------------------------------- parliament
def parl_stats(key: dict, ret: dict):
    anns = order(ret)
    ids = sorted(key)
    slots = []  # (id, witness_index, role)
    for i in ids:
        for w, role in enumerate(key[i]["witness_order"], start=1):
            slots.append((i, w, role))
    out = {"n_reports": len(ids), "n_slots": len(slots), "annotators": anns}

    def hum(a, s):
        i, w, _ = s
        return ret[a][i][f"witness{w}_label"].strip()

    def sem(s):
        i, _, role = s
        return key[i]["gemini"][role]

    per = {}
    for a in anns:
        miss = [s for s in slots if hum(a, s) not in PARL_LABELS]
        keep = [s for s in slots if s not in miss]
        S = [sem(s) for s in keep]
        H = [hum(a, s) for s in keep]
        cited_s = ["C" if x != "OMITTED" else "O" for x in S]
        cited_h = ["C" if x != "OMITTED" else "O" for x in H]
        both = [(x, y) for x, y in zip(S, H) if x != "OMITTED" and y != "OMITTED"]
        per[a] = {
            "n": len(keep), "missing": len(miss),
            "agree_semantic": sum(x == y for x, y in zip(S, H)),
            "kappa_semantic": round(cohen_kappa(S, H), 3),
            "cited_binary_kappa": round(cohen_kappa(cited_s, cited_h), 3),
            "both_cited": len(both),
            "both_cited_qualifier_agree": sum(x == y for x, y in both),
            "loss_among_cited_semantic": frac(sum(x == "CITED_WITHOUT_QUALIFIER" for x in S),
                                              sum(x != "OMITTED" for x in S)),
            "loss_among_cited_human": frac(sum(x == "CITED_WITHOUT_QUALIFIER" for x in H),
                                           sum(x != "OMITTED" for x in H)),
            "label_dist": dict(collections.Counter(H)),
            "confusion_semantic_rows_human_cols": confusion(S, H, PARL_LABELS),
        }
    out["per_annotator"] = per
    common = [s for s in slots if all(hum(a, s) in PARL_LABELS for a in anns)]
    out["n_complete"] = len(common)
    pair = {}
    for a, b in itertools.combinations(anns, 2):
        la = [hum(a, s) for s in common]
        lb = [hum(b, s) for s in common]
        pair[f"{a}|{b}"] = {
            "agree": sum(x == y for x, y in zip(la, lb)),
            "kappa": round(cohen_kappa(la, lb), 3),
            "cited_binary_kappa": round(cohen_kappa(
                ["C" if x != "OMITTED" else "O" for x in la],
                ["C" if x != "OMITTED" else "O" for x in lb]), 3)}
    out["pairwise"] = pair
    rows = [[hum(a, s) for a in anns] for s in common]
    out["fleiss_kappa"] = round(fleiss_kappa(rows), 3)
    out["fleiss_kappa_cited_binary"] = round(fleiss_kappa(
        [["C" if x != "OMITTED" else "O" for x in r] for r in rows]), 3)
    maj = {s: majority([hum(a, s) for a in anns]) for s in common}
    dec = [s for s in common if maj[s] is not None]
    out["n_unanimous"] = sum(len(set(r)) == 1 for r in rows)
    out["n_majority"] = len(dec)
    out["n_no_majority"] = len(common) - len(dec)
    S = [sem(s) for s in dec]
    M = [maj[s] for s in dec]
    both = [(x, y) for x, y in zip(S, M) if x != "OMITTED" and y != "OMITTED"]
    out["majority"] = {
        "n": len(dec),
        "agree_semantic": sum(x == y for x, y in zip(S, M)),
        "kappa_semantic": round(cohen_kappa(S, M), 3),
        "cited_binary_kappa": round(cohen_kappa(
            ["C" if x != "OMITTED" else "O" for x in S],
            ["C" if x != "OMITTED" else "O" for x in M]), 3),
        "both_cited": len(both),
        "both_cited_qualifier_agree": sum(x == y for x, y in both),
        "both_cited_semantic_credits_qualifier_human_not": sum(
            x == "CITED_WITH_QUALIFIER" and y == "CITED_WITHOUT_QUALIFIER" for x, y in both),
        "both_cited_human_credits_qualifier_semantic_not": sum(
            x == "CITED_WITHOUT_QUALIFIER" and y == "CITED_WITH_QUALIFIER" for x, y in both),
        "loss_among_cited_semantic": frac(sum(x == "CITED_WITHOUT_QUALIFIER" for x in S),
                                          sum(x != "OMITTED" for x in S)),
        "loss_among_cited_majority": frac(sum(x == "CITED_WITHOUT_QUALIFIER" for x in M),
                                          sum(x != "OMITTED" for x in M)),
        "label_dist": dict(collections.Counter(M)),
        "confusion_semantic_rows_majority_cols": confusion(S, M, PARL_LABELS),
    }
    return out


# ------------------------------------------------------------------ reviews
def review_stats(key: dict, ret: dict):
    anns = order(ret)
    ids = sorted(key)
    out = {"n": len(ids), "annotators": anns, "per_annotator": {}, "by_quality": {}}
    qual = {i: key[i]["quality"] for i in ids}
    groups = collections.defaultdict(list)
    for i in ids:
        groups[qual[i]].append(i)
    for a in anns:
        spec = {i: int(ret[a][i]["specificity"]) for i in ids}
        d = {}
        for q, members in sorted(groups.items()):
            vals = sorted(spec[i] for i in members)
            d[q] = {"n": len(members), "specificity": vals,
                    "mean": round(statistics.mean(vals), 2),
                    "ge4": sum(v >= 4 for v in vals),
                    "direction": dict(collections.Counter(ret[a][i]["direction"] for i in members)),
                    "read": dict(collections.Counter(ret[a][i]["read"] for i in members))}
        out["per_annotator"][a] = {
            "by_quality": d,
            "direction": dict(collections.Counter(ret[a][i]["direction"] for i in ids)),
        }
    # mean specificity across annotators per review, then threshold separation
    mean_spec = {i: statistics.mean(int(ret[a][i]["specificity"]) for a in anns) for i in ids}
    for q, members in sorted(groups.items()):
        vals = sorted(round(mean_spec[i], 2) for i in members)
        out["by_quality"][q] = {"n": len(members), "mean_specificity_per_review": vals,
                                "mean": round(statistics.mean(vals), 2),
                                "ge3_5": sum(v >= 3.5 for v in vals),
                                "ge4": sum(v >= 4 for v in vals)}
    # inter-annotator: Spearman-free simple stats: pairwise exact agreement and |diff|<=1
    pair = {}
    for a, b in itertools.combinations(anns, 2):
        da = [int(ret[a][i]["specificity"]) for i in ids]
        db = [int(ret[b][i]["specificity"]) for i in ids]
        pair[f"{a}|{b}"] = {
            "specificity_exact": sum(x == y for x, y in zip(da, db)),
            "specificity_within1": sum(abs(x - y) <= 1 for x, y in zip(da, db)),
            "specificity_kendall_tau_b": round(kendall_tau_b(da, db), 3),
            "direction_agree": sum(ret[a][i]["direction"] == ret[b][i]["direction"] for i in ids),
        }
    out["pairwise"] = pair
    # separation: constructed-vote styles by every annotator
    sep = {}
    for a in anns:
        s = {i: int(ret[a][i]["specificity"]) for i in ids}
        sp = [s[i] for i in groups["specific"]]
        ge = [s[i] for i in groups["generic"]]
        sep[a] = {"specific_min": min(sp), "generic_max": max(ge),
                  "perfect_threshold_exists": min(sp) > max(ge),
                  "specific_ge4": sum(v >= 4 for v in sp), "generic_ge4": sum(v >= 4 for v in ge)}
    out["separation_specific_vs_generic"] = sep
    return out


def kendall_tau_b(x, y):
    n = len(x)
    c = d = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                c += 1
            else:
                d += 1
    den = ((c + d + tx) * (c + d + ty)) ** 0.5
    return (c - d) / den if den else 0.0


# ---------------------------------------------------------------------- tex
def tex_tables(court, parl):
    L = []
    # court: semantic vs majority confusion
    m = court["majority"]["confusion_semantic_rows_majority_cols"]
    L.append(r"\begin{table}[h]")
    L.append(r"\centering\small")
    L.append(r"\caption{Court: semantic-layer label (rows) against the majority of three "
             rf"annotators (columns), ${court['majority']['n']}$ of $100$ stratified rulings "
             r"with a majority label.}")
    L.append(r"\label{tab:human-court}")
    L.append(r"\begin{tabular}{l rrrr r}")
    L.append(r"\toprule")
    L.append(r"Semantic $\backslash$ Human & " + " & ".join(COURT_SC[c] for c in COURT_LABELS) + r" & Total \\")
    L.append(r"\midrule")
    tot = collections.Counter()
    for r in COURT_LABELS:
        row = [m[r][c] for c in COURT_LABELS]
        for c, v in zip(COURT_LABELS, row):
            tot[c] += v
        L.append(f"{COURT_SC[r]:22s} & " + " & ".join(f"{v:2d}" for v in row) + f" & {sum(row)} \\\\")
    L.append(r"\midrule")
    L.append("Total & " + " & ".join(str(tot[c]) for c in COURT_LABELS) + f" & {sum(tot.values())} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L.append("")
    # parliament
    m = parl["majority"]["confusion_semantic_rows_majority_cols"]
    L.append(r"\begin{table}[h]")
    L.append(r"\centering\small")
    L.append(r"\caption{Parliament: semantic-layer witness label (rows) against the majority of three "
             rf"annotators (columns), ${parl['majority']['n']}$ of $200$ witness slots in $100$ "
             r"stratified reports with a majority label.}")
    L.append(r"\label{tab:human-parliament}")
    L.append(r"\begin{tabular}{l rrr r}")
    L.append(r"\toprule")
    L.append(r"Semantic $\backslash$ Human & " + " & ".join(PARL_SC[c] for c in PARL_LABELS) + r" & Total \\")
    L.append(r"\midrule")
    tot = collections.Counter()
    for r in PARL_LABELS:
        row = [m[r][c] for c in PARL_LABELS]
        for c, v in zip(PARL_LABELS, row):
            tot[c] += v
        name = {"CITED_WITH_QUALIFIER": r"\textsc{cited with qualifier}",
                "CITED_WITHOUT_QUALIFIER": r"\textsc{cited without qualifier}",
                "OMITTED": r"\textsc{omitted}"}[r]
        L.append(f"{name:34s} & " + " & ".join(f"{v:3d}" for v in row) + f" & {sum(row)} \\\\")
    L.append(r"\midrule")
    L.append("Total & " + " & ".join(str(tot[c]) for c in PARL_LABELS) + f" & {sum(tot.values())} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L.append("")
    # per-annotator summary table
    L.append(r"\begin{table}[h]")
    L.append(r"\centering\small")
    L.append(r"\caption{Per-annotator agreement with the semantic layer. Court: $\kappa$ over four labels, "
             r"recall and precision of \textsc{adopted}. Parliament: $\kappa$ over three labels, "
             r"$\kappa$ on whether a claim was cited, and loss among cited claims by the annotator "
             r"(semantic layer on the same slots in the last row).}")
    L.append(r"\label{tab:human-annotators}")
    L.append(r"\begin{tabular}{l rrr rrr}")
    L.append(r"\toprule")
    L.append(r" & \multicolumn{3}{c}{Court ($n=100$)} & \multicolumn{3}{c}{Parliament ($200$ slots)} \\")
    L.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    L.append(r"Annotator & $\kappa$ & Recall & Precision & $\kappa$ & $\kappa_{\text{cited}}$ & Loss \\")
    L.append(r"\midrule")
    for k, (a, b) in enumerate(zip(court["annotators"], parl["annotators"]), start=1):
        c = court["per_annotator"][a]
        p = parl["per_annotator"][b]
        n_note = "" if c["n"] == 100 else "\textsuperscript{*}"
        L.append(f"A{k}{n_note} & {c['kappa_semantic']:.2f} & {c['adopted_tp']}/{c['adopted_hum']} & "
                 f"{c['adopted_tp']}/{c['adopted_sem']} & {p['kappa_semantic']:.2f} & "
                 f"{p['cited_binary_kappa']:.2f} & {p['loss_among_cited_human']} \\\\")
    cm = court["majority"]
    pm = parl["majority"]
    L.append(r"\midrule")
    L.append(f"Majority & {cm['kappa_semantic']:.2f} & {cm['adopted_tp']}/{cm['adopted_maj']} & "
             f"{cm['adopted_tp']}/{cm['adopted_sem']} & {pm['kappa_semantic']:.2f} & "
             f"{pm['cited_binary_kappa']:.2f} & {pm['loss_among_cited_majority']} \\\\")
    L.append(f"Semantic layer & -- & -- & -- & -- & -- & {pm['loss_among_cited_semantic']} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--tex", action="store_true")
    args = ap.parse_args()
    ck, pk, rk, cr, pr, rr = load_returns()
    court = court_stats(ck, cr)
    parl = parl_stats(pk, pr)
    rev = review_stats(rk, rr)
    if args.tex:
        print(tex_tables(court, parl))
        return
    res = {"court": court, "parliament": parl, "reviews30": rev}
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
