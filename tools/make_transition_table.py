"""Paired Vanilla -> Obligation transition tables for Court rulings and
Parliament witness slots, written as one LaTeX table for the appendix.

Pairs are formed within (case, model, arm) for Court and within
(inquiry, model, witness) for Parliament, from results/*.jsonl.
Usage: python tools/make_transition_table.py > build/tables/table_transitions.tex
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(name):
    return [json.loads(l) for l in open(RES / name)]


def court():
    rows = load("court.jsonl")
    key = lambda r: (r["case_id"], r["model"], r["arm"])
    v, o = {}, {}
    for r in rows:
        (v if r["condition"] == "vanilla" else o)[key(r)] = r["final"]
    cats = ["TRUTH_STATED", "CORRECTED", "ADOPTED", "SILENT"]
    t = collections.Counter((v[k], o[k]) for k in v if k in o)
    return cats, t, sum(t.values())


def parliament():
    rows = load("parliament.jsonl")
    v, o = {}, {}
    for r in rows:
        if not isinstance(r["final"], dict):
            continue
        d = v if r["condition"] == "vanilla" else o
        for w in ("aligned", "opposed"):
            d[(r["case_id"], r["model"], w)] = r["final"][w]
    cats = ["OMITTED", "CITED_WITH_QUALIFIER", "CITED_WITHOUT_QUALIFIER"]
    t = collections.Counter((v[k], o[k]) for k in v if k in o)
    return cats, t, sum(t.values())


NAMES = {
    "TRUTH_STATED": r"\textsc{truth}",
    "CORRECTED": r"\textsc{corrected}",
    "ADOPTED": r"\textsc{adopted}",
    "SILENT": r"\textsc{silent}",
    "OMITTED": r"omitted",
    "CITED_WITH_QUALIFIER": r"cited with",
    "CITED_WITHOUT_QUALIFIER": r"cited without",
}


def block(cats, t):
    head = " & ".join(NAMES[c] for c in cats)
    lines = [rf"Vanilla $\downarrow$ & {head} \\ \midrule"]
    for a in cats:
        cells = " & ".join(str(t[(a, b)]) for b in cats)
        lines.append(rf"{NAMES[a]} & {cells} \\")
    return "\n".join(lines)


def main():
    ccats, ct, cn = court()
    pcats, pt, pn = parliament()
    print(r"\begin{table}[h]\centering\small\setlength{\tabcolsep}{4pt}")
    print(
        r"\caption{Paired Vanilla$\rightarrow$Obligation transitions. Top: "
        f"{cn:,}".replace(",", "{,}")
        + r" Court case--arm--model pairs by the ruling's treatment of the planted error. "
        r"Bottom: "
        + f"{pn:,}".replace(",", "{,}")
        + r" Parliament witness slots (inquiry, model, witness) by how the report handles the "
        r"witness's headline claim. Rows are the Vanilla label, columns the Obligation label. "
        r"Labels use the semantic layer with pattern fallback.}"
    )
    print(r"\label{tab:transitions}")
    print(r"\begin{tabular}{l" + "c" * len(ccats) + r"}\toprule")
    print(block(ccats, ct))
    print(r"\bottomrule\end{tabular}\\[0.8em]")
    print(r"\begin{tabular}{l" + "c" * len(pcats) + r"}\toprule")
    print(block(pcats, pt))
    print(r"\bottomrule\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
