#!/usr/bin/env python3
"""Sensitivity of the Court / Parliament results to the human reading of the label .

For each reader R in {A1, A2, A3, majority of three} we estimate, from the stratified v1 calibration sample, the
confusion P(R label | semantic label) and re-express every model x condition result of the full run under R's reading:

    E[#R=h in cell] = sum_s  n_cell(semantic = s) * P(R = h | semantic = s)

The sample was stratified on the semantic label, so conditioning on it is the correct way to account for the strata;
rates are never computed from raw sample proportions. Assumption (stated in the output): the confusion does not depend
on model or condition. The sample was drawn from 5 models (Court) / 5 models (Parliament); GLM-5.3-Flash, Qwen3.5 and
DeepSeek-V4.1-Flash are extrapolated. Intervals: 2,000 bootstrap draws resampling annotated items within stratum
(calibration uncertainty only; run-level case-cluster uncertainty is in the main tables).

Usage: python3 tools/human_label_sensitivity.py [--json out.json] [--md out.md]
"""
import argparse
import collections
import csv
import io
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data" / "human_annotation"
COURT_L = ["ADOPTED", "CORRECTED", "TRUTH_STATED", "SILENT"]
PARL_L = ["CITED_WITH_QUALIFIER", "CITED_WITHOUT_QUALIFIER", "OMITTED"]
MODELS = ["claude-opus-5", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra", "glm-5.3", "glm-5.3-flash",
          "qwen3.5-35b-a3b", "deepseek-v4.1-flash"]
SAMPLED = {"claude-opus-5", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra", "glm-5.3"}
B = 2000


def rcsv(p):
    return list(csv.DictReader(io.StringIO(Path(p).read_text(encoding="utf-8-sig"))))


def load():
    key = json.loads((QA / "machine_labels.json").read_text())
    ret = {"court": {}, "parl": {}}
    for f in sorted((QA / "returns").glob("*.csv")):
        rows = rcsv(f)
        ann = rows[0]["annotator"]
        if "court" in f.name:
            ret["court"][ann] = {r["id"]: r["label"].strip() for r in rows}
        elif "parliament" in f.name:
            d = {}
            for r in rows:
                d[(r["id"], 0)] = r["witness1_label"].strip()
                d[(r["id"], 1)] = r["witness2_label"].strip()
            ret["parl"][ann] = d
    order = lambda anns: sorted(anns)  # noqa: E731
    # sample items: (item_id, semantic label)
    court_items = [(r["id"], r["gemini"]) for r in key["court"]]
    parl_items = [((r["id"], i), r["gemini"][role]) for r in key["parliament"] for i, role in enumerate(r["witness_order"])]
    readers = {}
    for scen, items, labels in (("court", court_items, COURT_L), ("parl", parl_items, PARL_L)):
        anns = order(ret[scen])
        rd = {}
        for n, a in enumerate(anns, 1):
            rd[f"A{n}"] = {i: (ret[scen][a].get(i) if ret[scen][a].get(i) in labels else None) for i, _ in items}
        maj = {}
        for i, _ in items:
            labs = [rd[f"A{n}"][i] for n in range(1, len(anns) + 1)]
            c = collections.Counter(x for x in labs if x).most_common()
            maj[i] = c[0][0] if c and c[0][1] >= 2 else None
        rd["Majority"] = maj
        readers[scen] = (items, rd, dict(zip([f"A{n}" for n in range(1, len(anns) + 1)], anns)))
    pop = {"court": [json.loads(x) for x in open(ROOT / "results/pooled8/court.jsonl")],
           "parl": [json.loads(x) for x in open(ROOT / "results/pooled8/parliament.jsonl")]}
    return readers, pop


def confusion(items, reader, labels):
    """P(reader | semantic) from items with a reader label; strata with no labelled item fall back to identity."""
    cnt = {s: collections.Counter() for s in labels}
    for i, s in items:
        h = reader.get(i)
        if h:
            cnt[s][h] += 1
    out = {}
    for s in labels:
        n = sum(cnt[s].values())
        out[s] = {h: (cnt[s][h] / n if n else float(h == s)) for h in labels}
    return out


def cells(pop, scen):
    """Semantic-label counts per (model, condition) cell."""
    c = collections.defaultdict(collections.Counter)
    for r in pop:
        if scen == "court":
            c[(r["model"], r["condition"])][r["final"]] += 1
        else:
            for v in r["final"].values():
                c[(r["model"], r["condition"])][v] += 1
    return c


def expected(cnt, conf, labels):
    return {h: sum(cnt[s] * conf[s][h] for s in labels) for h in labels}


def metrics(cellc, conf, scen):
    labels = COURT_L if scen == "court" else PARL_L
    m = {}
    tot = collections.defaultdict(lambda: collections.Counter())
    for (model, cond), cnt in cellc.items():
        e = expected(cnt, conf, labels)
        n = sum(e.values())
        tot[cond].update(e)
        tot["all"].update(e)
        if scen == "court":
            m[(model, cond, "adopt")] = e["ADOPTED"] / n
            m[(model, cond, "correct")] = e["CORRECTED"] / n
        else:
            cited = e["CITED_WITH_QUALIFIER"] + e["CITED_WITHOUT_QUALIFIER"]
            m[(model, cond, "loss")] = e["CITED_WITHOUT_QUALIFIER"] / cited if cited else float("nan")
            m[(model, cond, "cited")] = cited / n
    for cond, e in tot.items():
        n = sum(e.values())
        if scen == "court":
            m[("pooled", cond, "adopt")] = e["ADOPTED"] / n
            m[("pooled", cond, "correct")] = e["CORRECTED"] / n
        else:
            cited = e["CITED_WITH_QUALIFIER"] + e["CITED_WITHOUT_QUALIFIER"]
            m[("pooled", cond, "loss")] = e["CITED_WITHOUT_QUALIFIER"] / cited
    return m


def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def derived(m, scen):
    """Central quantities and conclusions checked in the paper."""
    d = {}
    if scen == "court":
        d["adopt_pooled"] = m[("pooled", "all", "adopt")]
        d["correct_vanilla"] = m[("pooled", "vanilla", "correct")]
        d["correct_obligation"] = m[("pooled", "obligation", "correct")]
        d["obligation_correct_gain"] = d["correct_obligation"] - d["correct_vanilla"]
        d["obligation_adopt_change"] = m[("pooled", "obligation", "adopt")] - m[("pooled", "vanilla", "adopt")]
        van = {mo: m[(mo, "vanilla", "correct")] for mo in MODELS}
        d["opus_top_correct_vanilla"] = float(max(van, key=van.get) == "claude-opus-5")
        d["_adopt_by_model"] = [(m[(mo, "vanilla", "adopt")] + m[(mo, "obligation", "adopt")]) / 2 for mo in MODELS]
        d["_correct_gain_by_model"] = [m[(mo, "obligation", "correct")] - m[(mo, "vanilla", "correct")] for mo in MODELS]
    else:
        d["loss_pooled"] = m[("pooled", "all", "loss")]
        d["loss_vanilla"] = m[("pooled", "vanilla", "loss")]
        d["loss_obligation"] = m[("pooled", "obligation", "loss")]
        d["obligation_loss_reduction"] = d["loss_vanilla"] - d["loss_obligation"]
        van = {mo: m[(mo, "vanilla", "loss")] for mo in MODELS}
        d["opus_lowest_loss_vanilla"] = float(min(van, key=van.get) == "claude-opus-5")
        d["_loss_by_model_vanilla"] = [van[mo] for mo in MODELS]
        d["n_models_obligation_lowers_loss"] = float(sum(m[(mo, "obligation", "loss")] < m[(mo, "vanilla", "loss")] for mo in MODELS))
    return d


def run():
    readers, pop = load()
    out = {}
    for scen in ("court", "parl"):
        items, rd, names = readers[scen]
        labels = COURT_L if scen == "court" else PARL_L
        cellc = cells(pop["court" if scen == "court" else "parl"], scen)
        identity = {s: {h: float(h == s) for h in labels} for s in labels}
        base = derived(metrics(cellc, identity, scen), scen)
        res = {"machine": {"point": base}}
        strata = collections.defaultdict(list)
        for i, s in items:
            strata[s].append((i, s))
        for name, reader in rd.items():
            conf = confusion(items, reader, labels)
            pt = derived(metrics(cellc, conf, scen), scen)
            rng = random.Random(20261107)
            draws = []
            for _ in range(B):
                bs = [x for s in labels for x in (rng.choices(strata[s], k=len(strata[s])) if strata[s] else [])]
                draws.append(derived(metrics(cellc, confusion(bs, reader, labels), scen), scen))
            ci = {}
            for k in pt:
                if k.startswith("_"):
                    continue
                v = sorted(d[k] for d in draws)
                ci[k] = (v[int(0.025 * B)], v[int(0.975 * B) - 1])
            rank_key = "_adopt_by_model" if scen == "court" else "_loss_by_model_vanilla"
            pt["spearman_vs_machine"] = spearman(pt[rank_key], base[rank_key])
            if scen == "court":
                pt["spearman_gain_vs_machine"] = spearman(pt["_correct_gain_by_model"], base["_correct_gain_by_model"])
            sp = sorted(spearman(d[rank_key], base[rank_key]) for d in draws)
            ci["spearman_vs_machine"] = (sp[int(0.025 * B)], sp[int(0.975 * B) - 1])
            res[name] = {"point": pt, "ci95": ci, "confusion": conf,
                         "n_labelled": sum(1 for i, _ in items if reader.get(i))}
        out[scen] = {"readers": names, "results": res}
    return out


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def to_md(out):
    L = ["# Label-reader sensitivity (generated by tools/human_label_sensitivity.py)", "",
         "Confusion P(reader | semantic) estimated on the stratified v1 sample and applied to every model x condition "
         "cell of the full run; assumes the confusion is the same for all models and conditions. 95% intervals: "
         f"{B} bootstrap draws of annotated items within stratum.", ""]
    for scen, title in (("court", "Court"), ("parl", "Parliament")):
        r = out[scen]["results"]
        names = ["machine"] + [k for k in r if k != "machine"]
        keys = [k for k in r["machine"]["point"] if not k.startswith("_")] + ["spearman_vs_machine"]
        if scen == "court":
            keys.append("spearman_gain_vs_machine")
        L += [f"## {title}", "", "| quantity | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for k in keys:
            row = []
            for n in names:
                p = r[n]["point"].get(k)
                if p is None:
                    row.append("—")
                    continue
                c = r[n].get("ci95", {}).get(k)
                row.append(fmt(p) + (f" [{c[0]:.3f}, {c[1]:.3f}]" if c else ""))
            L.append(f"| {k} | " + " | ".join(row) + " |")
        L += ["", "Readers: " + ", ".join(f"{k} = {v}" for k, v in out[scen]["readers"].items()), ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args()
    out = run()
    s = to_md(out)
    print(s)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=list))
    if a.md:
        Path(a.md).write_text(s)
