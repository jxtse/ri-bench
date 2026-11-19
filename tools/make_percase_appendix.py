#!/usr/bin/env python3
"""Export all case strata, role traces and generated longtables from frozen results."""
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from revision_statistics import CONDS, COURT, PARLIAMENT, binary_records, counts, aggregate, outcome, paired_records, witness_records

ROOT = Path(os.environ.get("RIBENCH_ROOT") or Path(__file__).resolve().parents[1])
SCENARIOS = ("court", "open_review_v2", "open_review_v3", "parliament")
LABELS = {"court": "Court", "open_review_v2": "Open Review v2 (historical supplementary)",
          "open_review_v3": "Open Review v3", "parliament": "Parliament"}
NAMES = {"claude-opus-5": "Claude Opus 5", "gpt-5.6-sol": "GPT-5.6 Sol", "gpt-5.6-terra": "GPT-5.6 Terra",
         "gpt-5.6-luna": "GPT-5.6 Luna", "glm-5.3": "GLM-5.3", "glm-5.3-flash": "GLM-5.3-Flash", "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B"}
METRICS = {"tool_calls": "Tool calls", "files_read": "Logged distinct read paths", "output_tokens": "Reported output tokens",
           "reasoning_tokens": "Reported reasoning tokens", "wall_seconds": "Wall seconds", "paper_coverage": "Read-volume proxy"}


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = json.dumps(item, ensure_ascii=False) if isinstance(item, list) else item
    return result


def export(stem, records):
    (ROOT / "results" / f"{stem}.json").write_text(json.dumps(records, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    flattened = [flatten(record) for record in records]
    fields = list(dict.fromkeys(key for record in flattened for key in record))
    with (ROOT / "results" / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def identity(scenario, case_id):
    directory = "open_review" if scenario.startswith("open_review") else scenario
    path = ROOT / "data" / "cases" / directory / f"{case_id}.json"
    if path.exists():
        raw = json.loads(path.read_text())
        for key in ("case_name", "title", "publication_title", "inquiry_title"):
            if raw.get(key):
                return raw[key], str(path.relative_to(ROOT))
    return case_id, "case_id (no title available)"


def scorer_counts(statuses):
    counter = Counter(statuses)
    return dict(verified=counter["verified"], fallback_unverified=counter["unverified"],
                missing=counter[None], other=sum(value for key, value in counter.items() if key not in ("verified", "unverified", None)),
                planned=len(statuses))


def case_export(scenario, case_id, rows, models):
    title, source = identity(scenario, case_id)
    entry = dict(stratum=scenario, case_id=case_id, identity=title, identity_source=source,
                 n_models_planned=len(models), models=models, conditions={}, paired={})
    for condition in CONDS:
        current = [row for row in rows if row["condition"] == condition]
        groups = ("aligned", "opposed") if scenario == "parliament" else sorted({row["arm"] for row in rows})
        cells = {}
        for group in groups:
            selected = current if scenario == "parliament" else [row for row in current if row["arm"] == group]
            labels = [(row.get("final") or {}).get(group) if scenario == "parliament" else
                      row["final"] if scenario == "court" else row["final"].get("decision") for row in selected]
            categories = PARLIAMENT if scenario == "parliament" else COURT if scenario == "court" else ("accept", "reject")
            counter = Counter(labels)
            valid = sum(counter[label] for label in categories)
            cell = dict(n_valid=valid, n_unclear_or_missing=len(models) - valid, n_planned=len(models),
                        categories={label: dict(k=counter[label], n=valid) for label in categories})
            if scenario == "parliament":
                cell["ql"] = counts(aggregate([case_id], witness_records(selected, (group,), "ql")))
                statuses = [((row.get("scorer") or {}).get(group) or {}).get("status") for row in selected]
            elif scenario == "court":
                statuses = [(row.get("scorer") or {}).get("status") for row in selected]
            else:
                statuses = None
                for metric in ("following", "historical_agreement"):
                    cell[metric] = counts(aggregate([case_id], binary_records(selected, metric)))
                references = sorted({row["final"].get("truth", row["final"].get("iclr_decision")) for row in selected} - {None})
                cell["historical_reference"] = references
            if statuses is not None:
                cell["scorer"] = scorer_counts(statuses + [None] * (len(models) - len(selected)))
            cells[group] = cell
        entry["conditions"][condition] = cells

    def pair(name, left, right, metric):
        records, transitions = paired_records(left, right, metric)
        entry["paired"][name] = dict(**counts(aggregate([case_id], records)), **transitions)

    if scenario != "parliament":
        arms = sorted({row["arm"] for row in rows})
        metric = "ADOPTED" if scenario == "court" else "minority" if scenario.endswith("v2") else "away_historical"
        for condition in CONDS:
            current = [row for row in rows if row["condition"] == condition]
            left = {(case_id, row["model"]): row for row in current if row["arm"] == arms[0]}
            right = {(case_id, row["model"]): row for row in current if row["arm"] == arms[1]}
            pair(f"{condition}/{arms[0]}_to_{arms[1]}/{metric}", left, right, metric)
            if scenario == "court":
                pair(f"{condition}/A_to_B/CORRECTED", left, right, "CORRECTED")
        metrics = ("ADOPTED", "CORRECTED") if scenario == "court" else ("following", "historical_agreement")
        for arm in ["both"] + arms:
            current = [row for row in rows if arm == "both" or row["arm"] == arm]
            left = {(case_id, row["model"], row["arm"]): row for row in current if row["condition"] == "vanilla"}
            right = {(case_id, row["model"], row["arm"]): row for row in current if row["condition"] == "obligation"}
            for metric in metrics:
                pair(f"obligation_minus_vanilla/{arm}/{metric}", left, right, metric)
    else:
        for condition in CONDS:
            current = [row for row in rows if row["condition"] == condition]
            valid = [row for row in current if all(row["final"].get(side) in PARLIAMENT for side in ("aligned", "opposed"))]
            entry["paired"][f"{condition}/witness_categories"] = dict(
                n_valid=len(valid), n_missing=len(models) - len(valid), n_planned=len(models),
                counts={f"{aligned}/{opposed}": sum(row["final"]["aligned"] == aligned and row["final"]["opposed"] == opposed for row in valid)
                        for aligned in PARLIAMENT for opposed in PARLIAMENT})
            for metric in ("citation", "ql"):
                aligned = counts(aggregate([case_id], witness_records(current, ("aligned",), metric)))
                opposed = counts(aggregate([case_id], witness_records(current, ("opposed",), metric)))
                entry["paired"][f"{condition}/aligned_minus_opposed/{metric}"] = dict(
                    left=aligned, right=opposed,
                    estimate=aligned["rate"] - opposed["rate"] if aligned["n"] and opposed["n"] else None)
        for side in ("aligned", "opposed"):
            lookup = {(row["model"], row["condition"]): row["final"].get(side) for row in rows}
            pairs = [(lookup.get((model, "vanilla")), lookup.get((model, "obligation"))) for model in models]
            valid = [(before, after) for before, after in pairs if before in PARLIAMENT and after in PARLIAMENT]
            entry["paired"][f"obligation_minus_vanilla/{side}/categories"] = dict(
                n_valid=len(valid), n_missing=len(models) - len(valid), n_planned=len(models),
                counts={f"{before}/{after}": valid.count((before, after)) for before in PARLIAMENT for after in PARLIAMENT})
        for side in ("aligned", "opposed", "both"):
            for metric in ("citation", "ql"):
                sides = ("aligned", "opposed") if side == "both" else (side,)
                margins = {condition: counts(aggregate([case_id], witness_records(
                    [row for row in rows if row["condition"] == condition], sides, metric))) for condition in CONDS}
                before, after = margins["vanilla"], margins["obligation"]
                entry["paired"][f"obligation_minus_vanilla/{side}/{metric}"] = dict(
                    left=after, right=before, estimate=after["rate"] - before["rate"] if after["n"] and before["n"] else None)
    return entry


def describe(values, planned):
    present = [value for value in values if isinstance(value, (float, int)) and np.isfinite(value)]
    quantiles = np.quantile(present, [0.25, 0.5, 0.75]).tolist() if present else [None, None, None]
    return dict(n_valid=len(present), n_missing=planned - len(present), n_planned=planned,
                q1=quantiles[0], median=quantiles[1], q3=quantiles[2])


def tex(value):
    substitutions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                     "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(substitutions.get(char, char) for char in str(value)).replace("\n", " ")


def longtable_start(caption, spec, header, label=None):
    return [r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}",
            r"\begin{longtable}{" + spec + "}", r"\caption{" + caption + "}" + (r"\label{" + label + "}" if label else "") + r"\\",
            r"\toprule " + header + r"\\\midrule\endfirsthead", r"\toprule " + header + r"\\\midrule\endhead",
            r"\midrule\multicolumn{3}{r}{Continued on next page}\\\endfoot", r"\bottomrule\endlastfoot"]


def render_appendix(cases, trace):
    lines = [r"\section{Per-case results and trace statistics}\label{app:percase}",
             "All 477 case--stratum entries are listed below, including zero-citation cases. Counts aggregate models, not independent cases. "
             "The CSV/JSON exports preserve full IDs, exact category numerators and valid denominators, paired transitions and scorer status. "
             "Unknown decisions are excluded, never interpreted as resistance. A rate with no valid denominator is undefined. "
             "No distributional shape or causal susceptibility claim is inferred from these tables.",
             "Each cell lists category counts divided by the valid model denominator. U is unclear/missing out of planned models; "
             "S is scorer verified/fallback-unverified/missing/other. Court: Ado/Cor/Truth/Silent. "
             "Open Review: accept/reject, F=majority following. Parliament: with qualifier/without qualifier/omitted, QL=without qualifier among cited. "
             "A complete-model panel requires the valid denominator to equal the planned model count."]
    for scenario in SCENARIOS:
        selected = [entry for entry in cases if entry["stratum"] == scenario]
        lines.extend(longtable_start(tex(LABELS[scenario]) + f": all {len(selected)} cases.",
                                    r"p{0.38\linewidth}p{0.28\linewidth}p{0.28\linewidth}", "Case identity and full ID & Vanilla & Obligation"))
        for entry in selected:
            cells = []
            for condition in CONDS:
                parts = []
                for group, cell in entry["conditions"][condition].items():
                    values = "/".join(str(value["k"]) for value in cell["categories"].values())
                    parts.append(tex(group) + ": (" + values + f")/{cell['n_valid']}; U {cell['n_unclear_or_missing']}/{cell['n_planned']}")
                    if "following" in cell:
                        parts.append(f"F {cell['following']['k']}/{cell['following']['n']}")
                    if "ql" in cell:
                        parts.append(f"QL {cell['ql']['k']}/{cell['ql']['n']}" + (" (undefined)" if not cell['ql']['n'] else ""))
                    if "scorer" in cell:
                        parts.append("S " + "/".join(str(cell["scorer"][key]) for key in ("verified", "fallback_unverified", "missing", "other")))
                cells.append(r"\newline ".join(parts))
            identity_cell = tex(entry["identity"]) + r"\newline {\tiny\path|" + entry["case_id"] + "|}"
            lines.append(identity_cell + " & " + " & ".join(cells) + r"\\[3pt]")
        lines.append(r"\end{longtable}\endgroup")
    lines.extend([r"\paragraph{Paired movements.}",
                  "Machine-readable per-case exports include every arm and prompt comparison. The following full-case tables give the primary "
                  "arm transition: positive/negative/unchanged/missing (valid/planned). Positive is toward the minority in v2, away from the historical "
                  "reference in v3, and increased adoption in Court (B minus A). Parliament lists the full aligned-by-opposed category matrix, "
                  "row-major in with-qualifier/without-qualifier/omitted order, followed by valid/planned reports."])
    for scenario in SCENARIOS:
        lines.extend(longtable_start(tex(LABELS[scenario]) + ": paired per-case counts.",
                                    r"p{0.40\linewidth}p{0.27\linewidth}p{0.27\linewidth}", "Full case ID & Vanilla & Obligation"))
        for entry in [entry for entry in cases if entry["stratum"] == scenario]:
            cells = []
            for condition in CONDS:
                value = next(value for key, value in entry["paired"].items() if key.startswith(condition + "/"))
                if scenario == "parliament":
                    text = ", ".join(str(number) for number in value["counts"].values())
                else:
                    text = "/".join(str(value[key]) for key in ("positive", "negative", "unchanged", "missing"))
                cells.append(text + f" ({value['n_valid']}/{value['n_planned']})")
            lines.append(r"{\tiny\path|" + entry["case_id"] + "|} & " + " & ".join(cells) + r"\\[2pt]")
        lines.append(r"\end{longtable}\endgroup")
    lines.extend([r"\paragraph{Per-case prompt transitions and witness gaps.}",
                  "Each prompt-transition entry is positive/negative/unchanged/missing (valid/planned), Obligation minus Vanilla, "
                  "separately by arm and endpoint. Ado=adoption, Cor=correction, F=following, H=historical agreement. "
                  "Parliament reports exact conditional-ratio margins (Obligation minus Vanilla, or aligned minus opposed); "
                  "a zero denominator is undefined. Full category-to-category prompt transitions are retained in the CSV/JSON."])
    for scenario in SCENARIOS:
        lines.extend(longtable_start(tex(LABELS[scenario]) + ": per-case prompt comparisons and witness gaps.",
                                    r"p{0.36\linewidth}p{0.31\linewidth}p{0.27\linewidth}", "Full case ID & Obligation minus Vanilla & Additional endpoints / gaps"))
        for entry in [entry for entry in cases if entry["stratum"] == scenario]:
            primary, secondary = [], []
            for key, value in entry["paired"].items():
                if scenario == "parliament":
                    if "estimate" not in value:
                        continue
                    left, right = value["left"], value["right"]
                    text = f"{left['k']}/{left['n']} - {right['k']}/{right['n']}"
                    if value["estimate"] is None:
                        text += " (undefined)"
                    if key.startswith("obligation_minus_vanilla/"):
                        primary.append(tex(key.removeprefix("obligation_minus_vanilla/")) + ": " + text)
                    else:
                        secondary.append(tex(key.replace("aligned_minus_opposed/", "A-O/")) + ": " + text)
                elif key.startswith("obligation_minus_vanilla/") and "/both/" not in key:
                    metric = key.split("/")[-1]
                    short = {"ADOPTED": "Ado", "CORRECTED": "Cor", "following": "F", "historical_agreement": "H"}[metric]
                    text = key.split("/")[1] + "/" + short + ": " + "/".join(str(value[field]) for field in ("positive", "negative", "unchanged", "missing"))
                    text += f" ({value['n_valid']}/{value['n_planned']})"
                    (primary if metric in ("ADOPTED", "following") else secondary).append(text)
                elif scenario == "court" and key.endswith("A_to_B/CORRECTED"):
                    secondary.append(tex(key.split("/")[0]) + " B-A/Cor: " + "/".join(str(value[field]) for field in ("positive", "negative", "unchanged", "missing"))
                                     + f" ({value['n_valid']}/{value['n_planned']})")
            lines.append(r"{\tiny\path|" + entry["case_id"] + "|} & " + r"\newline ".join(primary) + " & " + r"\newline ".join(secondary) + r"\\[3pt]")
        lines.append(r"\end{longtable}\endgroup")
    lines.extend([r"\paragraph{Trace statistics.}",
                  "Every scenario, model and senior role is included, including area chair (AC) and program chair (PC) in both Open Review strata. "
                  "Values are median [Q1, Q3] using linear quantiles, with observed/planned counts; missing measurements are not zero-filled. "
                  "Files read means logged distinct read paths, not independently verified unique files or complete reading. "
                  "Reported output and reasoning tokens are separate provider fields; they are not summed or asserted comparable across providers. "
                  "Read-volume proxy is the logged paper coverage measure, not unique or complete reading. "
                  "The trace-statistics raw CSV/JSON preserves all case and episode IDs and role-level measurements."])
    lines.extend(longtable_start("Role-level trace statistics; median [Q1, Q3] and observed/planned.",
                                r"p{0.43\linewidth}p{0.24\linewidth}p{0.27\linewidth}", "Scenario / model / role & Metric & Median [IQR]; n/N", "tab:app-trace"))
    for entry in trace:
        for index, (metric, value) in enumerate(entry["metrics"].items()):
            label = tex(LABELS[entry["scenario"]] + " / " + NAMES[entry["model"]] + " / " + entry["role"].replace("_", " ")) if index == 0 else ""
            display = "undefined" if value["median"] is None else f"{value['median']:,.2f} [{value['q1']:,.2f}, {value['q3']:,.2f}]"
            lines.append(label + " & " + tex(METRICS[metric]) + " & " + display + f"; {value['n_valid']}/{value['n_planned']}" + r"\\")
    lines.append(r"\end{longtable}\endgroup")
    return "\n".join(lines) + "\n"


def render_compact_appendix(cases, trace):
    """Print each case once; keep full category/transition/telemetry detail in CSV/JSON."""
    lines=[r'\section{Per-case results and trace statistics}\label{app:percase}',
           'All 477 case--stratum entries are listed once below, including zero-citation cases. '
           'Full case IDs join directly to the accompanying CSV/JSON, which retain names, every category, scorer status, '
           'all paired and prompt transitions, and raw role telemetry. Unknown decisions are excluded, never counted as resistance. '
           'Court cells are adoption counts by arm; review cells are majority-following counts by arm; '
           'Parliament cells give loss counts among cited claims (L), with citation counts among valid witness slots (C). '
           'Every displayed fraction is numerator/valid denominator across models. A zero denominator is undefined, not zero.']
    for scenario in SCENARIOS:
        selected=[x for x in cases if x['stratum']==scenario]
        lines.extend(longtable_start(tex(LABELS[scenario])+f': all {len(selected)} cases; primary endpoint counts.',
                                    r'p{0.46\linewidth}p{0.23\linewidth}p{0.23\linewidth}', 'Full case ID & Vanilla & Obligation'))
        for e in selected:
            cells=[]
            for cond in CONDS:
                parts=[]
                for arm,cell in e['conditions'][cond].items():
                    if scenario=='court': v=cell['categories']['ADOPTED']
                    elif scenario.startswith('open_review'): v=cell['following']
                    else:
                        v=cell['ql']; c={'k':v['n'],'n':cell['n_valid']}
                        ratio=f"{v['k']}/{v['n']}" if v['n'] else 'undefined (0/0)'
                        parts.append(f"{arm[0]}: L {ratio}; C {c['k']}/{c['n']}");continue
                    parts.append(f"{arm}: {v['k']}/{v['n']}" if v['n'] else f'{arm}: undefined (0/0)')
                cells.append(r'\newline '.join(parts))
            lines.append(r'\path|'+e['case_id']+'| & '+' & '.join(cells)+r'\\[1pt]')
        lines.append(r'\end{longtable}\endgroup')
    lines.extend([r'\paragraph{Complete role-level trace statistics.}',
                  'Every scenario/model/role group is included. Values are median [Q1, Q3] with observed/planned counts; '
                  'missing telemetry is not zero-filled. Logged paths do not prove unique file coverage or comprehension. '
                  'Provider output and reasoning tokens are shown separately, never summed; cross-provider conventions differ. '
                  'Full case/episode-linked measurements are in the trace-statistics CSV/JSON.'])
    lines.extend(longtable_start('All role-level trace statistics; median [Q1,Q3] and observed/planned.',
                                r'p{0.40\linewidth}p{0.25\linewidth}p{0.29\linewidth}',
                                'Scenario / model / role & Metric & Median [IQR]; n/N','tab:app-trace'))
    for e in trace:
        for index,(metric,v) in enumerate(e['metrics'].items()):
            label=tex(LABELS[e['scenario']]+' / '+NAMES[e['model']]+' / '+e['role'].replace('_',' ')) if index==0 else ''
            display='undefined' if v['median'] is None else f"{v['median']:,.2f} [{v['q1']:,.2f}, {v['q3']:,.2f}]"
            lines.append(label+' & '+tex(METRICS[metric])+' & '+display+f"; {v['n_valid']}/{v['n_planned']}"+r'\\')
    lines.append(r'\end{longtable}\endgroup')
    return '\n'.join(lines)+'\n'


def main():
    datasets = {scenario: [json.loads(line) for line in (ROOT / "results" / f"{scenario}.jsonl").read_text().splitlines()] for scenario in SCENARIOS}
    cases, trace, raw = [], [], []
    for scenario, rows in datasets.items():
        by_case = defaultdict(list)
        for row in rows:
            by_case[row["case_id"]].append(row)
        models = sorted({row["model"] for row in rows})
        cases.extend(case_export(scenario, case_id, by_case[case_id], models) for case_id in sorted(by_case))
        roles = ("area_chair", "program_chair") if scenario.startswith("open_review") else ("judge",) if scenario == "court" else ("chair",)
        for model in models:
            selected = sorted((row for row in rows if row["model"] == model), key=lambda row: (row["case_id"], row["condition"], row["arm"], row["episode_id"]))
            for role in roles:
                values = [(row.get("roles") or {}).get(role) or {} for row in selected]
                trace.append(dict(scenario=scenario, model=model, role=role, n_episodes=len(selected),
                                  metrics={metric: describe([value.get(metric) for value in values], len(selected)) for metric in METRICS}))
                for row, value in zip(selected, values):
                    raw.append(dict(scenario=scenario, model=model, role=role, case_id=row["case_id"], episode_id=row["episode_id"],
                                    condition=row["condition"], arm=row["arm"], role_present=bool(value),
                                    **{metric: value.get(metric) for metric in METRICS},
                                    **{metric + "_missing": value.get(metric) is None for metric in METRICS}))
    assert Counter(entry["stratum"] for entry in cases) == dict(court=120, open_review_v2=119, open_review_v3=118, parliament=120)
    export("per_case", cases)
    export("trace_statistics", trace)
    export("trace_statistics_raw", raw)
    (ROOT / "build" / "tables" / "app_percase.tex").write_text(render_compact_appendix(cases, trace))
    print(f"Exported {len(cases)} cases, {len(trace)} scenario/model/role groups, {len(raw)} role traces")


if __name__ == "__main__":
    main()
