#!/usr/bin/env bash
# Build results/revision8, the eight-model view used by the paper.
# court / open_review_v2 / parliament: copied from results/pooled8 (unchanged).
# open_review_v3: results/rebuttal/or_r.jsonl (constructed-vote design rerun on de-identified paper text), mapped to the
# v3 schema (scenario name, DeepSeek model id) with area/program-chair paper_coverage recomputed from runs/agentic.
# Then summary.json, the paper tables and figures. Deterministic; no model calls.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="env -u PYTHONPATH -u VIRTUAL_ENV .venv/bin/python"
OUT=results/revision8; mkdir -p "$OUT"
for s in court open_review_v2 parliament; do cp "results/pooled8/$s.jsonl" "$OUT/$s.jsonl"; done
$PY - <<'PYEOF'
import json, os
R = os.getcwd(); OUT = "results/revision8"
rows = [json.loads(l) for l in open("results/rebuttal/or_r.jsonl")]
with open(f"{OUT}/open_review_v3.jsonl", "w") as f:
    for r in rows:
        x = dict(r); x["scenario"] = "open_review_v3"; x.pop("lane", None)
        if x["model"] == "deepseek-flash": x["model"] = "deepseek-v4.1-flash"
        x.setdefault("agreement", None); x.setdefault("scorer", None)
        ed = f"runs/agentic/{x['batch']}/episodes/{x['episode_id']}"
        ft = f"data/agentic/or_r/{x['case_id']}__{x['arm']}/docs/paper/fulltext.md"
        for role in ("area_chair", "program_chair"):
            e = json.load(open(f"{ed}/{role}/episode.json"))
            got = sum(int(t.get("result_size") or 0) for t in e.get("tools") or []
                      if (t.get("args") or {}).get("path") == "docs/paper/fulltext.md" and not t.get("isError"))
            x["roles"][role]["paper_coverage"] = round(min(1.0, got / os.path.getsize(ft)), 4)
        f.write(json.dumps(x, sort_keys=True) + "\n")
json.dump({"models": ["claude-opus-5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "glm-5.3", "glm-5.3-flash",
                      "qwen3.5-35b-a3b", "deepseek-v4.1-flash"],
           "note": "Revision view: court/open_review_v2/parliament = results/pooled8 (unchanged); open_review_v3 = "
                   "results/rebuttal/or_r.jsonl (constructed-vote design on de-identified paper text, 8 models). "
                   "Built by tools/build_revision8.sh; no model calls."}, open(f"{OUT}/RUNS_INDEX.json", "w"), indent=1)
PYEOF
export RIBENCH_RESULTS_DIR="$PWD/$OUT"
$PY tools/summarize_results.py
$PY tools/make_revision_tables.py
$PY tools/make_ri_table.py > manuscript/sections/table_ri.tex
$PY tools/make_rebuttal_tables.py > /dev/null
$PY manuscript/figures/make_results_figures.py
