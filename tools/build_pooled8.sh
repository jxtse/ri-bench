#!/usr/bin/env bash
# Build the eight-model pooled view used by the manuscript's pooled numbers.
# results/pooled8/<scenario>.jsonl = frozen seven-model export (results/*.jsonl, untouched)
#                                   + supplementary DeepSeek-V4.1-Flash export (results/supplement/deepseek/*.jsonl).
# Then recompute summary.json for that view with the unchanged summarizer. Deterministic; no model calls.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/pooled8
mkdir -p "$OUT"
for s in court open_review_v2 open_review_v3 parliament; do
  cat "results/$s.jsonl" "results/supplement/deepseek/$s.jsonl" > "$OUT/$s.jsonl"
done
python3 - <<'EOF'
import json
json.dump({"models": ["claude-opus-5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "glm-5.3", "glm-5.3-flash",
                      "qwen3.5-35b-a3b", "deepseek-v4.1-flash"],
           "note": "Eight-model pooled view: concatenation of the frozen seven-model export results/*.jsonl and the "
                   "supplementary DeepSeek-V4.1-Flash export results/supplement/deepseek/*.jsonl. Built by "
                   "tools/build_pooled8.sh; no model calls."},
          open("results/pooled8/RUNS_INDEX.json", "w"), indent=1)
EOF
RIBENCH_RESULTS_DIR="$PWD/$OUT" .venv/bin/python tools/summarize_results.py
