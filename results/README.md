# results/ — frozen per-episode results of the RI-Bench agentic experiments

One JSON line per completed episode, exported from the raw run directories by `tools/export_results.py`
and summarized by `tools/summarize_results.py`.
Model identifiers are `<route>/<model>`; see the main README for the route names. They were renamed from internal
gateway labels for this release; nothing else in the records changed. Primary endpoint counts and case-cluster
intervals are computed from these four files; the figure script reads only this directory.
Per-case identity enrichment uses `data/cases/`; prompt and new conclusion-audit
verification additionally use the archived raw invocations/reports.

| File | Rows | What one row is |
|---|---:|---|
| `court.jsonl` | 3,360 | one judge episode: 120 cases × 2 arms (error in prevailing / losing side's excerpts) × 2 prompts × 7 models |
| `open_review_v2.jsonl` | 2,380 | one AC+PC episode with real 3:1 votes: 119 papers × 2 arms (A: majority well-argued, B: minority well-argued) × 2 prompts × 5 models |
| `open_review_v3.jsonl` | 3,304 | one AC+PC episode with constructed votes: 118 papers × 2 arms (C: control, X: adversarial) × 2 prompts × 7 models |
| `parliament.jsonl` | 1,680 | one committee-chair episode: 120 inquiries × 2 prompts × 7 models |
| `summary.json` | | primary counts and paired contrasts; `case_cluster` contains the corrected whole-case bootstrap intervals. Legacy Wilson/McNemar fields are retained for compatibility, not pooled inference |
| `RUNS_INDEX.json` | | row counts and SHA-256 per file, and the source batch directory of every row |

Rows are sorted by (model, condition, arm, case_id). Row counts equal the frozen plans exactly; the exporter exits non-zero otherwise.

## Columns

- `scenario`, `episode_id` (SHA-256 of the frozen inputs), `batch` (source lane directory), `case_id`, `arm`, `condition` (`vanilla` / `obligation`).
- `model` (short id, e.g. `gpt-5.6-terra`), `model_requested` (provider-prefixed id as sent).
- `final`: the endpoint.
  - Court: one of `ADOPTED`, `CORRECTED`, `TRUTH_STATED`, `SILENT` (how the ruling treats the planted procedural-fact error).
  - Open Review: `decision`, `recommendation`, `majority_stance`, `majority_alignment` (decision followed the 3:1 majority); v3 adds `truth` (real outcome) and `truth_alignment`.
  - Parliament: `aligned` / `opposed` witness each in `CITED_WITH_QUALIFIER`, `CITED_WITHOUT_QUALIFIER`, `OMITTED`.
- `pattern_label`: the deterministic layer's full record. Court: `{classification, ruling, ambiguous}`; Parliament: `{aligned, opposed, both_cited, selective_omission, ambiguous}`; Open Review: the parsed decision object (same keys as `final`, plus `truth`/`truth_alignment` in v3).
- `scorer` (Court, Parliament; `null` for Open Review): the semantic layer's record from Gemini 3.8 Flash. Court: `{classification, quote, reason, status, model, provider, requested_model}`. Parliament: one such record per witness (`aligned`, `opposed`) with `cited`, `qualifiers_preserved`, `hedge_present` instead of `classification`; `verification_error` appears on failed evidence checks. `verified` means accepted by the frozen checker, not human-validated accuracy: SILENT/uncited labels can pass without a quote. Only accepted judgments override the pattern label. Raw model output and token usage are retained in the run directory, not in these exports.
- `agreement` (Court only): pattern classification == scorer classification; `null` elsewhere.
- v2 rows carry `final.iclr_decision` (`accept`/`reject`/absent when ICLR recorded none), the evaluator-only real outcome frozen here so `summary.json` is computable from this directory alone.
- `episode_status`: always `complete` in this export; error episodes are excluded and archived under `runs/agentic/*/retried_episodes/`.
- `roles.<role>`: per senior role (`judge`; `area_chair`, `program_chair`; `chair`): `status`, `model_returned`, `tool_calls`, `files_read` (distinct logged read paths), token fields, `wall_seconds`, `integrity_error`, and legacy `paper_coverage`. That last field is capped serialized-JSON tool-result bytes divided by paper bytes, for successful calls targeting the paper path. It includes wrapper overhead and repeated chunks, not unique reading coverage.

Token accounting is provider-dependent: one route reports only uncached input and output
may already include reasoning. Output and reasoning are reported separately, never added.
These are final-artifact role measurements, not a complete bill including retries; wall
time can include provider waiting and is not isolated inference time.

## Exclusions already applied

- The route `foundry-a` served Sol under the Terra identifier (`foundry-a/gpt-5.6-terra`, 609 episodes); those episodes are excluded, and Terra was rerun on route `foundry-b`, which returns its identity.
- TokenDance Qwen batches (404 valid + 407 failed attempts) are excluded; Qwen3.5-35B-A3B results come from OpenRouter pinned to Alibaba.
- Episodes with a provider error, empty output, unsettled tool invocation or out-of-workspace access were archived and rerun; the archived attempts remain in `retried_episodes/` and are counted in the paper's infrastructure disclosures, not here.
- Four Parliament inquiries whose material exceeded the agents' context budget were excluded before analysis (`configs/agentic/parliament_exclusions.json`).

## Regenerating

```
.venv/bin/python tools/export_results.py      # needs the raw run directories under runs/agentic (not included); validates counts before touching results/
.venv/bin/python tools/summarize_results.py   # needs only results/*.jsonl
RIBENCH_RESULTS_DIR=$PWD/results/revision8 .venv/bin/python tools/make_figures.py
```

## Additional analyses

- `per_case.{csv,json}`: all 477 case–stratum records, both prompts, counts, missingness and paired transitions. Each case appears once in the printed primary-endpoint appendix; the full detail remains in these files.
- `trace_statistics.{csv,json}`: all 38 scenario/model/role groups; `trace_statistics_raw.{csv,json}` contains 16,408 final-role records and explicit missing fields.
- `scoring_audit.json`: complete scorer/fallback counts, separate pattern–final and pattern–accepted-semantic matrices, and the 0/1/2-limitation breakdown.
- `prompt_template_audit.json`: exact static-template checks against 76 selected frozen invocations, not a claim that all individual prompts were independently inspected.
- `parliament_conclusion_audit*`: a post hoc 20-inquiry/280-report sample, not the full population. Complete dispositions include unresolved technical failures and non-comparable government-response references. Labels are not human-calibrated.

After `tools/summarize_results.py`, regenerate printable tables with
`tools/make_revision_tables.py`, `tools/make_scoring_audit.py`, and
`tools/make_percase_appendix.py`. These commands make no model calls.
`tools/summarize_conclusion_audit.py` revalidates retained audit receipts and likewise makes
no model calls, but requires those local raw receipts. Do not run the separate audit
collection script as part of routine paper reproduction. `make paper` compiles the
already-generated TeX; first-time TeX installations may need the standard `longtable` package.

## Eight-model and follow-up views

- `supplement/deepseek/`: DeepSeek-V4.1-Flash on the same frozen plans (official API), exported by `tools/export_results_deepseek.py`.
- `pooled8/`: the seven-model files above concatenated with the DeepSeek supplement (`tools/build_pooled8.sh`).
- `rebuttal/`: OpenReview follow-up designs on de-identified paper text, exported by `tools/export_rebuttal.py` and analysed by `tools/analyze_rebuttal.py`. `or_r` reruns the constructed-vote design; `or_e1` gives the chairs the paper only; `or_e2a` withholds review scores; `or_e2b` balances specific and generic reviews 2:2; `or_e3a`, `or_e3b`, `or_e3c` vary the majority size and position. `RUNS_INDEX.json` lists planned and exported counts per model, with every incomplete episode and its reason.
- `revision8/`: the view behind the paper's main tables and figures: `pooled8` with `open_review_v3` replaced by `rebuttal/or_r.jsonl` (`tools/build_revision8.sh`). Run `make summary` and `make tables` to regenerate `summary.json` and the tables from it.
