# Verified injection inputs

Court and Parliament bundles are built from the case corpus and an independently verified injection record per case
(`data/interventions/*.jsonl`, or an explicit `--injection` path). The canonical schema is
`src/ribench_agentic/schemas/injection.schema.json`.

- Court: `arms.petitioner` and `arms.respondent` each contain `anchor_text` and `replacement_text`. One complete
  cleaned excerpt bullet is replaced in the winner brief for A, loser brief for B, asserting the same `error` through
  that side's own excerpt. Optional `arm_side.A/B` must agree with the case verdict.
- Parliament: `design` is `paired_headline_qualification_v1`, with one arm `P`. `witnesses.aligned` and
  `witnesses.opposed` each contain `witness_index`, `witness_name`, `stance_rationale`, `anchor_text`,
  `qualified_text`, `qualifiers`, and `rules`. The index addresses `witness_profiles`, not submissions. Select each
  distinct submission by `submission_id` (or unique normalized `submitter` when ID is absent). Replace each existing
  headline sentence's exact original span after whitespace-normalized matching. Missing or repeated anchors fail
  atomically. `related_conclusion` and the complete injection remain private to the evaluator.
