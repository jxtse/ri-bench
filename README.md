# RI-Bench

Code, results and paper source for **"Do Senior Agents Do Their Job? Measuring Role Integrity in How LLM Agents Use
Subordinate Material"** (Xie, Zhang, Jiang, Liu, Yang and Su, 2026).

Multi-agent LLM systems hand long tasks to subagents whose reports a senior agent turns into the final answer. RI-Bench
measures whether that senior agent meets the obligations of its role toward the material it receives. Each scenario
takes a real institutional case, changes one property of the subordinate material, and reads the response from the
senior agent's own decision, ruling or report. Senior agents run as tool-using agents over an isolated case directory,
and every file they read is logged.

| Scenario | Senior role | Obligation | Intervention | Endpoint |
|---|---|---|---|---|
| OpenReview (ICLR 2024–2026) | area chair, program chair | weigh arguments, not votes | constructed votes: a majority of generic reviews against a minority of specific ones | how often the decision follows the majority |
| Court (U.S. Supreme Court) | judge | verify counsel against the record | one procedural-fact error planted in one side's argument | how often the ruling adopts the error |
| Parliament (UK select committees) | committee chair | relay evidence with its limitations | two stated limitations added to two opposed witnesses' claims | how often a cited claim loses its limitation |

Eight models (Claude Opus 5, GPT-5.6 Sol / Terra / Luna, GLM-5.3, GLM-5.3-Flash, Qwen3.5-35B-A3B,
DeepSeek-V4.1-Flash), two prompts (Vanilla, Obligation), 12,392 episodes.

- **Case corpus:** [huggingface.co/datasets/jxtse/RI-Bench](https://huggingface.co/datasets/jxtse/RI-Bench)
  (1,636 cases, CC BY-NC 4.0)
- **Paper:** `manuscript/` (LaTeX source)

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```sh
make bootstrap   # .venv with the test and figure extras
make test        # unit tests, no model calls
make summary     # recompute results/summary.json and results/revision8/summary.json
make tables      # regenerate the paper's tables from results/
make figures     # redraw the result figures
make paper       # compile the paper (needs a TeX installation with latexmk)
```

`make summary`, `make tables` and `make figures` read only the frozen per-episode exports in `results/` and make no
model calls. To build case bundles or run episodes you also need the case corpus:

```sh
make cases       # download the Hugging Face dataset into data/cases/
```

## Repository layout

```
src/ribench_agentic/   pipeline: case bundles, pi harness runner, alternative harnesses, extraction, scoring
tests/agentic/         unit tests (no model calls)
tools/                 export, statistics, table and figure scripts; case sampling and stimulus generation
configs/agentic/       frozen case lists, run plans, exclusions, pi harness configuration template
results/               frozen per-episode exports and summaries (see results/README.md)
data/interventions/    generated and verified Court errors and Parliament limitations, one record per case
data/generated_reviews/ the generated OpenReview reviews for every design and arm
data/human_annotation/ human calibration of the semantic scorer (3 annotators, anonymized)
manuscript/            paper source and figure script
```

### Results views

| Directory | Contents | Used for |
|---|---|---|
| `results/*.jsonl` | the seven-model runs: Court, Parliament, OpenReview real-vote (`open_review_v2`) and constructed-vote (`open_review_v3`) | inputs to the views below |
| `results/supplement/deepseek/` | DeepSeek-V4.1-Flash on the same plans | added to the seven-model runs |
| `results/pooled8/` | seven-model runs + DeepSeek | Court, Parliament and real-vote tables |
| `results/rebuttal/` | the constructed-vote design rerun on de-identified paper text (`or_r`) and the OpenReview control designs (`or_e1`–`or_e3c`) | constructed-vote results and controls |
| `results/revision8/` | `pooled8` with `open_review_v3` replaced by `or_r` | the paper's main tables and figures |

Each view has a `RUNS_INDEX.json` with row counts and SHA-256 hashes. Columns are documented in `results/README.md`.

### Serving routes

Model identifiers in plans and results have the form `<route>/<model>`, for example `foundry-a/claude-opus-5`.
`foundry-a` and `foundry-b` are two Microsoft Foundry deployments; `mint` is Mind Lab's MinT endpoint; `ark` is
Volcengine Ark; `openrouter` is OpenRouter pinned to one upstream provider; `deepseek` is DeepSeek's official API;
`vllm` is the open weights served with vLLM 0.29.0. The paper's appendix lists which route served which table. Every
episode also records the model identifier the provider returned (`roles.<role>.model_returned`).

## Running episodes

The senior agent runs in the [pi](https://github.com/badlogic/pi-mono) coding-agent runtime restricted to read-only
`read`, `ls` and `grep` inside its case directory. Copy `configs/agentic/pi-agent/models.example.json` to
`models.json` in the same directory, fill in your endpoints, and export the API keys it references. The semantic scorer
reads `RIBENCH_SCORER_URL`, `RIBENCH_SCORER_MODEL` and `RIBENCH_SCORER_API_KEY` (or set `RIBENCH_SCORER=openrouter`
with `OPENROUTER_API_KEY`).

```sh
make cases
uv run ribench-agentic build-bundles --scenario court --injections data/interventions/court.jsonl --cases-root data/cases
uv run ribench-agentic controller --plan configs/agentic/plans/smoke_court.json --resume
uv run ribench-agentic extract --runs-root runs/agentic/<batch>
uv run ribench-agentic score --runs-root runs/agentic/<batch>
```

Rerunning the paper's episodes requires access to the same model versions, whose serving may change. The raw traces
of the paper's runs (about 20 GB) are not included.

## License

Code: MIT (`LICENSE`). Results, generated reviews, interventions and annotations in this repository: CC BY-NC 4.0. The
case corpus on Hugging Face is CC BY-NC 4.0, and its underlying sources keep their own terms (OpenReview CC BY 4.0, UK
Parliament Open Parliament Licence v3.0, Oyez CC BY-NC 4.0).

## Citation

```bibtex
@misc{xie2026ribench,
  title  = {Do Senior Agents Do Their Job? Measuring Role Integrity in How {LLM} Agents Use Subordinate Material},
  author = {Xie, Jinxiang and Zhang, Xinyi and Jiang, Changjiang and Liu, Kexu and Yang, Qi and Su, Yuxin},
  year   = {2026},
  url    = {https://github.com/jxtse/ri-bench}
}
```
