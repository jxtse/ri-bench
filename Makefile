.DEFAULT_GOAL := help
PYTHON := $(CURDIR)/.venv/bin/python
export PYTHONDONTWRITEBYTECODE := 1
export PYTHONPATH := $(CURDIR)/src

.PHONY: help bootstrap cases test summary tables figures
help:
	@printf '%s\n' \
	  'bootstrap: create .venv with uv (test and figures extras)' \
	  'cases:     download the case corpus from Hugging Face into data/cases/' \
	  'test:      unit tests for the agentic pipeline (no model calls)' \
	  'summary:   recompute results/summary.json and results/revision8/summary.json from the frozen exports' \
	  'tables:    regenerate the paper tables from results/ into build/tables/' \
	  'figures:   redraw the result figures from results/revision8 into build/figures/'
bootstrap:
	uv sync --extra test --extra figures --link-mode copy
cases:
	uv run --no-project --with 'huggingface_hub>=0.24' python tools/fetch_cases.py
test:
	mkdir -p build/test-tmp
	$(PYTHON) -m pytest tests -q
summary:
	$(PYTHON) tools/summarize_results.py
	RIBENCH_RESULTS_DIR=$(CURDIR)/results/revision8 $(PYTHON) tools/summarize_results.py
tables:
	mkdir -p build/tables
	RIBENCH_RESULTS_DIR=$(CURDIR)/results/revision8 $(PYTHON) tools/make_ri_table.py > build/tables/table_ri.tex
	RIBENCH_RESULTS_DIR=$(CURDIR)/results/revision8 $(PYTHON) tools/make_revision_tables.py
	$(PYTHON) tools/make_rebuttal_tables.py > /dev/null
figures:
	RIBENCH_RESULTS_DIR=$(CURDIR)/results/revision8 $(PYTHON) tools/make_figures.py
