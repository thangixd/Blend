PYTHON ?= .venv/bin/python
COMPOSE ?= docker compose

.DEFAULT_GOAL := help
.PHONY: help install index test test-index test-clean clean \
        benchmark-prepare benchmark-index benchmark benchmark-clean \
        benchmark-all benchmark-compare benchmark-clean-all \
        benchmark-test benchmark-test-vllm \
        build up down shell bench logs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install python dependencies via uv.
	uv sync

index:  ## Build Blend value-index + NLSeeker NL-index over data/lake/*.csv (production: blend.duckdb + nl-out/).
	$(PYTHON) scripts/create_blend_index.py

test: test-index  ## Run the full pytest suite (rebuilds the test indexes if missing).
	$(PYTHON) -m pytest

# `test-index` is what the integration tests in tests/test_blend_plans_lake.py
# read. Build paths are deliberately separate from the production
# `make index` artifacts so a developer can keep their own blend.duckdb
# while still running the suite.
test-index: test_duckdb test_nl-out/.built  ## Build the integration-test indexes (test_duckdb + test_nl-out/).
	@echo "Test indexes ready: $(CURDIR)/test_duckdb + $(CURDIR)/test_nl-out"

test_duckdb test_nl-out/.built:
	@echo "Building integration-test indexes over data/lake/*.csv ..."
	$(PYTHON) scripts/create_blend_index.py
	mv blend.duckdb test_duckdb
	mv nl-out test_nl-out
	@$(PYTHON) -c "import duckdb; \
con = duckdb.connect('test_duckdb', read_only=False); \
con.execute(\"UPDATE blend_nl_indexes SET vector_path = REPLACE(vector_path, '/nl-out/', '/test_nl-out/'), fulltext_path = REPLACE(fulltext_path, '/nl-out/', '/test_nl-out/')\"); \
con.close(); \
print('  rewired blend_nl_indexes registry → test_nl-out paths')"
	mkdir -p test_nl-out
	touch test_nl-out/.built

test-clean:  ## Drop the integration-test indexes (test_duckdb + test_nl-out/).
	rm -f test_duckdb
	rm -rf test_nl-out

clean:  ## Drop the production value-index and NL-index artifacts (blend.duckdb + nl-out/).
	rm -f blend.duckdb
	rm -rf nl-out

DATASET ?= adventure_works
BENCH_PY = $(PYTHON) -m scripts.benchmark.cli

BENCHMARK_DATASETS ?= adventure_works chembl public_bi chicago_open fetaqa

benchmark-prepare:  ## Extract tar+zip into benchmark-data/lakes/$(DATASET)/.
	$(BENCH_PY) prepare --dataset $(DATASET)


benchmark-index: benchmark-prepare  ## Build NLSeeker-only index over the prepared lake (value_index=False).
	$(BENCH_PY) build --dataset $(DATASET)

benchmark: benchmark-index  ## Run the benchmark and write benchmark-data/results/$(DATASET)/<ts>/.
	PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 $(BENCH_PY) run --dataset $(DATASET)

# `benchmark-all` runs every dataset in BENCHMARK_DATASETS sequentially.
# Each dataset is a fresh `make benchmark` invocation, so a failure on
# dataset N still leaves N-1's results on disk. After the loop finishes
# we run the comparison summary.
benchmark-all:  ## Run benchmark for every dataset, then print paper comparison.
	@set -e; \
	for ds in $(BENCHMARK_DATASETS); do \
	  echo ""; \
	  echo "============================================================"; \
	  echo "  benchmark-all: starting $$ds"; \
	  echo "============================================================"; \
	  $(MAKE) benchmark DATASET=$$ds; \
	done
	@$(MAKE) benchmark-compare

benchmark-compare:  ## Compare latest results to the PNEUMA paper (per dataset).
	$(PYTHON) -m scripts.benchmark.compare_to_pneuma_paper

# Wipe a single dataset's prepare/index/results artifacts.
benchmark-clean:  ## Drop benchmark-data/{lakes,indexes,results}/$(DATASET)/.
	rm -rf benchmark-data/lakes/$(DATASET) benchmark-data/indexes/$(DATASET) benchmark-data/results/$(DATASET)

benchmark-clean-all:  ## Drop benchmark-data/{lakes,indexes,results}/ for every dataset.
	@for ds in $(BENCHMARK_DATASETS); do \
	  $(MAKE) benchmark-clean DATASET=$$ds; \
	done

benchmark-test:  ## Run benchmark unit tests (no vLLM needed).
	$(PYTHON) -m pytest tests/test_benchmark/ -m "not requires_vllm"

benchmark-test-vllm:  ## Run the live smoke test (requires vLLM containers on :8001/:8002).
	$(PYTHON) -m pytest tests/test_benchmark/ -m requires_vllm


build:  ## Build the Blend Docker image.
	$(COMPOSE) build

up: build  ## Build + start the Blend container in the background (host networking → reaches vLLMs on 127.0.0.1:8001-8005).
	$(COMPOSE) up -d
	@echo ""
	@echo "  blend container is up."
	@echo "    make shell  → bash inside the container"
	@echo "    make bench  → run benchmark-all inside the container"
	@echo "    make down   → stop the container"

down:  ## Stop the Blend container.
	$(COMPOSE) down

shell:  ## Open bash inside the running blend container.
	$(COMPOSE) exec blend bash

bench:  ## Run `make benchmark-all` inside the container (results land in benchmark-data/results/ on the host).
	$(COMPOSE) exec blend make benchmark-all

logs:  ## Tail the blend container's logs.
	$(COMPOSE) logs -f blend
