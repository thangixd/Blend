PYTHON ?= .venv/bin/python

.DEFAULT_GOAL := help
.PHONY: help install index test test-index test-clean clean

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
