PYTHON ?= .venv/bin/python

.DEFAULT_GOAL := help
.PHONY: help install index test clean

help:  
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: 
	uv sync

index:  ## Build Blend value-index + NLSeeker NL-index over data/lake/*.csv.
	$(PYTHON) scripts/create_blend_index.py

test: 
	$(PYTHON) -m pytest -v

clean:  ## Drop the DuckDB value-index and the NL-index artifacts.
	rm -f blend.duckdb
	rm -rf nl-out
