COMPOSE ?= docker compose
CONFIG  ?= evaluation/nlseeker/configs/default.yaml

.DEFAULT_GOAL := help
.PHONY: help build up down shell logs evaluation-nlseeker eval-clean eval-clean-cache

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

build:  ## Build the Blend image from the working tree, always from scratch.
	$(COMPOSE) build --no-cache

up: build  ## Start the blend container (host networking reaches the vLLMs).
	$(COMPOSE) up -d

down:  ## Stop the blend container.
	$(COMPOSE) down

shell:  ## Bash inside the running container.
	$(COMPOSE) exec blend bash

logs:  ## Tail container logs.
	$(COMPOSE) logs -f blend

evaluation-nlseeker:  ## Run the NLSeeker evaluation. CONFIG=<yaml> for fresh, RESUME=<run-dir name under results/> to resume.
ifdef RESUME
	@case '$(RESUME)' in *..*|*[!A-Za-z0-9_.:-]*|"") echo "invalid RESUME: $(RESUME)"; exit 1;; esac
	$(COMPOSE) exec blend python -m evaluation.nlseeker --resume 'data/Evaluation/NLSeeker/results/$(RESUME)'
else
	$(COMPOSE) exec blend python -m evaluation.nlseeker --config '$(CONFIG)'
endif

# The container runs as root, so bind-mounted results are root-owned on the
# host and a host-side rm would fail with EPERM. Route deletions through the
# container: exec if it is up, one-shot run --rm otherwise.
define IN_CONTAINER
	@if $(COMPOSE) ps --services --filter status=running 2>/dev/null | grep -qx blend; then \
	  $(COMPOSE) exec -T blend $(1); \
	else \
	  $(COMPOSE) run --rm -T blend $(1); \
	fi
endef

eval-clean:  ## Delete one run dir: make eval-clean RUN=<run-dir name under results/>.
ifndef RUN
	$(error RUN is required, e.g. make eval-clean RUN=2026-08-05T14-30-00Z__default)
endif
	@case '$(RUN)' in *..*|*[!A-Za-z0-9_.:-]*|"") echo "invalid RUN: $(RUN)"; exit 1;; esac
	$(call IN_CONTAINER,rm -rf 'data/Evaluation/NLSeeker/results/$(RUN)')

eval-clean-cache:  ## Delete the extracted lake/bench cache (datasets/ is never touched).
	$(call IN_CONTAINER,rm -rf data/Evaluation/NLSeeker/cache)
