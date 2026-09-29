UV ?= uv
SHELL := bash
.SHELLFLAGS := -eo pipefail -c
RUN := $(UV) run --no-sync
HYPOTHESIS_PROFILE ?= ci

.PHONY: help install baseline lint format types test property acceptance architecture crap mutation smoke docs-build gauntlet

help:  ## List targets
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:  ## Dev environment (CPU; tokenizer extra for planning) + git hooks
	$(UV) sync --extra tokenize --extra map
	git config core.hooksPath scripts/hooks

baseline:  ## Existing suite before changing anything
	$(RUN) pytest

lint:  ## ruff format check + lint
	$(RUN) ruff format --check .
	$(RUN) ruff check .

format:  ## Apply ruff formatting and fixes
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

types:  ## ty static types
	$(RUN) ty check src

test:  ## Unit + property tests with coverage
	HYPOTHESIS_PROFILE=$(HYPOTHESIS_PROFILE) $(RUN) pytest tests/unit tests/property --cov --cov-report=term-missing --cov-report=json

property:  ## Hypothesis property tests
	HYPOTHESIS_PROFILE=$(HYPOTHESIS_PROFILE) $(RUN) pytest tests/property

acceptance:  ## Executable Gherkin scenarios
	$(RUN) pytest tests/acceptance

architecture:  ## Dependency boundaries (AST test + import-linter)
	$(RUN) pytest tests/architecture
	$(RUN) lint-imports

crap: test  ## CRAP < 6 on the domain
	$(RUN) python scripts/crap.py --limit src/landuse_filter/domain=6 --allowlist scripts/crap-allowlist.json

mutation:  ## Mutation testing gated on reviewed survivors
	$(RUN) mutmut run --max-children 4 || true
	$(RUN) python scripts/check_mutants.py

smoke:  ## CLI smoke test
	$(RUN) luf version
	$(RUN) luf fingerprint --json

docs-build:  ## Strict docs build
	$(UV) run --only-group docs mkdocs build --strict

gauntlet: baseline lint types test acceptance architecture crap mutation smoke docs-build  ## Full QA gauntlet
