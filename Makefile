UV ?= uv
SHELL := bash
.SHELLFLAGS := -eo pipefail -c
RUN := $(UV) run --no-sync
HYPOTHESIS_PROFILE ?= ci
# The gated scope (mutation testing and CRAP) lives once, in [tool.mutmut] source_paths.

.PHONY: help install baseline lint format types test property acceptance architecture crap mutation smoke docs-build gauntlet

help:  ## List targets
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:  ## Dev environment (CPU; tokenizer extra for planning) + git hooks
	$(UV) sync --extra tokenize --extra map
	git config core.hooksPath scripts/hooks

baseline:  ## Existing suite before changing anything
	$(RUN) pytest

lint:  ## ruff format check + lint + shellcheck
	$(RUN) ruff format --check .
	$(RUN) ruff check .
	shellcheck scripts/node_job.sh scripts/ops/*.sh scripts/hooks/pre-commit

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

# The scope helper runs as its own command so a failure (or an empty scope) stops the gate.
crap: test  ## CRAP < 6 on the gated scope
	limits="$$($(RUN) python scripts/quality_scope.py --limit 6)" && $(RUN) python scripts/crap.py $$limits --allowlist scripts/crap-allowlist.json

mutation:  ## Mutation testing gated on reviewed survivors
	$(RUN) mutmut run --max-children 4 || true
	$(RUN) python scripts/check_mutants.py

smoke:  ## CLI smoke test
	$(RUN) luf version
	$(RUN) luf fingerprint --json

docs-build:  ## Strict docs build
	$(UV) run --only-group docs mkdocs build --strict

gauntlet: lint types test acceptance architecture crap mutation smoke docs-build  ## Full QA gauntlet (baseline is run before a change, not inside it)
