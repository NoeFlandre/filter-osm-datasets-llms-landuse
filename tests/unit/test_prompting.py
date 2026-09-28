import hashlib
from pathlib import Path

import pytest

from landuse_filter.domain.prompting import (
    PROMPT_SHA256,
    PromptError,
    check_prompt_digest,
    render_prompt,
)

PROMPT = Path(__file__).parents[2] / "data" / "prompt.txt"


def test_vendored_prompt_is_the_benchmark_prompt():
    check_prompt_digest(hashlib.sha256(PROMPT.read_bytes()).hexdigest())


def test_other_prompt_refused():
    with pytest.raises(PromptError):
        check_prompt_digest("0" * 64)


def test_render_keeps_braces_in_sentence():
    assert render_prompt("S: {} end", "a {} b {x}") == "S: a {} b {x} end"


def test_render_requires_placeholder():
    with pytest.raises(PromptError):
        render_prompt("no slot", "x")


def test_digest_constant_matches_reference_run():
    assert PROMPT_SHA256.startswith("2fb48569")


def test_render_replaces_only_first_placeholder():
    assert render_prompt("{} and {}", "x") == "x and {}"


def test_error_messages_name_the_problem():
    with pytest.raises(PromptError, match=r"^prompt template has no '\{\}' placeholder$"):
        render_prompt("no slot", "x")
    with pytest.raises(PromptError, match=f"^prompt sha256 abc != benchmark {PROMPT_SHA256}$"):
        check_prompt_digest("abc")
