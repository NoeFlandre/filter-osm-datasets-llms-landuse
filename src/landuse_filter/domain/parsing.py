"""Verdicts from thinking-mode generations (ADR-0008 of the benchmark, hardened).

The LFM2.5 chat template ends the generation prompt with ``<think>``, so a finished
generation reads ``<reasoning></think><answer>``. Only the answer region after the last
``</think>`` is parsed; the reasoning restates the rubric and mentions both labels.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

from landuse_filter.domain.sentences import Decision

PARSER_VERSION = "2"
THINK_CLOSE = "</think>"

_LABEL = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_EXACT = re.compile(r"^\W*(yes|no)\W*$", re.IGNORECASE)
_LEADING = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
_INPUT_TOO_LONG = re.compile(r"^luf_failure:input_too_long:(\d+)(?:\+(\d+))?(>=|>)(\d+)$")
# Answers in the sentence's language instead of English; counted, never mapped.
_FOREIGN = re.compile(
    r"^\W*(oui|non|ja|nein|nee|sí|si|não|nao|sim|да|нет|是|否|はい|いいえ|evet|hayır|tak|nie)\W*$",  # noqa: RUF001 - Turkish word
    re.IGNORECASE,
)


class ParseMode(StrEnum):
    EXACT = "exact"
    LEADING = "leading"
    LAST = "last"


class Failure(StrEnum):
    INPUT_TOO_LONG = "input_too_long"
    TRUNCATED = "truncated"
    UNCLOSED_THINK = "unclosed_think"
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"
    NON_ENGLISH_TOKEN = "non_english_token"  # noqa: S105 - a failure reason, not a secret
    NO_LABEL = "no_label"


@dataclass(frozen=True, slots=True)
class Verdict:
    decision: Decision
    mode: ParseMode | None = None
    failure: Failure | None = None


def _failed(reason: Failure) -> Verdict:
    return Verdict(Decision.FAILED, failure=reason)


def _is_input_too_long(raw: str) -> bool:
    match = _INPUT_TOO_LONG.fullmatch(raw)
    if match is None:
        return False
    input_tokens = int(match.group(1))
    max_new_tokens = match.group(2)
    requested_tokens = input_tokens + int(max_new_tokens or 0)
    operator = match.group(3)
    context_limit = int(match.group(4))
    if operator == ">":
        return requested_tokens > context_limit
    return max_new_tokens is None and requested_tokens >= context_limit


def parse_generation(raw: str, *, truncated: bool) -> Verdict:
    """Recover ``yes``/``no`` from a thinking-mode generation, or say why not."""
    if _is_input_too_long(raw):
        return _failed(Failure.INPUT_TOO_LONG)
    if truncated:
        return _failed(Failure.TRUNCATED)
    if THINK_CLOSE not in raw:
        return _failed(Failure.UNCLOSED_THINK)
    return parse_answer(raw.rsplit(THINK_CLOSE, 1)[1].strip())


def parse_answer(answer: str) -> Verdict:
    """Parse the answer region: exact, then leading, then a single distinct label."""
    if not answer:
        return _failed(Failure.EMPTY)
    return _anchored(answer) or _by_labels(answer)


def _anchored(answer: str) -> Verdict | None:
    for pattern, mode in ((_EXACT, ParseMode.EXACT), (_LEADING, ParseMode.LEADING)):
        match = pattern.search(answer)
        if match:
            return Verdict(Decision(match.group(1).lower()), mode)
    return None


def _by_labels(answer: str) -> Verdict:
    labels = {m.lower() for m in _LABEL.findall(answer)}
    if len(labels) == 1:
        return Verdict(Decision(labels.pop()), ParseMode.LAST)
    if labels:
        return _failed(Failure.AMBIGUOUS)
    foreign = _FOREIGN.search(answer)
    return _failed(Failure.NON_ENGLISH_TOKEN if foreign else Failure.NO_LABEL)
