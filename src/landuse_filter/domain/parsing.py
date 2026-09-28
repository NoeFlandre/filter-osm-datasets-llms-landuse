"""Verdicts from thinking-mode generations (ADR-0008 of the benchmark, hardened).

The LFM2.5 chat template ends the generation prompt with ``<think>``, so a finished
generation reads ``<reasoning></think><answer>``. Only the answer region after the last
``</think>`` is parsed; the reasoning restates the rubric and mentions both labels.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

from landuse_filter.domain.sentences import Decision

PARSER_VERSION = "1"
THINK_CLOSE = "</think>"

_LABEL = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_EXACT = re.compile(r"^\W*(yes|no)\W*$", re.IGNORECASE)
_LEADING = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
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


def parse_generation(raw: str, *, truncated: bool) -> Verdict:
    """Recover ``yes``/``no`` from a thinking-mode generation, or say why not."""
    if truncated:
        return _failed(Failure.TRUNCATED)
    if THINK_CLOSE not in raw:
        return _failed(Failure.UNCLOSED_THINK)
    return parse_answer(raw.rsplit(THINK_CLOSE, 1)[1].strip())


def parse_answer(answer: str) -> Verdict:
    """Parse the answer region: exact, then leading, then a single distinct label."""
    if not answer:
        return _failed(Failure.EMPTY)
    for pattern, mode in ((_EXACT, ParseMode.EXACT), (_LEADING, ParseMode.LEADING)):
        match = pattern.search(answer)
        if match:
            return Verdict(Decision(match.group(1).lower()), mode)
    labels = {m.lower() for m in _LABEL.findall(answer)}
    if len(labels) == 1:
        return Verdict(Decision(labels.pop()), ParseMode.LAST)
    if labels:
        return _failed(Failure.AMBIGUOUS)
    if _FOREIGN.search(answer):
        return _failed(Failure.NON_ENGLISH_TOKEN)
    return _failed(Failure.NO_LABEL)
