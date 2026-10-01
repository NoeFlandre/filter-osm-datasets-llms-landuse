"""The unit of work: one pre-segmented sentence of an input dataset."""

from dataclasses import dataclass
from enum import StrEnum

from landuse_filter.domain.hashing import sha256_parts, sha256_text

Locator = tuple[tuple[str, str | int], ...]


class Decision(StrEnum):
    """Every in-scope sentence gets exactly one of these (ADR-0003)."""

    YES = "yes"
    NO = "no"
    FAILED = "failed"
    SKIPPED_UNSPLIT = "skipped_unsplit"


@dataclass(frozen=True, slots=True)
class SentenceRef:
    """A sentence position in an input file and its text.

    ``locator`` holds the join keys that identify the position in the input table,
    e.g. ``(("polygon_id", "andorra-latest:way/1"), ("field", "website"),
    ("sentence_index", 3))``.
    """

    dataset: str
    source_file: str
    locator: Locator
    text: str
    language: str | None = None
    unsplit: bool = False

    @property
    def text_sha256(self) -> str:
        return sha256_text(self.text)

    @property
    def label_id(self) -> str:
        keys = [f"{name}={value}" for name, value in self.locator]
        return sha256_parts(self.dataset, *keys)


def has_no_letters(text: str) -> bool:
    """True when ``text`` holds no alphabetic character at all (ADR-0024).

    Digits, punctuation, symbols, emoji and whitespace do not count; any Unicode letter
    does, so CJK, Arabic or Cyrillic text still goes to the model.
    """
    return not any(ch.isalpha() for ch in text)
