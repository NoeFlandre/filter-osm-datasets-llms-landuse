"""What one generation records (one row of a ``parts/`` or ``generations/`` table)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Generation:
    text_sha256: str
    raw_output: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str
    truncated: bool
    verify_steps: int | None
    accepted_drafts: int | None
    proposed_drafts: int | None
    latency_s: float | None


def from_sglang(text_sha256: str, output: dict, prompt_tokens: int) -> Generation:
    """Read one SGLang completion dict (``text`` + ``meta_info``)."""
    meta = output.get("meta_info", {})
    finish = meta.get("finish_reason") or {}
    kind = str(finish.get("type", "unknown")) if isinstance(finish, dict) else str(finish)
    steps = meta.get("spec_verify_ct")
    latency = meta.get("e2e_latency")
    return Generation(
        text_sha256=text_sha256,
        raw_output=str(output.get("text", "")),
        prompt_tokens=prompt_tokens,
        generated_tokens=int(meta.get("completion_tokens") or 0),
        finish_reason=kind,
        truncated=kind == "length",
        verify_steps=int(steps) if steps else None,
        accepted_drafts=meta.get("spec_num_correct_drafts"),
        proposed_drafts=meta.get("spec_num_proposed_drafts"),
        latency_s=None if latency is None else float(latency),
    )


# finish_reason of a row decided by rule instead of by the model (ADR-0024).
RULE_NO_LETTERS = "rule:no_letters"


def rule_decided_no(text_sha256: str, prompt_tokens: int) -> Generation:
    """The row of a letterless sentence: a plain ``no`` that parses like a model answer."""
    return Generation(
        text_sha256=text_sha256,
        raw_output="</think>no",
        prompt_tokens=prompt_tokens,
        generated_tokens=0,
        finish_reason=RULE_NO_LETTERS,
        truncated=False,
        verify_steps=None,
        accepted_drafts=None,
        proposed_drafts=None,
        latency_s=None,
    )
