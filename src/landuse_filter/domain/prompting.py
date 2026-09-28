"""The benchmark prompt, verbatim (ADR-0004)."""

PLACEHOLDER = "{}"
PROMPT_SHA256 = "2fb48569c8bbb73fcf584fb7549b34ddd5e4cc365b7a5c75be7429906e312097"


class PromptError(ValueError):
    """The prompt template is not the benchmark's."""


def render_prompt(template: str, sentence: str) -> str:
    """Insert ``sentence`` at the first ``{}``; braces in the sentence stay untouched."""
    if PLACEHOLDER not in template:
        raise PromptError(f"prompt template has no {PLACEHOLDER!r} placeholder")
    return template.replace(PLACEHOLDER, sentence, 1)


def check_prompt_digest(digest: str) -> None:
    """Refuse any template other than the one the benchmark was scored with."""
    if digest != PROMPT_SHA256:
        raise PromptError(f"prompt sha256 {digest} != benchmark {PROMPT_SHA256}")
