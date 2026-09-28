"""LFM2.5 chat-template encoding, identical to the benchmark's SGLang runs.

The benchmark passes ``enable_thinking=False``; the LFM2.5-2.6B template ignores it
and always opens ``<think>``. We pass the same kwargs so token ids match byte for byte.

For a single user turn the rendered template is ``prefix + prompt + suffix`` with a
constant prefix and suffix, so :func:`chat_encoder` renders the template once and then
tokenizes prompts in large batches with the fast (Rust, multi-threaded) tokenizer. The
integration test checks the ids equal :func:`reference_encoder`, the per-prompt path.
"""

from collections.abc import Callable
from typing import Any

CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}
SENTINEL = "\x00LUF_PROMPT\x00"
BATCH = 4096


def _tokenizer(model_id: str, revision: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def _render(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )


def reference_encoder(model_id: str, revision: str) -> Callable[[str], list[int]]:
    """One prompt at a time through the full template (slow; the ground truth)."""
    tokenizer = _tokenizer(model_id, revision)

    def encode(prompt: str) -> list[int]:
        return list(tokenizer(_render(tokenizer, prompt), add_special_tokens=False)["input_ids"])

    return encode


def split_template(rendered: str) -> tuple[str, str]:
    """Constant text before and after the prompt in a rendered single-turn template."""
    if rendered.count(SENTINEL) != 1:
        raise ValueError("chat template must contain the user prompt exactly once")
    prefix, suffix = rendered.split(SENTINEL)
    return prefix, suffix


def chat_encoder(model_id: str, revision: str) -> Callable[[list[str]], list[list[int]]]:
    """Batch encoder: prompts -> token ids, byte-identical to :func:`reference_encoder`."""
    tokenizer = _tokenizer(model_id, revision)
    prefix, suffix = split_template(_render(tokenizer, SENTINEL))

    def encode(prompts: list[str]) -> list[list[int]]:
        out: list[list[int]] = []
        for start in range(0, len(prompts), BATCH):
            texts = [prefix + p + suffix for p in prompts[start : start + BATCH]]
            out.extend(list(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"])
        return out

    return encode


def template_digest(model_id: str, revision: str) -> str:
    from landuse_filter.domain.hashing import sha256_text

    return sha256_text(str(_tokenizer(model_id, revision).chat_template))
