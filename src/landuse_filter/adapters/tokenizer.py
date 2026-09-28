"""LFM2.5 chat-template encoding, identical to the benchmark's SGLang runs.

The benchmark passes ``enable_thinking=False``; the LFM2.5-2.6B template ignores it
and always opens ``<think>``. We pass the same kwargs so token ids match byte for byte
(verified against the benchmark encoder in the integration suite).
"""

from collections.abc import Callable
from typing import Any

CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}


def chat_encoder(model_id: str, revision: str) -> Callable[[str], list[int]]:
    from transformers import AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def encode(prompt: str) -> list[int]:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    return encode


def template_digest(model_id: str, revision: str) -> str:
    from transformers import AutoTokenizer

    from landuse_filter.domain.hashing import sha256_text

    tokenizer: Any = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return sha256_text(str(tokenizer.chat_template))
