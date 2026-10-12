import asyncio
import re

import pytest

from landuse_filter.adapters.engine import (
    ContextLengthExceededError,
    SGLangEngine,
    context_length_overflow,
)


def context_error(input_tokens, context_limit):
    return (
        f"The input ({input_tokens} tokens) is longer than the model's context length "
        f"({context_limit} tokens)."
    )


@pytest.mark.parametrize(
    ("input_tokens", "context_limit", "expected"),
    [
        (330_533, 131_072, (330_533, 131_072)),
        (131_073, 131_072, (131_073, 131_072)),
        (131_072, 131_072, None),
        (131_071, 131_072, None),
    ],
)
def test_context_error_classifier_requires_a_strict_overflow(input_tokens, context_limit, expected):
    assert context_length_overflow(context_error(input_tokens, context_limit)) == expected


def test_context_error_classifier_ignores_unrelated_or_unstorable_errors():
    assert context_length_overflow("CUDA out of memory") is None
    assert context_length_overflow(context_error(2_147_483_648, 131_072)) is None


class FailingBackend:
    def __init__(self, message):
        self.message = message

    async def async_generate(self, **_kwargs):
        raise ValueError(self.message)


def engine_with_backend(backend):
    engine = object.__new__(SGLangEngine)
    engine._engine = backend
    engine._sampling = {}
    return engine


def test_engine_translates_the_synthetic_330533_token_error():
    engine = engine_with_backend(FailingBackend(context_error(330_533, 131_072)))

    with pytest.raises(ContextLengthExceededError) as raised:
        asyncio.run(engine.generate([1]))

    assert (raised.value.input_tokens, raised.value.context_limit) == (330_533, 131_072)


def test_engine_preserves_other_value_errors_and_exact_limit_boundary():
    for message in ("CUDA out of memory", context_error(131_072, 131_072)):
        engine = engine_with_backend(FailingBackend(message))
        with pytest.raises(ValueError, match=re.escape(message)):
            asyncio.run(engine.generate([1]))
