import pytest

from landuse_filter import config
from landuse_filter.adapters.tokenizer import chat_encoder, reference_encoder

pytestmark = pytest.mark.integration

PROMPTS = [
    "Classify: vineyards on terraced slopes.",
    "Classify: {} braces {x} stay",
    "Classify: Driehoekige gebied in noordoostelike Afrika.",
    "Classify: 中文句子，带标点。",  # noqa: RUF001 - CJK punctuation on purpose
    "Classify: <|im_end|> literal special-looking text",
    "Classify: trailing spaces   ",
    "",
]


def test_batch_encoder_is_byte_identical_to_the_template_path():
    fast = chat_encoder(config.MODEL_ID, config.MODEL_REVISION)
    slow = reference_encoder(config.MODEL_ID, config.MODEL_REVISION)
    assert fast(PROMPTS) == [slow(p) for p in PROMPTS]
