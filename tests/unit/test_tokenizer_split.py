import pytest

from landuse_filter.adapters.tokenizer import SENTINEL, split_template


def test_split_template_around_the_prompt():
    assert split_template(f"<bos>user\n{SENTINEL}\nassistant\n<think>") == (
        "<bos>user\n",
        "\nassistant\n<think>",
    )


@pytest.mark.parametrize("rendered", ["no prompt", f"{SENTINEL}{SENTINEL}"])
def test_split_template_requires_exactly_one_prompt(rendered):
    with pytest.raises(ValueError, match="exactly once"):
        split_template(rendered)
