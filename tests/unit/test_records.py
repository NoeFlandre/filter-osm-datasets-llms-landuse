import pytest

from landuse_filter.domain.records import (
    MAX_RECORDED_TOKEN_COUNT,
    Generation,
    from_sglang,
    input_too_long_failure,
)


def test_from_sglang_reads_every_field():
    output = {
        "text": "yes",
        "meta_info": {
            "finish_reason": {"type": "length"},
            "completion_tokens": 7,
            "spec_verify_ct": 3,
            "spec_num_correct_drafts": 5,
            "spec_num_proposed_drafts": 9,
            "e2e_latency": 2,
        },
    }
    assert from_sglang("sha", output, 11) == Generation(
        text_sha256="sha",
        raw_output="yes",
        prompt_tokens=11,
        generated_tokens=7,
        finish_reason="length",
        truncated=True,
        verify_steps=3,
        accepted_drafts=5,
        proposed_drafts=9,
        latency_s=2.0,
    )
    assert isinstance(from_sglang("sha", output, 11).latency_s, float)


def test_from_sglang_defaults_for_missing_fields():
    assert from_sglang("sha", {}, 4) == Generation(
        "sha", "", 4, 0, "unknown", False, None, None, None, None
    )


def test_from_sglang_finish_reason_variants():
    def kind(finish):
        return from_sglang("s", {"meta_info": {"finish_reason": finish}}, 0).finish_reason

    assert kind("stop") == "stop"
    assert kind({"type": "stop"}) == "stop"
    assert kind({}) == "unknown"
    assert kind(None) == "unknown"
    assert from_sglang("s", {"meta_info": {"finish_reason": "stop"}}, 0).truncated is False


def test_from_sglang_zero_latency_is_kept_and_zero_steps_dropped():
    g = from_sglang("s", {"meta_info": {"e2e_latency": 0, "spec_verify_ct": 0}}, 0)
    assert g.latency_s == 0.0
    assert g.verify_steps is None


@pytest.mark.parametrize(
    ("input_tokens", "context_limit", "expected"),
    [
        (330_533, 131_072, "luf_failure:input_too_long:330533>131072"),
        (131_072, 131_072, "luf_failure:input_too_long:131072>=131072"),
        (
            MAX_RECORDED_TOKEN_COUNT,
            MAX_RECORDED_TOKEN_COUNT,
            f"luf_failure:input_too_long:{MAX_RECORDED_TOKEN_COUNT}>={MAX_RECORDED_TOKEN_COUNT}",
        ),
    ],
)
def test_input_too_long_failure_records_prompt_overflow_without_truncating(
    input_tokens, context_limit, expected
):
    assert input_too_long_failure("sha", input_tokens, context_limit) == Generation(
        "sha", expected, input_tokens, 0, "input_too_long", False, None, None, None, None
    )


@pytest.mark.parametrize("context_limit", [0, -1, MAX_RECORDED_TOKEN_COUNT + 1])
def test_input_too_long_failure_rejects_unrecordable_context_limits(context_limit):
    with pytest.raises(ValueError, match="context limit must be positive"):
        input_too_long_failure("sha", 1, context_limit)


@pytest.mark.parametrize("input_tokens", [0, -1, MAX_RECORDED_TOKEN_COUNT + 1])
def test_input_too_long_failure_rejects_unrecordable_prompt_counts(input_tokens):
    with pytest.raises(ValueError, match="input token count must be positive and recordable"):
        input_too_long_failure("sha", input_tokens, 2)


def test_input_too_long_failure_requires_prompt_overflow_without_completion_budget():
    with pytest.raises(ValueError, match="input token count must meet or exceed the context limit"):
        input_too_long_failure("sha", 4, 5)


@pytest.mark.parametrize(
    ("input_tokens", "context_limit", "max_new_tokens"),
    [
        (5, 5, 1),
        (4, 5, 1),
        (MAX_RECORDED_TOKEN_COUNT - 2, MAX_RECORDED_TOKEN_COUNT - 1, 3),
    ],
)
def test_input_too_long_failure_rejects_invalid_completion_budget_overflow(
    input_tokens, context_limit, max_new_tokens
):
    with pytest.raises(
        ValueError, match="requested token count must exceed a positive context limit"
    ):
        input_too_long_failure("sha", input_tokens, context_limit, max_new_tokens)


@pytest.mark.parametrize(
    ("input_tokens", "context_limit", "max_new_tokens", "expected"),
    [
        (127_000, 131_072, 4_096, "luf_failure:input_too_long:127000+4096>131072"),
        (
            MAX_RECORDED_TOKEN_COUNT - 2,
            MAX_RECORDED_TOKEN_COUNT - 1,
            2,
            f"luf_failure:input_too_long:{MAX_RECORDED_TOKEN_COUNT - 2}+2>"
            f"{MAX_RECORDED_TOKEN_COUNT - 1}",
        ),
    ],
)
def test_input_too_long_failure_records_completion_budget_overflow(
    input_tokens, context_limit, max_new_tokens, expected
):
    assert input_too_long_failure("sha", input_tokens, context_limit, max_new_tokens) == Generation(
        "sha", expected, input_tokens, 0, "input_too_long", False, None, None, None, None
    )
