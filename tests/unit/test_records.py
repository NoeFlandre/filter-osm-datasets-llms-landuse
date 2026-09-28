from landuse_filter.domain.records import Generation, from_sglang


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
