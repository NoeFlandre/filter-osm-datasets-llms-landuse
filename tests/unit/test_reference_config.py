from landuse_filter import config

# Changing any value below changes the generation fingerprint, which keys every result part
# on the bucket (parts/<fp>/). Update deliberately: it orphans all computed results.
GOLDEN_REFERENCE_CONFIG = {
    "chat_template_kwargs": {"enable_thinking": False},
    "draft": "LiquidAI/LFM2.5-2.6B-DSpark@458cedab07d0f7b2b05700c77e1aa463d43d6f04",
    "engine": {
        "disable_radix_cache": True,
        "dtype": "bfloat16",
        "mem_fraction_static": 0.75,
        "random_seed": 0,
        "speculative_algorithm": "DSPARK",
        "speculative_draft_attention_backend": "flashinfer",
    },
    "model": "LiquidAI/LFM2.5-2.6B@654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
    "prompt_sha256": "2fb48569c8bbb73fcf584fb7549b34ddd5e4cc365b7a5c75be7429906e312097",
    "sampling": {"max_new_tokens": 4096, "temperature": 0.0},
}
GOLDEN_GENERATION_FP = "71dd8471f52321ab"


def test_reference_config_is_pinned_literally():
    assert config.reference_config() == GOLDEN_REFERENCE_CONFIG


def test_generation_fingerprint_is_pinned_literally():
    assert config.GENERATION_FP == GOLDEN_GENERATION_FP


def test_engine_kwargs_merges_engine_and_speed():
    kwargs = config.engine_kwargs(config.reference_config(), {"max_running_requests": 9})
    assert kwargs["model_path"] == config.MODEL_ID
    assert kwargs["max_running_requests"] == 9
    assert kwargs["speculative_algorithm"] == "DSPARK"
