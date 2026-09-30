from landuse_filter import config
from landuse_filter.domain.fingerprint import config_fingerprint


def test_reference_config_shape_and_fingerprint():
    cfg = config.reference_config()
    assert set(cfg) == set(config.ReferenceConfig.__annotations__)
    assert set(cfg["sampling"]) == set(config.Sampling.__annotations__)
    assert set(cfg["engine"]) == set(config.EngineArgs.__annotations__)
    assert cfg["sampling"]["max_new_tokens"] == config.MAX_NEW_TOKENS
    assert config_fingerprint(cfg) == config.GENERATION_FP


def test_engine_kwargs_merges_engine_and_speed():
    kwargs = config.engine_kwargs(config.reference_config(), {"max_running_requests": 9})
    assert kwargs["model_path"] == config.MODEL_ID
    assert kwargs["max_running_requests"] == 9
    assert kwargs["speculative_algorithm"] == "DSPARK"
