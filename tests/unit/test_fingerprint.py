from landuse_filter.domain.fingerprint import config_fingerprint, serving_fingerprint

BASE = {
    "model": "m@1",
    "max_new_tokens": 4096,
    "engine": {"dtype": "bfloat16", "mem_fraction_static": 0.75},
}


def test_speed_only_args_do_not_change_generation_identity():
    tuned = {
        **BASE,
        "engine": {"dtype": "bfloat16", "mem_fraction_static": 0.9, "max_running_requests": 128},
    }
    assert config_fingerprint(tuned) == config_fingerprint(BASE)
    assert serving_fingerprint(tuned) != serving_fingerprint(BASE)


def test_output_affecting_fields_change_identity():
    assert config_fingerprint({**BASE, "max_new_tokens": 2048}) != config_fingerprint(BASE)
    assert config_fingerprint({**BASE, "engine": {"dtype": "float16"}}) != config_fingerprint(BASE)


def test_key_order_is_irrelevant():
    assert config_fingerprint(dict(reversed(list(BASE.items())))) == config_fingerprint(BASE)
