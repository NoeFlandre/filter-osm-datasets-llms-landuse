import gzip
import json
import os
from pathlib import Path

import pytest
from hypothesis import settings

settings.register_profile("ci", max_examples=200, deadline=None)
settings.register_profile("dev", max_examples=50, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def reference_sample() -> list[dict]:
    with gzip.open(FIXTURES / "reference_dspark_sample.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(autouse=True)
def _no_real_hub(request, monkeypatch):
    """Unit tests never reach the Hugging Face Hub (regression: a test listed the real
    bucket through the developer's login). Integration tests are exempt."""
    if request.node.get_closest_marker("integration"):
        return

    class NoHub:
        def __init__(self, *a, **k):
            raise RuntimeError("unit tests must not call the Hugging Face Hub")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", NoHub)
