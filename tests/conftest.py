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
