"""Arrow schemas of every table the pipeline writes (schema_version 1)."""

from collections.abc import Sequence
from dataclasses import asdict

import pyarrow as pa

from landuse_filter.domain.records import Generation

SCHEMA_VERSION = "1"

CHUNK = pa.schema(
    [
        ("text_sha256", pa.string()),
        ("text", pa.large_string()),
        ("input_ids", pa.list_(pa.int32())),
    ]
)

PROVENANCE = [
    ("config_fingerprint", pa.string()),
    ("serving_fingerprint", pa.string()),
    ("model", pa.string()),
    ("draft", pa.string()),
    ("prompt_sha256", pa.string()),
    ("sglang_version", pa.string()),
    ("gpu", pa.string()),
    ("site", pa.string()),
    ("oar_job_id", pa.string()),
    ("assignment_id", pa.string()),
    ("code_commit", pa.string()),
    ("created_at", pa.string()),
]

GENERATION = pa.schema(
    [
        ("text_sha256", pa.string()),
        ("raw_output", pa.large_string()),
        ("prompt_tokens", pa.int32()),
        ("generated_tokens", pa.int32()),
        ("finish_reason", pa.string()),
        ("truncated", pa.bool_()),
        ("verify_steps", pa.int32()),
        ("accepted_drafts", pa.int32()),
        ("proposed_drafts", pa.int32()),
        ("latency_s", pa.float64()),
        *PROVENANCE,
    ]
)


def generation_table(rows: Sequence[Generation], provenance: dict[str, str]) -> pa.Table:
    missing = {name for name, _ in PROVENANCE} - provenance.keys()
    if missing:
        raise ValueError(f"provenance lacks {sorted(missing)}")
    records = [{**asdict(r), **provenance} for r in rows]
    return pa.Table.from_pylist(records, schema=GENERATION)
