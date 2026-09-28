"""The land-use relevance benchmark and its published reference run, from the Hub."""

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

BENCHMARK_REPO = "NoeFlandre/benchmark-llms-landuse-relevance"
REFERENCE_RUN = "LiquidAI__LFM2.5-2.6B+DSpark-throughput-b16.json"
BENCHMARK_SHA256 = "81587e4aec2f8ce21bc95195c6bb8cba945a5ed30c2dfe4c678c3a0d0d3010ff"


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    item_id: str
    language: str
    sentence: str
    label: str


@dataclass(frozen=True, slots=True)
class ReferencePrediction:
    item_id: str
    language: str
    expected: str
    predicted: str | None
    generated_tokens: int
    truncated: bool
    raw_output: str


def download(local_dir: Path, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            BENCHMARK_REPO,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"*/{REFERENCE_RUN}", "data/*"],
            local_dir=local_dir,
        )
    )


def read_items(root: Path) -> list[BenchmarkItem]:
    with (root / "data" / "train.csv").open(encoding="utf-8", newline="") as f:
        return [
            BenchmarkItem(r["item_id"], r["language"], r["sentence"], r["label"])
            for r in csv.DictReader(f)
        ]


def read_reference(root: Path) -> Iterator[ReferencePrediction]:
    for path in sorted(root.glob(f"*/{REFERENCE_RUN}")):
        language = path.parent.name
        for p in json.loads(path.read_text(encoding="utf-8"))["predictions"]:
            yield ReferencePrediction(
                p["item_id"], language, p["expected"], p["predicted"],
                int(p["generated_tokens"]), bool(p["truncated"]), p["raw_output"],
            )
