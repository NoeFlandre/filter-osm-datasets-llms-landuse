import asyncio

import pyarrow as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.adapters.schema import CHUNK, PROVENANCE
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.node import Runner, RunStats
from landuse_filter.domain.parsing import parse_generation
from landuse_filter.domain.records import RULE_NO_LETTERS, rule_decided_no
from landuse_filter.domain.sentences import Decision, has_no_letters

PROV = {name: "x" for name, _ in PROVENANCE if name != "created_at"}


@pytest.mark.parametrize(
    "text", ["", " ", "\n\t", "12345", "--", "+33 1 23", "...", "😀", "①②", "1,5 % — 2", "€ 5"]
)
def test_letterless_texts_are_ruled(text):
    assert has_no_letters(text)


@pytest.mark.parametrize(
    "text", ["a", "A1", "1a", "é", "日本語", "Привет", "مرحبا", "12 km", "-x-", "ß", "ª"]
)
def test_texts_with_a_letter_go_to_the_model(text):
    assert not has_no_letters(text)


@given(st.text())
def test_rule_is_exactly_no_alpha_char(text):
    assert has_no_letters(text) == (not any(c.isalpha() for c in text))


def test_rule_row_parses_as_a_plain_no():
    g = rule_decided_no("sha", 7)
    verdict = parse_generation(g.raw_output, truncated=g.truncated)
    assert verdict.decision is Decision.NO
    assert (g.finish_reason, g.truncated, g.generated_tokens) == (RULE_NO_LETTERS, False, 0)
    assert (g.text_sha256, g.prompt_tokens) == ("sha", 7)
    assert g.latency_s is None


class CountingEngine:
    def __init__(self):
        self.calls = 0

    async def generate(self, input_ids):
        self.calls += 1
        return {
            "text": "t</think>yes",
            "meta_info": {"completion_tokens": 5, "finish_reason": {"type": "stop"}},
        }


def test_runner_skips_the_model_for_letterless_texts(tmp_path):
    store = WorkStore(tmp_path)
    texts = ["12345", "hello", "--", "world"]
    shas = [f"s{i}" for i in range(4)]
    store.write_chunk(
        "c1",
        pa.table({"text_sha256": shas, "text": texts, "input_ids": [[1, 2]] * 4}, schema=CHUNK),
    )
    engine = CountingEngine()
    r = Runner(store, engine, "fp", PROV, window=2, flush_every=2, flush_seconds=999)
    stats = asyncio.run(r.run(["c1"]))
    assert engine.calls == 2
    assert stats.completed == 4
    assert stats.rule_decided == 2
    assert stats.failed == 0
    assert stats.chunks_done == ["c1"]
    rows = {}
    for path in store.part_paths("fp", "c1"):
        for row in store.read_part(path).to_pylist():
            rows[row["text_sha256"]] = row
    assert rows["s0"]["finish_reason"] == RULE_NO_LETTERS
    assert rows["s1"]["finish_reason"] == "stop"
    # resume: everything is done, nothing is asked again
    again = asyncio.run(Runner(store, engine, "fp", PROV).run(["c1"]))
    assert engine.calls == 2
    assert again.completed == 0


def test_throughput_counts_only_model_processed_sentences():
    s = RunStats(completed=10, rule_decided=6)
    s.started -= 2.0
    assert 1.9 < s.sentences_per_second < 2.0
