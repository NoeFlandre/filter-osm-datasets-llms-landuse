import hashlib
import sqlite3

import pyarrow as pa
import pytest

from landuse_filter.adapters.readers import WEBSITE
from landuse_filter.adapters.schema import CHUNK
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.plan import Planner
from landuse_filter.domain.planning import chunk_id


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def legacy_index(store: WorkStore, *, chunk: str, sha: str, text: str) -> None:
    path = store.path(f"index/{WEBSITE}.sqlite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE texts (sha TEXT PRIMARY KEY, text TEXT, file_idx INTEGER, chunk_id TEXT)"
        )
        db.execute(
            "INSERT INTO texts (sha, text, file_idx, chunk_id) VALUES (?, ?, ?, ?)",
            (sha, text, 0, chunk),
        )


@pytest.mark.parametrize(
    ("new_prompt", "new_tokenizer", "new_ids"),
    [
        ("Classify under policy v2: {}", "tokenizer-v1", [201, 8]),
        ("Classify under policy v1: {}", "tokenizer-v2", [202, 8]),
        ("Classify under policy v2: {}", "tokenizer-v2", [202, 9]),
    ],
    ids=["prompt-changed", "tokenizer-changed", "encoded-input-changed"],
)
def test_new_fingerprint_cannot_recover_a_legacy_chunk_with_old_input_ids(
    tmp_path, new_prompt, new_tokenizer, new_ids
):
    text = "A shop"
    sha = text_sha(text)
    old_prompt = "Classify under policy v1: {}"
    old_tokenizer = "tokenizer-v1"
    old_ids = [101, len(old_prompt.format(text))]
    old_fp = f"{old_prompt}|{old_tokenizer}|{old_ids}"
    new_fp = f"{new_prompt}|{new_tokenizer}|{new_ids}"
    old_chunk = chunk_id(old_fp, [sha])
    store = WorkStore(tmp_path)

    legacy_index(store, chunk=old_chunk, sha=sha, text=text)
    store.write_chunk(
        old_chunk,
        pa.table({"text_sha256": [sha], "text": [text], "input_ids": [old_ids]}, schema=CHUNK),
    )

    assert old_fp != new_fp
    assert old_ids != new_ids
    with pytest.raises(ValueError, match="fingerprint"):
        Planner(store, WEBSITE, new_fp)

    assert store.read_chunk(old_chunk).column("input_ids").to_pylist() == [old_ids]
    assert not store.exists(f"plans/{WEBSITE}/{new_fp}/chunks.jsonl")
    with sqlite3.connect(store.path(f"index/{WEBSITE}.sqlite")) as db:
        assert db.execute("SELECT 1 FROM planner_meta WHERE name = 'config_fp'").fetchone() is None


def test_same_fingerprint_migrates_a_legacy_index_and_recovers_lost_plan_lines(tmp_path):
    fp = "prompt-v1|tokenizer-v1"
    text = "A shop"
    sha = text_sha(text)
    legacy_id = chunk_id(fp, [sha])
    store = WorkStore(tmp_path)
    legacy_index(store, chunk=legacy_id, sha=sha, text=text)

    resumed = Planner(store, WEBSITE, fp)
    assert resumed.recover_plan_lines() == 1
    assert store.read_jsonl(resumed.plan_path) == [
        {"chunk_id": legacy_id, "order": [2, 0], "size": 1, "prompt_tokens": 0}
    ]
    resumed.db.close()

    restarted = Planner(WorkStore(tmp_path), WEBSITE, fp)
    assert restarted.recover_plan_lines() == 0


def test_bound_planner_index_rejects_a_different_fingerprint(tmp_path):
    store = WorkStore(tmp_path)
    first = Planner(store, WEBSITE, "prompt-v1|tokenizer-v1")
    first.db.close()

    with pytest.raises(ValueError, match="fingerprint"):
        Planner(store, WEBSITE, "prompt-v2|tokenizer-v1")


def test_legacy_completed_text_without_a_chunk_is_not_bound(tmp_path):
    store = WorkStore(tmp_path)
    path = store.path(f"index/{WEBSITE}.sqlite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE texts (sha TEXT PRIMARY KEY, text TEXT, file_idx INTEGER, "
            "chunk_id TEXT, done INTEGER DEFAULT 0)"
        )
        db.execute(
            "INSERT INTO texts (sha, text, file_idx, done) VALUES (?, ?, ?, 1)",
            (text_sha("A shop"), "A shop", 0),
        )

    with pytest.raises(ValueError, match="completed texts"):
        Planner(store, WEBSITE, "prompt-v1|tokenizer-v1")

    with sqlite3.connect(path) as db:
        assert db.execute("SELECT 1 FROM planner_meta WHERE name = 'config_fp'").fetchone() is None
