import pytest

from landuse_filter.adapters.store import WorkStore


def test_append_repairs_torn_tail_then_appends_twice_and_reopens(tmp_path):
    store = WorkStore(tmp_path)
    ledger = store.path("ledger.jsonl")
    ledger.write_bytes(b'{"path": "first"}\n{"path":')

    assert store.read_jsonl("ledger.jsonl") == [{"path": "first"}]
    store.append_jsonl("ledger.jsonl", [{"path": "second"}])
    store.append_jsonl("ledger.jsonl", [{"path": "third"}])

    reopened = WorkStore(tmp_path)
    assert reopened.read_jsonl("ledger.jsonl") == [
        {"path": "first"},
        {"path": "second"},
        {"path": "third"},
    ]


@pytest.mark.parametrize("initial", [b"", b'{"path": "first"}\n'])
def test_append_to_empty_or_newline_terminated_file(tmp_path, initial):
    store = WorkStore(tmp_path)
    store.path("ledger.jsonl").write_bytes(initial)

    store.append_jsonl("ledger.jsonl", [{"path": "second"}])

    expected = ([{"path": "first"}] if initial else []) + [{"path": "second"}]
    assert WorkStore(tmp_path).read_jsonl("ledger.jsonl") == expected


def test_append_separates_a_valid_final_record_without_a_newline(tmp_path):
    store = WorkStore(tmp_path)
    store.path("ledger.jsonl").write_bytes(b'{"path": "first"}')

    store.append_jsonl("ledger.jsonl", [{"path": "second"}])

    assert WorkStore(tmp_path).read_jsonl("ledger.jsonl") == [
        {"path": "first"},
        {"path": "second"},
    ]


def test_read_keeps_valid_records_after_bad_or_truncated_lines(tmp_path):
    store = WorkStore(tmp_path)
    store.path("ledger.jsonl").write_bytes(
        b'{"path": "first"}\n{broken}\n{"path": "second"}\n{"path": "\xc3'
    )

    assert store.read_jsonl("ledger.jsonl") == [
        {"path": "first"},
        {"path": "second"},
    ]


def test_append_removes_only_a_truncated_utf8_tail(tmp_path):
    store = WorkStore(tmp_path)
    store.path("ledger.jsonl").write_bytes(b'{"path": "first"}\n{"path": "\xc3')

    store.append_jsonl("ledger.jsonl", [{"path": "second"}])

    assert WorkStore(tmp_path).read_jsonl("ledger.jsonl") == [
        {"path": "first"},
        {"path": "second"},
    ]


def test_append_finds_a_torn_tail_longer_than_one_read_window(tmp_path):
    store = WorkStore(tmp_path)
    tail = b'{"payload": "' + b"x" * 9000
    store.path("ledger.jsonl").write_bytes(b'{"path": "first"}\n' + tail)

    store.append_jsonl("ledger.jsonl", [{"path": "second"}])

    assert WorkStore(tmp_path).read_jsonl("ledger.jsonl") == [
        {"path": "first"},
        {"path": "second"},
    ]
