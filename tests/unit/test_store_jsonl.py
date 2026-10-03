import pytest

from landuse_filter.adapters.store import CorruptJSONLError, WorkStore


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


def test_read_tolerates_only_a_malformed_unterminated_tail(tmp_path):
    store = WorkStore(tmp_path)
    store.path("ledger.jsonl").write_bytes(b'{"path": "first"}\n{"path": "\xc3')

    assert store.read_jsonl("ledger.jsonl") == [{"path": "first"}]


@pytest.mark.parametrize(
    "corrupt_line", [b"{broken}", b'{"path": "\xff"}'], ids=["invalid-json", "invalid-utf8"]
)
def test_read_fails_closed_on_corrupt_newline_terminated_middle_record(tmp_path, corrupt_line):
    store = WorkStore(tmp_path)
    ledger = store.path("ledger.jsonl")
    first = b'{"path": "first"}\n'
    ledger.write_bytes(first + corrupt_line + b'\n{"path": "last"}\n')

    with pytest.raises(CorruptJSONLError) as error:
        store.read_jsonl("ledger.jsonl")

    invalid_byte = corrupt_line.find(b"\xff")
    relative_offset = invalid_byte if invalid_byte >= 0 else 1
    assert str(error.value).startswith(f"{ledger}:")
    assert f"line 2, byte offset {len(first) + relative_offset}" in str(error.value)


def test_read_fails_closed_on_corrupt_newline_terminated_final_record(tmp_path):
    store = WorkStore(tmp_path)
    ledger = store.path("ledger.jsonl")
    ledger.write_bytes(b'{"path": "first"}\n{"path": "\xff"}\n')

    with pytest.raises(CorruptJSONLError, match="line 2"):
        store.read_jsonl("ledger.jsonl")


@pytest.mark.parametrize(
    ("corrupt_line", "suffix"),
    [
        (b"{broken}", b'\n{"path": "last"}\n'),
        (b'{"path": "\xff"}', b'\n{"path": "last"}\n'),
        (b'{"path": "\xff"}', b"\n"),
    ],
    ids=["middle-json", "middle-utf8", "final-utf8"],
)
def test_append_fails_closed_without_changing_a_corrupt_complete_ledger(
    tmp_path, corrupt_line, suffix
):
    store = WorkStore(tmp_path)
    ledger = store.path("ledger.jsonl")
    original = b'{"path": "first"}\n' + corrupt_line + suffix
    ledger.write_bytes(original)

    with pytest.raises(CorruptJSONLError, match="line 2"):
        store.append_jsonl("ledger.jsonl", [{"path": "new"}])

    assert ledger.read_bytes() == original


def test_append_revalidates_after_same_length_external_corruption(tmp_path):
    store = WorkStore(tmp_path)
    ledger = store.path("ledger.jsonl")
    store.append_jsonl(
        "ledger.jsonl",
        [{"path": "first"}, {"path": "second"}, {"path": "last"}],
    )
    original = ledger.read_bytes()
    corrupt = original.replace(b"second", b"se\xffond", 1)
    ledger.write_bytes(corrupt)

    with pytest.raises(CorruptJSONLError, match="line 2"):
        store.append_jsonl("ledger.jsonl", [{"path": "new"}])

    assert ledger.read_bytes() == corrupt


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
