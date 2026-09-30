"""Properties of the publication path: label rows, published statistics and the dataset card."""

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from hypothesis import given, settings
from hypothesis import strategies as st

from landuse_filter.adapters.readers import DESCRIPTION, WEBSITE, WIKI
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import assemble
from landuse_filter.application import published_stats as ps
from landuse_filter.application.card import CardFacts, MapFacts, render_card
from landuse_filter.application.datasets import SPECS
from landuse_filter.domain.sentences import SentenceRef

ALLOWED = {"yes", "no", "failed", "skipped_unsplit", "pending"}

# --- assemble.label_rows / build_labels ---------------------------------------------

texts = st.sampled_from(["a", "b", "c", "d", "e"])
resolutions = st.tuples(
    st.sampled_from(["yes", "no", "failed"]),
    st.sampled_from([None, "strict", "lenient"]),
    st.sampled_from([None, "truncated", "empty"]),
)


@st.composite
def worlds(draw):
    """Distinct sentence positions (some unsplit) plus a partial lookup over their texts."""
    n = draw(st.integers(0, 15))
    refs = [
        SentenceRef(
            WIKI,
            "wikipedia/sentences/x.parquet",
            (("sentence_id", f"s{i}"),),
            draw(texts),
            "en",
            unsplit=draw(st.booleans()),
        )
        for i in range(n)
    ]
    wanted = {r.text for r in refs if not r.unsplit}
    resolved = {
        text: draw(resolutions)
        for text in wanted
        if draw(st.booleans())  # some texts have no generation yet
    }
    return refs, {SentenceRef(WIKI, "f", (), t).text_sha256: v for t, v in resolved.items()}


def unresolved(refs, lookup) -> bool:
    return any(not r.unsplit and r.text_sha256 not in lookup for r in refs)


@given(worlds(), st.booleans())
def test_label_rows_are_one_per_reference_with_unique_deterministic_ids(world, allow_pending):
    refs, lookup = world
    if unresolved(refs, lookup) and not allow_pending:
        return
    rows = assemble.label_rows(refs, lookup, "fp", "rev", allow_pending=allow_pending)
    assert len(rows) == len(refs)
    assert [r["label_id"] for r in rows] == [r.label_id for r in refs]
    assert len({r["label_id"] for r in rows}) == len(refs)
    assert rows == assemble.label_rows(refs, lookup, "fp", "rev", allow_pending=allow_pending)
    for ref, row in zip(refs, rows, strict=True):
        assert row["decision"] in ALLOWED
        assert row["text_sha256"] == ref.text_sha256
        assert row["sentence_id"] == dict(ref.locator)["sentence_id"]
        no_generation = row["decision"] in ("skipped_unsplit", "pending")
        assert (row["generation_id"] is None) == no_generation
        assert (row["decision"] == "skipped_unsplit") == ref.unsplit


@given(worlds())
def test_pending_rows_exist_only_when_allowed_and_only_for_unresolved_texts(world):
    refs, lookup = world
    if unresolved(refs, lookup):
        try:
            assemble.label_rows(refs, lookup, "fp", "rev")
        except assemble.MissingGenerationError:
            pass
        else:
            raise AssertionError("an unresolved text must not be publishable")
    else:
        rows = assemble.label_rows(refs, lookup, "fp", "rev", allow_pending=True)
        assert all(r["decision"] != "pending" for r in rows)
    rows = assemble.label_rows(refs, lookup, "fp", "rev", allow_pending=True)
    pending = {r["label_id"] for r in rows if r["decision"] == "pending"}
    expected = {r.label_id for r in refs if not r.unsplit and r.text_sha256 not in lookup}
    assert pending == expected


def test_build_labels_writes_a_table_with_one_row_per_sentence(tmp_path):
    local = tmp_path / "s.parquet"
    pq.write_table(
        pa.table(
            {
                "sentence_id": ["s1", "s2", "s3"],
                "language": ["en"] * 3,
                "text": ["a", "b", "c"],
                "segmentation_status": ["split", "unsupported_language", "split"],
            }
        ),
        local,
    )
    lookup = {SentenceRef(WIKI, "f", (), "a").text_sha256: ("yes", None, None)}
    n = assemble.build_labels(
        WIKI,
        "wikipedia/sentences/s.parquet",
        local,
        resolved=lookup,
        stamp=assemble.Stamp("fp", "rev"),
        out=tmp_path / "out",
        allow_pending=True,
    )
    table = pq.read_table(tmp_path / "out" / "labels" / "wikipedia" / "sentences" / "s.parquet")
    assert table.num_rows == n == 3
    assert table.column("decision").to_pylist() == ["yes", "skipped_unsplit", "pending"]
    assert len(set(table.column("label_id").to_pylist())) == 3


# --- published_stats.complete / totals ----------------------------------------------

counts = st.dictionaries(st.sampled_from(["yes", "no", "failed", "pending"]), st.integers(1, 50))
label_stats = st.builds(
    ps.FileStats,
    st.just(""),
    decisions=counts,
    failures=st.dictionaries(st.sampled_from(["truncated", "empty"]), st.integers(1, 9)),
    cells=st.one_of(
        st.none(),
        st.dictionaries(
            st.sampled_from(["c1", "c2", "c3"]),
            st.lists(st.integers(0, 20), min_size=2, max_size=2),
        ),
    ),
    labelled=st.integers(0, 100),
    located=st.integers(0, 100),
)
gen_stats = st.builds(
    ps.FileStats,
    st.just(""),
    rows=st.integers(0, 100),
    gpus=st.dictionaries(st.sampled_from(["l40s", "a100"]), st.integers(1, 50)),
)


@st.composite
def records(draw):
    """Distinct paths, each a labels or a generations record."""
    labels = draw(st.lists(label_stats, max_size=6))
    gens = draw(st.lists(gen_stats, max_size=6))
    out = [_with_path(r, f"labels/f{i}.parquet") for i, r in enumerate(labels)]
    out += [_with_path(r, f"generations/fp/p{i}.parquet") for i, r in enumerate(gens)]
    return out


def _with_path(record: ps.FileStats, path: str) -> ps.FileStats:
    return ps.FileStats(
        path, record.decisions, record.failures, record.rows, record.gpus,
        record.cells, record.labelled, record.located,
    )  # fmt: skip


def expected_sum(records, attr):
    total: dict[str, int] = {}
    for r in records:
        for k, v in getattr(r, attr).items():
            total[k] = total.get(k, 0) + v
    return total


@given(records(), st.randoms(use_true_random=False))
def test_totals_equal_the_sum_of_records_whatever_their_order(recs, rnd):
    shuffled = list(recs)
    rnd.shuffle(shuffled)
    a, b = ps.totals(recs), ps.totals(shuffled)
    assert a == b
    assert a.decisions == expected_sum(recs, "decisions")
    assert a.failures == expected_sum(recs, "failures")
    assert a.gpus == expected_sum(recs, "gpus")
    assert a.unique_texts == sum(r.rows for r in recs)
    assert (a.labelled, a.located) == (
        sum(r.labelled for r in recs),
        sum(r.located for r in recs),
    )
    cell_totals = [sum(v) for v in a.cells.values()]
    assert sum(cell_totals) == sum(sum(v) for r in recs for v in (r.cells or {}).values())


def opener_for(files: dict[str, Path]):
    return lambda p: files[p]


def labels_parquet(path: Path, decisions: list[str]) -> Path:
    pq.write_table(
        pa.table({"decision": decisions, "failure_reason": [None] * len(decisions)}), path
    )
    return path


@settings(max_examples=25)
@given(st.lists(st.sampled_from(["yes", "no", "failed"]), min_size=1, max_size=8),
       st.lists(st.sampled_from(["yes", "no", "failed"]), min_size=1, max_size=8))  # fmt: skip
def test_refreshing_a_path_replaces_its_record_and_never_double_counts(
    tmp_path_factory, before, after
):
    tmp = tmp_path_factory.mktemp("refresh")
    store = WorkStore(tmp / "w")
    path = "labels/a.parquet"
    published = {path}
    first = labels_parquet(tmp / "a1.parquet", before)
    ps.complete(store, DESCRIPTION, published, opener_for({path: first}))
    second = labels_parquet(tmp / "a2.parquet", after)
    records = ps.complete(store, DESCRIPTION, published, opener_for({path: second}), refresh={path})
    assert len(records) == 1
    assert ps.totals(records).decisions == {k: after.count(k) for k in set(after)}
    # a later plain call reads the ledger: still the refreshed record, counted once
    again = ps.complete(store, DESCRIPTION, published, opener_for({}))
    assert ps.totals(again) == ps.totals(records)


def test_legacy_ledger_lines_still_load(tmp_path):
    store = WorkStore(tmp_path)
    store.append_jsonl(
        ps.ledger(WEBSITE),
        [
            {"path": "labels/a.parquet", "decisions": {"yes": 2}, "failures": {}},  # pre-map
            {"path": "generations/fp/p.parquet", "rows": 4, "gpus": {"l40s": 4}},
            {"path": "labels/b.parquet"},  # oldest: counts only the path
        ],
    )
    published = {"labels/a.parquet", "generations/fp/p.parquet", "labels/b.parquet"}
    recs = ps.complete(store, WEBSITE, published, opener_for({}))
    total = ps.totals(recs)
    assert (total.decisions, total.unique_texts, total.gpus) == ({"yes": 2}, 4, {"l40s": 4})
    assert total.cells == {}
    assert total.labelled == total.located == 0


# --- card.render_card -----------------------------------------------------------------

GATE = {
    "delta_f1": 0.0,
    "delta_mcc": 0.0,
    "mcc_lower": -0.01,
    "delta_accuracy": 0.0,
    "delta_failed_rate": 0.0,
}


@st.composite
def card_facts(draw):
    yes, no, skipped, pending = (draw(st.integers(0, 10_000)) for _ in range(4))
    reasons = draw(
        st.dictionaries(st.sampled_from(["truncated", "empty", "no_label"]), st.integers(1, 500))
    )
    failed = sum(reasons.values()) + draw(st.integers(0, 50))
    decisions = {"yes": yes, "no": no, "failed": failed, "skipped_unsplit": skipped}
    if pending:
        decisions["pending"] = pending
    total_files = draw(st.integers(1, 400))
    labelled = draw(st.integers(0, total_files))
    labelled_rows = yes + no
    world_map = None
    if draw(st.booleans()) and labelled_rows:
        located = draw(st.integers(0, labelled_rows))
        world_map = MapFacts(draw(st.integers(0, 50)), located, labelled_rows, 0.5)
    return CardFacts(
        dataset=draw(st.sampled_from(sorted(SPECS))),
        revision="b4706eb0",
        model="org/m@654f1234",
        draft="org/d@458c1234",
        max_new_tokens=4096,
        fingerprint="fp",
        labelled_files=labelled,
        total_files=total_files,
        decisions=decisions,
        failures=reasons,
        unique_texts=draw(st.integers(0, 1000)),
        gpu_rows={"l40s": draw(st.integers(1, 100))},
        admitted={"l40s": GATE},
        world_map=world_map,
        partial_files=draw(st.integers(0, 5)),
    )


def decision_rows(card: str) -> dict[str, tuple[int, float]]:
    found = re.findall(r"^\| `(\w+)` \| ([\d,]+) \| ([\d.]+)% \|", card, re.M)
    return {k: (int(n.replace(",", "")), float(p)) for k, n, p in found if k in ALLOWED}


@given(card_facts())
def test_card_renders_with_consistent_shares_status_and_pending_row(f):
    card = render_card(f)
    total = sum(f.decisions.values())
    assert f"**total** | **{total:,}**" in card
    status = "complete" if f.labelled_files == f.total_files else "in_progress"
    assert f"dataset_status: {status}" in card
    rows = decision_rows(card)
    assert sum(n for n, _ in rows.values()) == total
    if total:
        assert abs(sum(p for _, p in rows.values()) - 100) <= 0.05 * len(rows) + 1e-9
        assert set(rows) == {k for k in f.decisions if k != "pending" or f.decisions[k]} | {
            "yes", "no", "failed", "skipped_unsplit"
        }  # fmt: skip
    else:
        assert rows == {}
    assert ("`pending`" in card.split("## Tables")[0]) == bool(f.decisions.get("pending"))
    assert ("### Failed rows" in card) == bool(f.failures)
    assert ("## Where the labels are" in card) == (f.world_map is not None)
    assert ("more partially" in card) == bool(f.partial_files)
