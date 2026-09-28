from landuse_filter.domain.hashing import sha256_parts, sha256_text
from landuse_filter.domain.sentences import Decision, SentenceRef


def test_text_hash_is_exact_bytes():
    assert sha256_text("é") != sha256_text("é")


def test_parts_separator_prevents_collisions():
    assert sha256_parts("ab", "c") != sha256_parts("a", "bc")


def test_label_id_depends_on_dataset_and_locator_only():
    a = SentenceRef("d", "f1", (("sentence_id", "x"),), "text one")
    b = SentenceRef("d", "f2", (("sentence_id", "x"),), "text two")
    c = SentenceRef("e", "f1", (("sentence_id", "x"),), "text one")
    assert a.label_id == b.label_id
    assert a.label_id != c.label_id
    assert a.text_sha256 == sha256_text("text one")


def test_decision_vocabulary():
    assert [d.value for d in Decision] == ["yes", "no", "failed", "skipped_unsplit"]
