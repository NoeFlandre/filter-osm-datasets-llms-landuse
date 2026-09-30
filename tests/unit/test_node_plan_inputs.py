from landuse_filter.cli import node


def test_plan_inputs_keys(monkeypatch):
    from landuse_filter.adapters import readers, tokenizer
    from landuse_filter.application import plan

    monkeypatch.setitem(readers.SOURCES, "fake", object())
    monkeypatch.setattr(plan, "list_input_files", lambda source, rev: ["a", "b"])
    monkeypatch.setattr(tokenizer, "chat_encoder", lambda *_: "enc")
    monkeypatch.setattr(node, "_template", lambda: "tpl")
    got = node._plan_inputs("fake", "rev", 7)
    assert set(got) == {"files", "fetch", "encode", "template", "chunk_size", "forget"}
    assert (got["files"], got["encode"], got["template"], got["chunk_size"]) == (
        ["a", "b"],
        "enc",
        "tpl",
        7,
    )
