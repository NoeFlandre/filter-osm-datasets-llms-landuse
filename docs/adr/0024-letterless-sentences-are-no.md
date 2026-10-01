# ADR-0024: Sentences with no letter are `no` by rule

Status: accepted (2026-10-01)

## Context

The owner's quality audit found that many website sentences are numeric or punctuation debris
(`12345`, `--`, `+33 1 23`). The model answers `no` on them, but each costs GPU time (about 2-5% of
the website input, see the PR for the measured share). The Hub viewer already hides such rows
(`docs/schema.md`, viewer table).

## Decision

- A sentence whose text has no alphabetic character, `not any(ch.isalpha() for ch in text)`
  (`domain.sentences.has_no_letters`), is labelled `no` without calling the model. The test is
  Unicode-aware: CJK, Arabic and Cyrillic text still goes to the model. The rule is not widened.
- The node runner builds the row itself (`domain.records.rule_decided_no`): `raw_output`
  `</think>no`, `truncated` false, zero generated tokens, no latency, and
  `finish_reason = "rule:no_letters"`, which is the provenance. Every reader parses `raw_output`,
  so assemble, publish, stats and the card see a plain `no` (never `failed`); the schema is unchanged.
- The generation config fingerprint is computed from the model, prompt and sampling config, not from
  the code path, so it is unchanged and existing `parts/<fp>/` results stay valid.
- Resume is unchanged: rows are keyed by `text_sha256`. `RunStats.rule_decided` counts rule rows;
  `sentences_per_second` counts model-processed sentences only, so calibrated profiles are not inflated.

## Consequences

Rows decided before this change were model answers; both kinds coexist. Query rule rows with
`finish_reason = 'rule:no_letters'`. A model that would answer `yes` on a letterless sentence is
assumed not to exist; the audit found none.
