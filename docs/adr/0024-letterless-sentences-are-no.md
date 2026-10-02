# ADR-0024: Sentences with no letter are `no` by rule

Status: accepted (2026-10-01)

## Context

The quality audit of the owner found that many website sentences are numeric or punctuation debris
(`12345`, `--`, `+33 1 23`). The model answers `no` on them, but each sentence costs GPU time (approximately 2-5% of
the website input; see the PR for the measured share). The Hub viewer already hides such rows
(`docs/schema.md`, viewer table).

## Decision

- The project labels a sentence `no` without a model call if its text has no alphabetic character:
  `not any(ch.isalpha() for ch in text)` (`domain.sentences.has_no_letters`). The test supports Unicode.
  Thus CJK, Arabic and Cyrillic text still goes to the model. The project does not widen the rule.
- The node runner builds the row itself (`domain.records.rule_decided_no`): `raw_output`
  `</think>no`, `truncated` false, zero generated tokens, no latency, and
  `finish_reason = "rule:no_letters"`, which is the provenance. Each reader parses `raw_output`.
  Thus assemble, publish, stats and the card see a plain `no` (never `failed`). The schema does not change.
- The project computes the generation config fingerprint from the model, the prompt and the sampling config, not from
  the code path. Thus it does not change and the existing `parts/<fp>/` results stay valid.
- Resume does not change: the rows have the key `text_sha256`. `RunStats.rule_decided` counts the rule rows.
  `sentences_per_second` counts only the sentences that the model processed. Thus the calibrated profiles are not inflated.

## Consequences

The rows that the project decided before this change are model answers. Both kinds exist together. To query the rule rows, use
`finish_reason = 'rule:no_letters'`. We assume that no model answers `yes` on a sentence without letters.
The audit found none.
