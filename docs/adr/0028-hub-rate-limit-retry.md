# ADR-0028: Retry Hub rate limits (429) in the bucket adapter

Status: accepted (2026-10-05)

## Context

About 35 GPU jobs, 3 controllers and the publish jobs share one Hub account, limited to 1000 API
requests per 5 minutes. The Hub then answers 429 with a `Retry-After` header. The bucket adapter
did not retry. The wiki publish job (OAR 6955164) crashed on `ls('index/')`. The controllers lost a
whole ingest or cycle (`ingest during submission failed`, `cycle failed; retrying in 120s`) on
`tree/jobs/` and `tree/parts/<fp>/` listings. `luf bench admit` failed the same way.

## Decision

- `domain/retry.py` (pure): `parse_retry_after`, `next_delay`, `give_up`. The server's value wins
  (+1 s, +0..1 s jitter); without it, exponential backoff with jitter from 2 s. One wait is at most
  300 s, the waits together at most 1200 s, at most 10 calls.
- `adapters/remote.py`: `with_retry` wraps every Hub call of `BucketRemote` (`ensure`, `put`,
  `get`, `ls`, `delete`). Hub and httpx 429 errors are retried; sleep and jitter are injectable.
  After the limits the original error is re-raised. One log line per wait.
- `ls` pages the tree listing itself, so that a 429 on a later page retries that page, not the
  whole listing.
- The `Remote` protocol and `DirRemote` do not change. Hub dataset helpers (`adapters/hub.py`,
  publish) are not wrapped.

## Consequences

Rate-limit bursts cost a delay, not a lost cycle or job. A sustained limit beyond 20 minutes
still fails, as before. Controllers must be restarted to pick up the change.
