# Payment fixture diagnostic (2026-11-12)

Historical intermediate diagnostic. The subsequent 643-test pass and current
verification are recorded in `mtg-297-heartbeat-of-spring.md`.

The 75 failures from publication were retried with the exact test names and
`RUST_BACKTRACE=1`, debug nextest CI profile, two build jobs and two test
threads. Result: **38 passed, 37 failed**. Log:
`/tmp/mtg297-repair-diagnostic.log`; exact selection:
`/tmp/mtg297-failed-filter`.

Fixture edits appeared during this investigation and continued during the
diagnostic build. This is evidence of intermediate repairs, not validation of a
stable revision. Those edits were preserved; no engine or fixture source was
changed by this investigation.

The remaining failure classes include:

- Generic payments still pending before priority, stack reads or a second
  activation. Iceberg's two activations each need their payment completed.
- Hybrid payment choices before generic pips: a colorless-only fixture driver
  cannot choose the hybrid pip for Pretending Poxbearers or Earth Kingdom
  Soldier. Pick the intended hybrid color explicitly or supply an exact pool.
- The Lesson setup replaces its pool for the resolution payment before
  completing the cast payment. Complete casting first, preserving the existing
  nested-production and optional-payment assertions.
- Default spending consumes future colored resources in the repeated Spectral
  Sailor witness. Explicit colorless payment preserves blue for all three draws.
- Kindly Customer's hand count is captured while the spell is still being paid
  for. Capture the count after casting completes, before its draw resolves.
- Fire Lord Azula's Bolt has regular and combat red: choose the intended expiry
  explicitly before expecting its cast trigger.

Some of these sites were already being repaired after the diagnostic compiled.
Reinspect the current diff before applying changes. Priority, targeting and
generic mode helpers must continue to reject a pending spending decision rather
than silently selecting its mana.
