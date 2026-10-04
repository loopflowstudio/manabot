# Learn same-Command cohort, w4 v3

All 16 seed/seat attempts were registered before scoring against source
checkpoint `dc0f04b8`. `./scripts/verify-learn-lesson run
conformance/learn-lesson-w4-v3` rebuilt the native extension, passed 57 focused
consumer checks, and completed all 16 games with zero live/headless/replay
state, ordered-consequence or viewer-observation mismatches.

The 3,843 Commands include 19 Lesson retrievals, 25 discard/draw choices and
14 declines. Every compressed tape is byte-identical to the corresponding
retained v2 tape; the source registration differs after diagnostic/import fixes.
Each gzip file retains the full synthetic authority tape, full setup, canonical
replay and both viewer digests. `summary.json` binds each file by SHA-256.

`./scripts/verify-learn-lesson` verifies the retained cohort without overwriting
it. Source or extension drift fails closed; a changed source needs a new
prospective registration, not edits to these receipts. The exact compiled binary
binding makes this a local build receipt, not a cross-platform CI certification.

This proves automated conformance only. It supplies no missing historical
human Command tapes, trained-challenger admission, strength or runtime budget.
The historical Python/advice gate blockers remain documented in
`docs/rules/learn-lesson.md`; ETU-75 remains open.
