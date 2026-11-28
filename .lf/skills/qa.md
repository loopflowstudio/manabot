---
extends: builtin
---
## In this repository

Never induce load in tests ([TESTING.md](../../TESTING.md)). Even one large
constructed board is a workload. Keep runs, builds and `ETUDE_HOME` under ignored `.runs/`.
Build the native extension once per pass, after the Rust edits.

When `MATCH_STATE_HASH_VERSION` changes, replay the live Academy fixtures with
`uv run --extra dev python scripts/refresh_academy_fixtures.py` and verify the
current [conformance fixtures](../../conformance/README.md). Check that recorded
commands still reach the expected decisions before recording new digests.
