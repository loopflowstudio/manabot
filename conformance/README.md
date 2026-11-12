# Conformance fixtures

Current-world (w17) corpora with executable consumers:

- `semantic-kernel-w17-v2/` — seeded cases (`manifest.json`) whose regenerated
  traces must match `evidence.json`'s per-case `trace_sha256`, plus property and
  Phase-matrix checks (`managym/tests/semantic_conformance_tests.rs`, CI fuzzing).
  A drifted case is written to `target/semantic-conformance/semantic-kernel-w17-v2/`;
  inspect it with `cargo run --manifest-path managym/Cargo.toml --bin
  semantic_conformance -- replay <file>`, and re-record an accepted change with
  `... -- record --root conformance/semantic-kernel-w17-v2`.
- `authored-match-w17-v1/` — the authored-match authority receipt
  (`academy.authored_match_receipt`, `scripts/generate_authored_match_receipt.py`).
- `authored-match-parity-w17-v1/` — Academy live/headless replay parity
  (`scripts/verify-authored-match-parity`).
- `public-commitment-parity-w17-v1/` — Academy public-commitment parity
  (`scripts/verify-public-commitment-parity`).

Any change to w17 card content moves its content digest, so every corpus above
is re-recorded together. The authored-match receipt's SHA-256 is pinned as
`AUTHORITY_SHA256` in `academy/authored_match_parity.py` and
`academy/public_commitment_parity.py`, and its terminal state as
`TERMINAL_STATE`. Update those pins, then run
`uv run --extra dev python -m academy.authored_match_parity write` and
`uv run --extra dev python -m academy.public_commitment_parity write`; the
latter needs the RUL-9 measurement from the artifact store. The same digest
binds `academy/fixtures/*-w17.json` and the belief v3 receipt pins in
`tests/belief/test_state.py`. A state-hash contract change (object ids or
`MATCH_STATE_HASH_VERSION`) moves every pinned digest the same way without
touching the content digest; `scripts/refresh_academy_fixtures.py` replays the
Academy fixtures' recorded Commands and rewrites their digests, and the
presentation pins in `tests/academy/test_authored_match_parity.py` follow.

Earlier worlds' corpora (semantic-kernel w8–w15, authored-match and parity v1
through w11, public-commitment-parity-v1, the Learn w4 cohorts) are archived
byte-for-byte in S3 under `experiments/data/archive/conformance/`; see the
[repository archive](../docs/development.md#retained-evidence-and-archived-corpora) for the fetch command. They
verify only at the revision that recorded them.

The preceding `semantic-kernel-w17-v1` corpus is retained unchanged as evidence;
`semantic-kernel-w17-v2` records the composable-layer rules.
