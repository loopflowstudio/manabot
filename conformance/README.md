# Conformance fixtures

Current-world (w17) corpora with executable consumers:

- `semantic-kernel-w17-v6/` — seeded cases (`manifest.json`) whose regenerated
  traces must match `evidence.json`'s per-case `trace_sha256`, plus property and
  Phase-matrix checks (`managym/tests/semantic_conformance_tests.rs`, CI fuzzing).
  A drifted case is written to `target/semantic-conformance/semantic-kernel-w17-v6/`;
  inspect it with `cargo run --manifest-path managym/Cargo.toml --bin
  semantic_conformance -- replay <file>`, and re-record an accepted change with
  `... -- record --root conformance/semantic-kernel-w17-v6`.
- `authored-match-w17-v2/` — the authored-match authority receipt
  (`conformance.authored_match_receipt`, `scripts/generate_authored_match_receipt.py`).
  Its declared policy uniformly chooses strategic offers and explicitly uses
  automatic payment at source-only decisions; directed payment tests cover manual taps.
- `authored-match-parity-w17-v2/` — Academy live/headless replay parity
  (`scripts/verify-authored-match-parity`).
- `public-commitment-parity-w17-v2/` — Academy public-commitment parity
  (`scripts/verify-public-commitment-parity`).

Any change to w17 card content moves its content digest, so every corpus above
is re-recorded together. The authored-match receipt's SHA-256 is pinned as
`AUTHORITY_SHA256` in `conformance/authored_match_parity.py` and
`conformance/public_commitment_parity.py`, and its terminal state as
`TERMINAL_STATE`. Update those pins, then run
`uv run --extra dev python -m conformance.authored_match_parity write` and
`uv run --extra dev python -m conformance.public_commitment_parity write`; the
latter needs the RUL-9 measurement from the artifact store. The same digest
binds `managym/academy/fixtures/*-w17.json` and the belief v3 receipt pins in
`tests/belief/test_state.py`. A state-hash contract change (object ids or
`MATCH_STATE_HASH_VERSION`) moves every pinned digest the same way without
touching the content digest; `scripts/refresh_academy_fixtures.py` replays the
Academy fixtures' recorded Commands and rewrites their digests, and the
presentation pins in `tests/managym/academy/test_authored_match_parity.py` follow.

Earlier worlds' corpora (semantic-kernel w8–w15, authored-match and parity v1
through w11, public-commitment-parity-v1, the Learn w4 cohorts) are archived
byte-for-byte in S3 under `experiments/data/archive/conformance/`; see the
[repository archive](../docs/development.md#retained-evidence-and-archived-corpora) for the fetch command. They
verify only at the revision that recorded them.

The preceding semantic-kernel w17 v1–v3 corpora and authored-match/parity w17
v1 receipts are retained unchanged as evidence. The current versions bind
the catalog containing Release to Memory, Lita and their distinct tokens;
semantic-kernel w17-v4 also hashes viewer-known library positions, which v3's
state hashes predate. Its commands, action spaces and pending choices equal v3's.

Semantic-kernel w17-v5 records the default priority surface with choice-output
mana floats disabled and automatic payment from simple choice-output sources.
The v4 corpus retains its original commands and evidence.

Semantic-kernel w17-v6 hashes states under `MATCH_STATE_HASH_VERSION` 17,
which adds the object that produced each pooled unit of mana and each unit a
payment spent. Its commands, action spaces and pending choices equal v5's;
the v5 corpus is retained unchanged.
