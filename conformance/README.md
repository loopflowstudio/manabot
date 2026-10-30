# Conformance fixtures

Only current-world (w17) corpora with executable consumers stay in source:

- `semantic-kernel-w17-v1/` — seeded cases (`manifest.json`) whose regenerated
  traces must match `evidence.json`'s per-case `trace_sha256`, plus property and
  Phase-matrix checks (`managym/tests/semantic_conformance_tests.rs`, CI fuzzing).
  A drifted case is written to `target/semantic-conformance/semantic-kernel-w17-v1/`;
  inspect it with `cargo run --manifest-path managym/Cargo.toml --bin
  semantic_conformance -- replay <file>`, and re-record an accepted change with
  `... -- record --root conformance/semantic-kernel-w17-v1`.
- `authored-match-w17-v1/` — the authored-match authority receipt
  (`academy.authored_match_receipt`, `scripts/generate_authored_match_receipt.py`).
- `authored-match-parity-w17-v1/` — Academy live/headless replay parity
  (`scripts/verify-authored-match-parity`).
- `public-commitment-parity-w17-v1/` — Academy public-commitment parity
  (`scripts/verify-public-commitment-parity`).

Earlier worlds' corpora (semantic-kernel w8–w15, authored-match and parity v1
through w11, public-commitment-parity-v1, the Learn w4 cohorts) are archived
byte-for-byte in S3 under `experiments/data/archive/conformance/`; see the
[repository archive](../docs/development.md#retained-evidence-and-archived-corpora) for the fetch command. They
verify only at the revision that recorded them.
