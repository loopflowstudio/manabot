# Conformance fixtures

Only current-world (w17) corpora with executable consumers stay in source:

- `semantic-kernel-w17-v1/` — Rust replay, property and Phase-matrix checks
  (`managym/tests/semantic_conformance_tests.rs`) and the CI conformance gate.
- `authored-match-w17-v1/` — the authored-match authority receipt
  (`academy.authored_match_receipt`, `scripts/generate_authored_match_receipt.py`).
- `authored-match-parity-w17-v1/` — Academy live/headless replay parity
  (`scripts/verify-authored-match-parity`).
- `public-commitment-parity-w17-v1/` — Academy public-commitment parity
  (`scripts/verify-public-commitment-parity`).

Earlier worlds' corpora (semantic-kernel w8–w15, authored-match and parity v1
through w11, public-commitment-parity-v1, the Learn w4 cohorts) are archived
byte-for-byte in S3 under `experiments/data/archive/conformance/`; see the
[repository archive](../docs/repository-archive.md) for the fetch command. They
verify only at the revision that recorded them.
