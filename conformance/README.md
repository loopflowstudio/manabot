# Conformance fixtures

Current-world (w17) corpora with executable consumers:

- `semantic-kernel-w17-v6/` — seeded cases (`manifest.json`) whose regenerated
  traces must match `evidence-opening-v1.json`'s per-case `trace_sha256`, plus property and
  Phase-matrix checks (`managym/tests/semantic_conformance_tests.rs`, CI fuzzing).
  A drifted case is written to `target/semantic-conformance/semantic-kernel-w17-v6/`;
  inspect it with `cargo run --manifest-path managym/Cargo.toml --bin
  semantic_conformance -- replay <file>`, and re-record an accepted change with
  `... -- record --root <staged-corpus>` using a copy of the manifest and Phase matrix.
- `authored-match-w17-v2/` — the authored-match authority receipt
  (`conformance.authored_match_receipt`, `scripts/generate_authored_match_receipt.py`).
  Its declared policy uniformly chooses strategic offers and explicitly uses
  automatic payment at source-only decisions; directed payment tests cover manual taps.
- `authored-match-parity-w17-v2/` — Academy live/headless replay parity
  (`scripts/verify-authored-match-parity`).
- `public-commitment-parity-w17-v2/` — Academy public-commitment parity
  (`scripts/verify-public-commitment-parity`).

The opening-hand rules use `evidence-opening-v1.json` and the authored-match
authority and parity files ending in `-opening.json`. The authored trace
takes two explicit keep Commands, then follows the seeded priority line.
Generate its authority with `scripts/generate_authored_match_receipt.py`
and its parity with `uv run --extra dev python -m conformance.authored_match_parity write`.
The earlier receipts and public-commitment evidence remain bound to the
revisions that recorded them.

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
payment spent. Mana made while a cost is being paid pays toward it at once,
so a pending payment's owed cost and receipt, and the offers after a tap made
during it, differ from v5's; the v5 corpus is retained unchanged.
