# MTG-276 — fixed-seat observations

## Contract and provenance

One opt-in native batch call writes a requested native seat's viewer-safe
observation per slot, including nonchoosing seats, into caller-owned arrays.
The experience and example are in `docs/engine.md#fixed-seat-observations`.
MTG-272 is integrated; no Wave or separate accepted design applies.

Jack Heart (Task brief, 2026-11-11): "Yeah i think we should do that one, it can
be its own task, but start it".

Jack Heart's MTG-272 direction: "the output of the engine is just waht was in
hand, choosing to use it as a belief SL target is the business of hte client".
Hand snapshots remain separate privileged records, never viewer-safe inputs.

Reversible API choices, not attributed user decisions:
`write_observations_for_players(players, buffers, action_overflow="error")`
returns immutable `managym.ObservationRead` records with `slot`, `player`,
`game_id`, `seed`, `revision`. Both wrappers accept caller-owned NumPy arrays;
Torch conversion/device transfer belongs to the consumer. HandSnapshot has no
slot field, so pair by list index and game ID/revision. No extra identity owner.

Scalar `Observation::for_player` owns visibility and deliberately omits
presentation events even for the chooser. Compare to scalar fixed-seat reads,
not ordinary step results. Requested seats never remap under player control.
Failed writes require discarding every output row. Ordinary calls do no extra
projection work. Frozen layouts, worlds, checkpoints and evidence stay unchanged.

## Implementation and remaining verification

The synced checkout already provided observation-only buffer validation and
field slices for `encode_observations_into`; reuse them without further buffer
refactoring. Native whole-request validation precedes callbacks. The Python
binding holds the GIL, derives capacities/version from caller arrays and keeps
execution buffers, config, last_info and capture prefixes untouched.

Compress (2026-11-14): native `write_observations_for_players` builds every
`ObservationRead` before the first write, replacing a separate revision-check
loop. Reads now take `Env::reset_seed()`, the value `HandSnapshot` reports; they
previously took `Env::seed()`, the next reset's seed, which differs after a
pending `set_seed` or failed reset. The native test sets a pending seed.

Focused fixtures cover both input versions, scalar Python/native encoding
parity, mixed/nonchoosing seats, closed decklists, inactive slots, identity
pairing, unchanged execution/capture state, malformed requests/buffers, capacities
and explicit action overflow, seed rebuilds, root reassignment, terminal and
failed-slot replacement. Player control uses a bounded native fixture because
the public vector API has no state-injection method for it.

Implementation and focused verification are complete. Gate owns affected suites
and full acceptance: Rust format, all-feature Clippy
and debug suite per `docs/development.md#validation-by-change-area`, plus
`tests/env/test_observation.py`, `tests/training/test_action_cutoff.py`,
`tests/verify/test_native_buffer_capacities.py`, `tests/env/test_hand_snapshots.py`
and `tests/env/test_captured_vector_bookkeeping.py`. No timing benchmark needed.

### Delete — do not maintain

No public predecessor exists and no buffer plumbing was replaced. Retain one
shared validation/encoding path, reset/step outcome writes, chooser buffers,
Commands, capture semantics and complete native legality. No migration or shim.

## Concept review (2026-11-14)

No design change and no open product choice. One action (a fixed-seat batch
read), three concepts already published by hand snapshots (slot = list index,
native seat, game ID/seed/revision boundary) and one recovery rule (discard all
rows after a failure). Considered and kept: `ObservationRead.slot` duplicates
the list index that `HandSnapshot` relies on, but it survives a collector's
mask filter and costs no extra identity owner; the method name follows scalar
`observation_for_player` rather than `*_into_buffers`, which means the
registered execution buffers this call must not touch.

## Checks

- After compress: `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib fixed_seat_observations -- --test-threads 2`: 1 passed (debug).
- `cargo fmt --manifest-path managym/Cargo.toml --all`, Ruff check/format and `git diff --check`: passed.
- `CARGO_BUILD_JOBS=2 uv run --extra play --extra dev maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`: passed; matching CPython 3.13 extension installed atomically from the wheel.
- `uv run --no-sync pytest tests/env/test_fixed_seat_observations.py -q`: 13 passed in 3.17s after the compress rebuild.
- Fixed-seat guide example executed against the rebuilt public API: passed.

Earlier planning/sync/write-docs history is in
`250bb3ba44ee699ae46cac41b0dfc78d110894de:scratch/questions.md`.
Unrelated b015d5/C1/C2 evidence and exact quotes remain at
`757bb6257ef07cf2536ad7f7ec24f0f7309628c6:scratch/questions.md`.

## QA (2026-11-14)

Guide example and an 8-slot Torch-wrapper collector (600 random steps, v9 and
v10, closed decklists) passed; report in `scratch/qa.html`. Fixed: Torch tensors
passed as buffers raised `AttributeError: 'torch.dtype' object has no attribute
'name'` from the shared `require_numpy_array`; it now raises a `TypeError` naming
NumPy and `torch.from_numpy`. Left as is: seat `-1` gives PyO3 `OverflowError`;
other-version buffers are accepted by design.
