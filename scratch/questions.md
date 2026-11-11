# MTG-272 implementation — 2026-11-11

Ryan's request: "one more request: can you add true hands in batched mode?"
The originating clarification: "yeah, he is training a belief model."
Task scope also includes a code-evidenced scalar/batch API audit. MTG-271 owns
version/dependency/release files; no publication belongs to this step.

## Plan and reversible choices

Implement the contract in `docs/engine.md`, Choosing a Python API and Hand
snapshots: explicit native seats, complete name counts, detached typed
records, slot/game/seed/revision identity, native/CapturedVector/Torch access.
No observation, policy input, capture payload or frozen identity changes.

Use the existing private Env reset generation as the snapshot's game identity.
Allocate its values process-wide instead of per Env so replacing a slot's Env
or rebuilding the Torch wrapper's native vector cannot reuse a generation.
It is an opaque lifetime token, never a replay/world identity or promised
consecutive counter. Capture origin and stale checks retain their semantics;
the token is not serialized into public captures. No hand work occurs until
requested. Failed-call rows remain caller-discarded as documented.

Delete — do not maintain: remove the prerelease `HandTarget` / `hand_targets`
names and rename their exclusive test file to `test_hand_snapshots.py`; retain
no compatibility aliases. Drop inherited
b015d5/C2 scratch, retained at
`757bb6257ef07cf2536ad7f7ec24f0f7309628c6:scratch/questions.md` and its linked
artifact evidence. Those experiments and their open decisions are unrelated.

## Implementation handoff

Implemented the documented native/typed/wrapper API. The code-evidenced audit
and three independent follow-ups are in `docs/python-api-audit.md`: capture
bookkeeping when mixing readable/buffered mutation, seed-saturation capture
rollover, and wrapper reset semantics. No release/dependency files changed.
The doc handoff's inactive-buffer claim was contradicted by the Python binding:
reward/done are cleared while observations are retained; docs and tests now agree.

Focused verification: `cargo test -p managym --lib hand_targets` (debug) —
2 passed; `hand_targets_keep_native_seats` — 1 passed. The Python cases passed
in focused runs, including stale capture rejection after same-seed reset.
Initial failures were test fixtures (Reward required its config; two one-card
decks auto-terminated before the empty-hand read), corrected without changing
runtime behavior. Ruff passed on the changed Python API/test files.

Final proof: release wheel rebuilt and local extension replaced atomically;
installed with `uv pip install --no-deps --target` outside the checkout. Running
`uv run --no-project --python <venv>/bin/python -m pytest --confcutdir=.
test_hand_targets.py -q` there passed all 8 cases in 1.97 s. Imported native and
wrapper paths were verified under the installed target; compatibility aliases
and packaged stubs matched. The exact engine-guide example ran (both targets
contained seven cards). This reuses development dependencies and proves API
packaging, not isolated dependency resolution or release readiness. Temporary
installation files were removed. No server or watcher was started.

Gate owns the complete debug native suite, Clippy, affected environment/capture
suites and package-boundary checks. No world or observation shape changed.

## Compress — 2026-11-11

`Env::hand_counts` owns native-seat validation; the batch accessor now checks
only batch shape/readiness before collecting those reads. The large-hand fixture
starts empty and asserts exactly 70 cards, avoiding an unexplained lower bound.
The guide keeps the full contract
with less repetition and describes the process-wide reset-token allocation.
The existing capture-origin/generation checks and typed wrapper methods remain.

Naming direction, comment c1389729: Jack Heart said "assumign its the input to
a belief model seems unnecessary, shouldnt wejust label it as a HandInfo or
smoething". The steer selects `HandSnapshot` / `hand_snapshots(players)`.
Follow-up d793c7d4: "the output of the engine is just waht was in hand, choosing
to use it as a belief SL target is the business of hte client". Engine docs now
describe authoritative hand contents only; fields and semantics are unchanged.
Verification above records the pre-rename API.

Post-rename verification: `cargo test --manifest-path managym/Cargo.toml -p
managym --lib hand_snapshots -j 2 -- --test-threads 2` — 3 passed in debug.
The release wheel rebuilt; the local extension was replaced atomically.
`uv run --no-project --python <venv>/bin/python -m pytest --confcutdir=.
test_hand_snapshots.py -q` against the temporary installed wheel — 8 passed in
1.55 s. The exact guide example ran; native/wrapper imports came from the
installation, both import paths and packaged stubs exposed the new API, and
old class/accessor names were absent. This again reuses development dependencies.
Ruff, Rust formatting and diff whitespace checks passed. Temporary installation
removed; no background work remains. All deletion targets above are complete.
