# MTG-272 hand snapshots and API audit — 2026-11-11

## Outcome and direction

Ryan's request: "one more request: can you add true hands in batched mode?"
The originating clarification: "yeah, he is training a belief model."
The outcome is an opt-in authoritative hand read across native VectorEnv,
CapturedVector and Torch VectorEnv, plus a code-evidenced scalar/batch API audit.

Jack Heart, comment c1389729: "assumign its the input to a belief model seems unnecessary, shouldnt wejust label it as a HandInfo or smoething".
The steer selects `HandSnapshot` / `hand_snapshots(players)`.
Follow-up d793c7d4: "the output of the engine is just waht was in hand, choosing to use it as a belief SL target is the business of hte client".
Engine-facing documentation describes authoritative hand contents at the read;
client uses do not define the API.

Preserve native-seat ownership, complete duplicate counts and explicit snapshot
timing/identity. Hidden hand contents stay outside viewer observations, policy
inputs, shared captures and viewer exports. No frozen layouts, world/checkpoint
identities, deck disclosure or legal actions change. MTG-271 owns versions,
dependencies and release workflow; this work supplies neither a release nor
isolated dependency-resolution proof.

## Implemented approach

- `managym/src/agent/env.rs::hand_counts` reads native hand zones and validates
  seats. `managym/src/agent/vector_env.rs::hand_snapshots` returns detached typed
  records in slot order with native seat, game generation, seed, revision and
  name counts. One native batch call does the work only when requested;
  capacities do not truncate hands. PyO3, installed stubs and both wrappers
  expose the same API.
- The existing capture generation now uses process-wide allocation in
  `Env::reset`, distinguishing repeated-seed resets, slot replacement and Torch
  native-vector rebuilds. It is an opaque lifetime token, not a portable replay
  identity or consecutive counter. Capture origin and stale checks still apply;
  generations are not serialized into public captures.
- `docs/engine.md#hand-snapshots` documents the contract and runnable example.
  Reads after autoreset describe the replacement game; terminal hands are not
  retained. Reads include inactive slots; collectors apply their active mask.
  Failed-call results must be discarded before recovery/reset.
- `docs/python-api-audit.md` covers native implementations, bindings, stubs and
  wrappers. Added stubs also describe existing roster, matchup and exposure
  methods. The guide's original inactive-buffer assumption was wrong: Python
  bindings clear reward/done fields while preserving observation rows. The guide
  and regression fixture now reflect that behavior.

The prerelease `HandTarget` / `hand_targets` names and their old test filename
are removed without aliases. Earlier implementation notes and pre-rename proof
are retained at `64779909e75041ffa9f35b4bc532a9240b688c55:scratch/questions.md`.
Unrelated b015d5/C2 experiment notes remain in
`757bb6257ef07cf2536ad7f7ec24f0f7309628c6:scratch/questions.md` and its linked evidence.

## Remaining work and limits

All accepted implementation cuts are present, including Passive removal. The
earlier iteration feedback saying Passive remains predates commit `4dc9574bf`.
Gate still needs the complete debug Rust suite and affected Python
environment/capture/package checks under `docs/development.md#validation-by-change-area`.
All-target/all-feature Clippy passed after the opponent cut and compression;
no native changes since then invalidate that result.
Relevant Python coverage includes `tests/env/`,
`tests/semantic/test_text_observation.py`,
`tests/training/test_automatic_game_capture.py`,
`tests/training/test_game_errors.py`, `tests/training/test_game_error_recovery.py`
and `tests/test_package_boundaries.py`. No world or observation shape changed.
Publication and Task movement belong to delivery, after validation.

The merge at `7de8b87dc` incorporates upstream `402dfd5b`, including MPS training
support from #457. It leaves the native snapshot/opponent code unchanged.
`manabot/env/vector_env.py::hand_snapshots` still returns CPU records, independent
of the tensor device. Existing CPU proof does not establish CUDA/MPS validation;
no training-device support is added by this Task.

Resolve the recorded QA failures before claiming gate success:

- Two retained TLA resource-limit replay cases returned `binding_mismatch` at
  action 370, also reproduced on `61a51a3ff` in the earlier QA pass. Preserve the
  frozen evidence and determine its disposition; do not rewrite it to pass.
- The exact-quota recovery test used removed `RolloutBatch.obs`; the current
  interface is `observations().observation` in
  `manabot/training/rollout/collection.py`. Its two assertions are corrected and
  the focused fixture passed on 2026-11-11 (command below).
- The conditional-search fixture lacked
  `experiments/data/int-13-conditional-strategy-fixture-v1.json`; obtain the
  retained artifact for gate or record the unavailable check explicitly.

`scratch/qa.html` records the earlier run (695 passed, four failures), not the
current gate status. Only the replay mismatch has recorded baseline reproduction;
the other diagnoses come from source inspection or the missing artifact.

Audit follow-ups, decided by Jack Heart (steer 14d5db38, exact reply:
"1 A / 2 A / 3a - lets discuss further / 3b A"; then steer 3437c194 on 3a:
"ok, remove auto reset"). All four are implemented in this PR (QA pass,
2026-11-11):

- 1A: `CapturedVector.__getattr__` delegates only `_VECTOR_READS`; `reset_all`,
  `step`, `take_exposure`, `enable_exposure_capture` and
  `scenario_naming_vocabulary_into_buffers` raise `AttributeError` naming the
  buffered calls. Every undelegated name gets the same guidance; there is no
  separate mutation denylist to maintain.
- 2A: native `VectorEnv.current_game_generations()` (binding + stub);
  `_synchronize` restarts prefixes on generation change, not seed inequality.
- 3bA: Torch `reset(seed=s)` rebuilds at `s` for every explicit seed.
- 3a: Gym `Env(auto_reset=...)` removed (TypeError), with the unread
  `true_terminated`/`true_truncated` info keys. All in-repo callers updated.

Release-notes items (breaking, for the 0.2.1 notes MTG-271's workflow writes):
Gym `auto_reset` removed; Torch `reset(seed=current_base)` now reproduces the
deal; CapturedVector rejects readable `reset_all`/`step`; built-in `passive`
opponents and the passive decision-profile CLI branch are removed. There is no
`docs/releases.md` on main; the guide (`docs/engine.md`) documents these changes.

No unresolved product decision blocks the hand snapshot contract. There is no
claim of throughput improvement, GPU validation, terminal-hand retention or full
scalar-helper parity. On 2026-11-11, `lf task status MTG-271 --json` reported its
Python/dependency work at realign with no PR; its implementation was not imported
or edited. This Task has no Wave memory to curate.

The pre-opponent-cut concept review and its rationale are retained at
`36d9451ef844fe23d59743f32dbdfaa121ab501e:scratch/questions.md` and
`36d9451ef844fe23d59743f32dbdfaa121ab501e:scratch/concept-review.html`.
The current review page includes the subsequent reset, capture and opponent cuts.

## Concept review after realignment — 2026-11-11

Outcome for loop-or-next: **advance**. No design change or open product choice.
The implementation and usage guide express the recorded direction above;
this review changes only the working notes and `scratch/concept-review.html`.
It does not establish gate success or rerun the retained runtime proofs.

The experience remains: request native seats, read detached `HandSnapshot`
records, and let clients decide their use and storage. A snapshot captures one
read boundary; the game generation distinguishes resets, while revision identifies
the decision boundary. Native zones own the contents, bindings and wrappers
carry the same record, and no observation or persisted replay schema changes.
CapturedVector owns its buffered mutations and restarts evidence prefixes from
native generations. Gym callers reset explicitly; vector collectors retain true
done flags with replacement-game observations. Vector opponents are caller-driven
(None) or native Random. No further lifecycle abstraction is needed for this scope.

Counterexamples retained: an equal seed (including saturated succession) does
not identify the same game; a chooser is not necessarily the requested hand's
native seat; inactive rows still appear in an all-slot read; and a post-done
read cannot recover terminal hands. On strict-call failure, discard that call's
transition results and recover/reset before another read; isolated failures
require failed-slot replacement. Earlier detached snapshots remain historical
reads, not proof that a failed transition succeeded. Terminal retention stays
outside this design; MTG-275/276/277 remain separate work.

Next show gate's disposition of the two retained replay mismatches and missing
conditional-search artifact, plus the required debug Rust and affected Python
checks. These are validation obligations, not unanswered product questions.

## Retained validation

After realignment, `uv run --extra dev --extra artifacts pytest
tests/training/test_game_errors.py::test_exact_quota_keeps_learned_rows_and_refills_after_a_capacity_failure
-q` — 1 passed in 2.56 s. Ruff check/format and `git diff --check` passed.
This repairs one of the four earlier QA failures; the other three and full gate
remain outstanding. Existing native and installed-package results were reused.

Post-rename proof from the preceding implementation pass:

- `cargo test --manifest-path managym/Cargo.toml -p managym --lib hand_snapshots -j 2 -- --test-threads 2`
  — 3 passed in debug: completeness/empty hands, native seats under control and
  identity across same-seed reset, reassignment and replacement.
- The release wheel was rebuilt and the local extension replaced atomically.
  `uv run --no-project --python <venv>/bin/python -m pytest --confcutdir=. test_hand_snapshots.py -q`
  against a temporary installation outside the checkout — 8 passed in 1.55 s.
  Native and wrapper imports came from that installation. Both import paths and
  packaged stubs exposed the new API; old names were absent. The guide example
  ran with seven cards in both snapshots. Development dependencies were reused;
  this proves API packaging, not clean dependency resolution or release readiness.
- The existing same-seed stale-capture regression passed before the rename.
  Ruff, Rust formatting and diff whitespace checks passed after the rename.
  Temporary installation files were removed.

## Accepted opponent cut — 2026-11-11

Jack Heart (5fde3453): "do C in this chagne, A in a nother diff";
"not worried about outside users yet". Native built-in opponents become
None/Random. MTG-277 owns the native heuristic opponent; MTG-276 owns fixed-seat
batched observations; MTG-275 owns batched fork/search. None is implemented here.

Delete — do not maintain: OpponentPolicy::Passive, its binding parser arm,
PassivePolicy and its exclusive tests, passive vector fixtures, and the
untrained-vs-passive decision-profile CLI branch that consumes PassivePolicy;
also its exclusive native `Env::pass_priority_action_index` helper. All are
removed. No deletion targets remain.
Preserve random automated turns, caller-driven None, masks/autoreset, exact
capture/replay inputs and historical evidence. Retained policy strings remain
readable; attempting to replay passive rejects through native admission.
Academy's independent scalar debugging opponent is outside this batched cut.
The decision-profile CLI already offers untrained-vs-random; remove its passive
branch rather than relabeling the historical baseline or duplicating random.

The cut is implemented, including the stub's None/Random literal values, native
admission through capture/replay and purpose-matched fixture replacements.
The release wheel was rebuilt and the cp312 extension installed atomically.

- `cargo test --manifest-path managym/Cargo.toml -p managym --lib agent::vector_env::tests -j 2 -- --test-threads 2` — 20 passed in debug.
- Focused `uv run --extra dev --extra artifacts pytest` nodes in
  `tests/env/test_opponents.py`, `test_env.py::TestEnvironment::test_vectorenv_single_agent_mode`,
  `tests/verify/test_observation_parity.py::{test_vector_env_buffer_hot_path_matches_compatibility_step_path,test_vector_env_set_buffers_validates_contract}`,
  `tests/training/test_automatic_game_capture.py::test_ordinary_vector_replays_current_game_after_autoreset`
  and `tests/model/test_agent.py::test_forward_pass_shapes` — 16 passed;
  the new retained-evidence fixture needed `by_alias=True` to match persisted
  JSON, then its focused rerun passed (17 cases total).
- Rust formatting, Ruff check/format on all changed Python files, diff whitespace
  and decision-profile CLI with both game counts zero — passed.

No heuristic, fixed-seat observations or search follow-up has been implemented.

## Compression — 2026-11-11

Native generations now own all prefix rollover: initialize the retained list
from native construction and compare it after each call, without a second reset
flag or first-call sentinel. Pre-reset evidence still retains attempted inputs
before entering native code. Torch no longer stores the unread `_base_seed`, and
native construction owns its supported-policy validation. HandSnapshot repr uses
the map's formatter directly. The README's obsolete Gym auto-reset claim is fixed.

- `cargo test --manifest-path managym/Cargo.toml -p managym --lib agent::vector_env::tests -j 2 -- --test-threads 2` — 20 debug tests passed; release wheel rebuilt and extension replaced atomically.
- `uv run --extra dev --extra artifacts pytest tests/env/test_captured_vector_bookkeeping.py tests/env/test_hand_snapshots.py tests/env/test_opponents.py tests/training/test_automatic_game_capture.py tests/training/test_training_exposure.py -q` — 64 passed; one exposure substitute lacked the new generation read. Added that method; its focused rerun passed (65 cases total).
- `PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --manifest-path managym/Cargo.toml -p managym --all-targets --all-features -j 2 -- -D warnings`, Rust formatting, changed-file Ruff check/format and `git diff --check` — passed. Full gate and the retained QA failure dispositions remain outstanding.

## QA pass — 2026-11-11 (see scratch/qa.html)

Fresh /tmp venv install of the rebuilt wheel ran the guide example, a native
belief-style collector (1600 cross-checks against own-view hands and public hand
counts) and a Torch loop with seeded reset. All passed. Fixed: reading snapshots
before reset now says "must be reset before reading hand snapshots" (was "...before
step"); debug `hand_snapshots` Rust tests passed, extension replaced atomically.

Dispositions: exact-quota test passes; conditional-search fixture is now present
locally and passes (16). The TLA receipt replay `binding_mismatch` fails all 3
parametrizations identically on base 402dfd5b2 (built in a temp worktree), so it
is pre-existing. Neither test file runs in any CI workflow. Affected suite: 424
passed, 3 failed (that pre-existing failure). Full debug Rust suite was not rerun
after the one-line error-message change. Torch is not a wheel dependency
(dependency group only); MTG-271 owns that.
