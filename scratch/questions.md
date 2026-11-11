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

Gate still needs the complete debug Rust suite, Clippy and affected Python
environment/capture/package checks under `docs/development.md#validation-by-change-area`.
Relevant Python coverage includes `tests/env/`,
`tests/semantic/test_text_observation.py`,
`tests/training/test_automatic_game_capture.py`,
`tests/training/test_game_errors.py`, `tests/training/test_game_error_recovery.py`
and `tests/test_package_boundaries.py`. No world or observation shape changed.
Publication and Task movement belong to delivery, after validation.

Audit follow-ups, decided by Jack Heart (steer 14d5db38, exact reply:
"1 A / 2 A / 3a - lets discuss further / 3b A"; then steer 3437c194 on 3a:
"ok, remove auto reset"). All four are implemented in this PR (QA pass,
2026-11-11):

- 1A: `CapturedVector.__getattr__` delegates only `_VECTOR_READS`; `reset_all`,
  `step`, `take_exposure`, `enable_exposure_capture` and
  `scenario_naming_vocabulary_into_buffers` raise `AttributeError` naming the
  buffered calls. Unknown names raise a plain `AttributeError`.
- 2A: native `VectorEnv.current_game_generations()` (binding + stub);
  `_synchronize` restarts prefixes on generation change, not seed inequality.
- 3bA: Torch `reset(seed=s)` rebuilds at `s` for every explicit seed.
- 3a: Gym `Env(auto_reset=...)` removed (TypeError), with the unread
  `true_terminated`/`true_truncated` info keys. All in-repo callers updated.

Release-notes items (breaking, for the 0.2.1 notes MTG-271's workflow writes):
Gym `auto_reset` removed; Torch `reset(seed=current_base)` now reproduces the
deal; CapturedVector rejects readable `reset_all`/`step`. There is no
`docs/releases.md` on main; the guide (`docs/engine.md`) documents all three.

No unresolved product decision blocks the hand snapshot contract. There is no
claim of throughput improvement, GPU validation, terminal-hand retention or full
scalar-helper parity. On 2026-11-11, `lf task status MTG-271 --json` reported its
Python/dependency work at realign with no PR; its implementation was not imported
or edited. This Task has no Wave memory to curate.

## Concept review — 2026-11-11

Outcome for loop-or-next: **nothing to change and no open product choice for
MTG-272; advance.** This is a design review, not gate completion. The naming and
consumer boundary in Jack Heart's two quoted directions above are already built.
Keep the current working design; no next implement direction is needed.

The experience is one explicit read: choose a native seat for each batch slot,
call `hand_snapshots`, and retain detached name counts with their game/revision
identity. Native zones own the contents; bindings and both wrappers expose the
same record. The client owns interpretation and persistence. No new observation,
capture, replay schema or automatic collection lifecycle is needed. The current
engine usage example already expresses this experience and needs no rewrite.

The principal trap is associating terminal reward/done with a hand read after
autoreset. That read belongs to the replacement observation. Preserve the
documented before-step read for current hands; retaining terminal hands would be
a separate capability. Same-seed reset is the counterexample to seed-only
identity; process-wide generation distinguishes it, while run identity remains
necessary across processes. Player control is the counterexample to treating a
seat as the chooser; explicit native seats preserve the intended ownership.

For failure/recovery, a strict call can advance a prefix: discard its transition
results, recover/reset, then read the new boundary. An isolated failed slot is
replaced before another read; inactive slots retain their game/revision. Earlier
detached records survive as records of earlier states, never proof of a failed
transition. This is a caller contract, not a promise that snapshot reads reject
every unrecovered failure. The three audit follow-ups above remain separate;
their compatibility choices do not block this read-only API.

Next show `scratch/concept-review.html`, then the gate results already listed
above. This pass inspected source and existing fixtures; it ran no runtime
tests and makes no additional validation claim.

## Retained validation

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

Realign changed only this working plan. Existing results were reused; no runtime
tests were repeated. Source inspection confirmed the rename, hidden-information
boundary and the three follow-ups; no additional code correction was needed.
