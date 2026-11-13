# Scalar and batched Python APIs

Audit for MTG-272, 2026-11-11. This compares the native implementations, PyO3
bindings, installed stubs and Python wrappers. The callable example
and snapshot contract are in [Hand snapshots](engine.md#hand-snapshots).

| Capability | Scalar native `managym.Env` | Native `managym.VectorEnv` | Python wrappers and disposition |
|---|---|---|---|
| Construction and setup | Seed, trivial skipping, optional profiling/behavior tracking; explicit two-player reset | Fixed slot count, seed stride, built-in none/random opponent; uniform or distinct deck sampling; per-slot buffered setups | Gym `Env` takes Match/ObservationSpace/Reward. Torch vector takes those plus device; uses CapturedVector. Different orchestration, same native rules. |
| Perspective | Reset/step return chooser view; `observation_for_player` reads a fixed native seat; `hand_snapshot` reads authoritative hand counts | Reset/step return each chooser view; built-in opponents advance to seat 0 | `current_agent_indices` reports choosers, including control effects. No general fixed-view vector API; hand snapshots supply authoritative contents without materializing observations. |
| Legal choices | Complete action list, positional steps, revision-bound Commands and captured-command submission | Positional indices; readable results have complete actions; buffered observations may select explicit `first-legal` prefix | No vector Command execution or capture-origin authorization. Intentional collection API difference; use scalar for external interactive submissions. Neither snapshot reads nor tensor cutoffs change native legality. |
| Observation encoding | Allocating native encode or caller-owned buffers, plus Python reference encoder | Bound NumPy buffers derive capacities/version from supplied arrays; readable mode bypasses tensor capacities | Both use the native encoder; Python `ObservationSpace` supplies allocations. No new observation fields or schema versions for snapshots. |
| Step and reward | One surfaced decision, terminal observation, reward from decision chooser | Same with opponent policy none; built-in policy plays opponent decisions and negates an opponent terminal reward for seat 0 | Gym and Torch can apply reward shaping. Torch returns reused tensors and empty step info; `get_last_info` reads native diagnostics separately. |
| End/reset | No implicit native reset; further step fails after game over | Terminal reward/flags/info, but replacement-game observation; seeds advance by stride | Gym `Env` never auto-resets: terminal steps return true done flags and the caller resets (its `auto_reset` argument is removed). Torch keeps native vector autoreset with true done flags, the only automatic-reset contract. |
| Partial execution | Caller chooses whether to step | Buffered active masks; inactive observations unchanged, reward/done cleared by Python bindings | CapturedVector records only active attempted steps; Torch step has no mask. Hand snapshots always return all slots; collector applies its mask. |
| Seeds and identity | `set_seed` affects the next reset; state digest and decision revision are distinct identities | Current seeds, seed restoration, root reassignment and failed-slot replacement | HandSnapshot retains the originating reset seed and an opaque generation distinct across resets, replacements, forks and materialized worlds. Generations are not frozen replay identities. |
| Errors and capacities | Invalid action rejects; later rules/encoder failure need not be transactional | Strict calls may advance a prefix before failing; isolated gameplay/capacity outcomes are opt-in; malformed buffers remain fatal | CapturedVector retains private replay inputs around buffered calls. Discard failed-call rows and recover/reset; neither an old buffer nor an empty snapshot denotes a successful transition. |
| Capture and diagnostics | Viewer decision captures, Commands, state/selected-search witnesses, optional card/exposure capture and profiler | Semantic decision frames, current setups/seeds/action summaries, exposure and buffered timing counters | Vector frames are not scalar DecisionCapture authorization. Snapshots are separate privileged records; no automatic collection or addition to capture/diagnostic payloads. |
| Search and possible worlds | Fork/selected fork, determinization, fixed-view spaces/materialization, MC/PUCT, RolloutPool | No arbitrary slot fork or possible-world operations | RolloutPool batches simulations from one scalar root; it is not VectorEnv. Closed-decklist search admission remains native. No parity shim added. |
| Installation and typing | `import managym` from the engine distribution; `from manabot import managym` is the compatibility alias | Same installed extension and stub, including HandSnapshot | Explicit typed methods on CapturedVector and Torch. Engine-only consumers need no training package; see [installation](engine-python.md). |

## Code evidence

- [`agent/env.rs`](../managym/src/agent/env.rs): `reset`, `step`,
  `observation_for_player`, `capture_decision`, `selected_revision`, `hand_snapshot`, `hand_counts`,
  `fork` and possible-world/search methods own single-game semantics.
- [`agent/vector_env.rs`](../managym/src/agent/vector_env.rs): `reset_each_into`,
  `reassign_roots_into`, `step_with_autoreset_on_env`, `step_one_on_env`,
  `step_isolated_into`, `replace_failed_into` and `hand_snapshots` own the batch.
  `for_each_env` currently visits slots sequentially, not in a thread pool.
- [`python/bindings.rs`](../managym/src/python/bindings.rs) and
  [`python/vector_env_bindings.rs`](../managym/src/python/vector_env_bindings.rs):
  scalar lock/Command checks; vector active-state clearing, buffer validation,
  GIL release around buffered calls, and Python result conversion.
- [`__init__.pyi`](../managym/__init__.pyi) is the installed type surface;
  [`manabot/managym/__init__.py`](../manabot/managym/__init__.py) owns the compatibility alias.
- [`env.py`](../manabot/env/env.py),
  [`vector_env.py`](../manabot/env/vector_env.py), and
  [`failures.py`](../manabot/env/failures.py): wrapper rewards, reset policies,
  tensor reuse and private failure evidence. `CapturedVector.__getattr__`
  delegates an explicit read-only allowlist; its instrumentation is buffered.
- [`belief/audit.py`](../manabot/belief/audit.py): `authority_hand` establishes
  the existing name-count representation used by a belief-model client. It
  now reads the shared scalar snapshot contract, as do sampler collection and
  materialized-world supervision.

## Bounded corrections and follow-ups

Implemented: complete scalar and batched hand contents with one record (batch
position comes from the list index), typed wrapper access, identity across
reset/replacement, missing vector method stubs, and corrected the guide's claim
that inactive Python reward/done buffers retain old values. The actual bindings
clear those fields; only observations remain untouched. Five audit findings
are also fixed:

- **CapturedVector readable calls bypassed capture bookkeeping.** Its
  `__getattr__` now delegates only native reads; `reset_all`, `step` and other
  native mutators raise `AttributeError` naming the buffered calls.
- **Same-seed capture rollover.** Native `current_game_generations()` exposes
  the reset generation; `CapturedVector._synchronize` starts a new prefix when
  it changes, covering terminal replacement at `u64::MAX` and same-seed resets.
  Generations stay process-local bookkeeping, never serialized evidence.
- **Torch explicit-seed reset.** `reset(seed=s)` now restarts at `s` for every
  explicit seed, including the current base; previously a reset to the current
  base advanced succession. `reset()` without a seed still advances.

- **Gym/vector done-flag contract.** Gym auto-reset suppressed done flags
  while vector preserved them. The Gym `auto_reset` argument and its step-time
  reset are removed (breaking: passing it raises `TypeError`), along with the
  `true_terminated` / `true_truncated` info keys nothing read. Vector autoreset
  with true done flags is the only automatic-reset contract.

- **Built-in opponents.** Removed native `Passive`, its Python policy and the
  passive decision-profile CLI branch. Vector construction (native, capture and
  Torch) now accepts only `none` or `random`; `passive` raises an unsupported-policy
  error. Capture retains the selected supported policy and replay uses the same
  native admission. Historical receipts keep their original policy strings;
  passive receipts require the original source to execute. Academy's independent
  scalar debugging opponent remains outside this vector API change.

Scalar identity also covers the full branch lifecycle. `Env` retains the live
reset seed separately from the next reset's configured seed, so `set_seed(99)`
after a reset at 42 leaves the snapshot at 42 until reset succeeds. Exact forks,
selected forks, sampled hands, indexed worlds and prepared materializations
share one branch constructor that allocates a distinct process-local generation.
They inherit the originating reset seed, independently of materialization RNG
seeds. Capture-origin checks and reset invalidation remain intact; portable
state, replay and world identities exclude this bookkeeping.

Separate follow-ups: **MTG-277** adds a native heuristic opponent; **MTG-276**
adds fixed-seat batched observations; **MTG-275** covers batched fork/search.
None is implemented in this PR.

No claims of throughput improvement, terminal-hand retention, GPU validation or
full scalar-helper parity follow from this change. Hand reads perform work only
on request; ordinary steps do not count or export hands.

## Focused verification

Native debug and installed-wheel snapshot cases cover complete counts,
native-seat parity, terminal/reset boundaries, pending seeds, independent
branches and capture-origin/reset rejection. The three migrated belief clients
have focused coverage. Capture bookkeeping tests cover rejected mutations,
delegated reads, maximum-seed replacement and both Torch reset forms. Opponent
tests cover rejection at each public boundary and retained passive receipts.

The full debug Rust suite passes after scalar unification and identity repairs
(1,057 passed, two skipped). Fresh macOS arm64 consumers pass 26 snapshot,
capture-bookkeeping and Command cases each, the guide example, packaged-stub
checks, and the installed training/checkpoint smoke: CPython 3.12.12 with Torch
2.10.0, and CPython 3.13.7 with Torch 2.15.0, both with Etude 0.2.15. These use
release-prepared wheels with their source manifests and independently installed
dependencies, outside the checkout.

The affected Python suite reports 343 passed and three retained TLA replay
`binding_mismatch` failures, previously reproduced with base `402dfd5b2`.
Recovery and search checks pass against the merged outcome-table path. The
earlier intermittent Etude `seal_logs` missing-file failure did not recur;
this does not establish a fix. Frozen evidence remains unchanged. Linux and
accelerator execution remain outside these local checks.
