# Scalar and batched Python APIs

Audit for MTG-272, 2026-11-11. This compares the native implementations, PyO3
bindings, installed stubs and Python wrappers. The callable example
and snapshot contract are in [Hand snapshots](engine.md#hand-snapshots).

| Capability | Scalar native `managym.Env` | Native `managym.VectorEnv` | Python wrappers and disposition |
|---|---|---|---|
| Construction and setup | Seed, trivial skipping, optional profiling/behavior tracking; explicit two-player reset | Fixed slot count, seed stride, built-in none/passive/random opponent; uniform or distinct deck sampling; per-slot buffered setups | Gym `Env` takes Match/ObservationSpace/Reward. Torch vector takes those plus device; uses CapturedVector. Different orchestration, same native rules. |
| Perspective | Reset/step return chooser view; `observation_for_player` reads a fixed native seat | Reset/step return each chooser view; built-in opponents advance to seat 0 | `current_agent_indices` reports choosers, including control effects. No general fixed-view vector API; hand snapshots supply authoritative contents without materializing observations. |
| Legal choices | Complete action list, positional steps, revision-bound Commands and captured-command submission | Positional indices; readable results have complete actions; buffered observations may select explicit `first-legal` prefix | No vector Command execution or capture-origin authorization. Intentional collection API difference; use scalar for external interactive submissions. Neither snapshot reads nor tensor cutoffs change native legality. |
| Observation encoding | Allocating native encode or caller-owned buffers, plus Python reference encoder | Bound NumPy buffers derive capacities/version from supplied arrays; readable mode bypasses tensor capacities | Both use the native encoder; Python `ObservationSpace` supplies allocations. No new observation fields or schema versions for snapshots. |
| Step and reward | One surfaced decision, terminal observation, reward from decision chooser | Same with opponent policy none; built-in policy plays opponent decisions and negates an opponent terminal reward for seat 0 | Gym and Torch can apply reward shaping. Torch returns reused tensors and empty step info; `get_last_info` reads native diagnostics separately. |
| End/reset | No implicit native reset; further step fails after game over | Terminal reward/flags/info, but replacement-game observation; seeds advance by stride | Gym `auto_reset` clears returned done flags and retains `info.true_terminated/true_truncated`; semantic stepping does not auto-reset. Torch keeps native done flags. These contracts are not interchangeable. |
| Partial execution | Caller chooses whether to step | Buffered active masks; inactive observations unchanged, reward/done cleared by Python bindings | CapturedVector records only active attempted steps; Torch step has no mask. Hand snapshots always return all slots; collector applies its mask. |
| Seeds and identity | `set_seed` affects the next reset; state digest and decision revision are distinct identities | Current seeds, seed restoration, root reassignment and failed-slot replacement | HandSnapshot adds opaque reset generation with seed/revision. A repeated seed or a Torch native-vector rebuild cannot reuse its game generation. Generations are not frozen replay identities. |
| Errors and capacities | Invalid action rejects; later rules/encoder failure need not be transactional | Strict calls may advance a prefix before failing; isolated gameplay/capacity outcomes are opt-in; malformed buffers remain fatal | CapturedVector retains private replay inputs around buffered calls. Discard failed-call rows and recover/reset; neither an old buffer nor an empty snapshot denotes a successful transition. |
| Capture and diagnostics | Viewer decision captures, Commands, state/selected-search witnesses, optional card/exposure capture and profiler | Semantic decision frames, current setups/seeds/action summaries, exposure and buffered timing counters | Vector frames are not scalar DecisionCapture authorization. Snapshots are separate privileged records; no automatic collection or addition to capture/diagnostic payloads. |
| Search and possible worlds | Fork/selected fork, determinization, fixed-view spaces/materialization, MC/PUCT, RolloutPool | No arbitrary slot fork or possible-world operations | RolloutPool batches simulations from one scalar root; it is not VectorEnv. Closed-decklist search admission remains native. No parity shim added. |
| Installation and typing | `from manabot import managym`; legacy `import managym` aliases the same classes | Same installed extension and stub, now including HandSnapshot | Explicit typed methods on CapturedVector and Torch. Added missing vector stubs for existing roster, matchup and exposure methods. Release/version/dependency changes remain separate. |

## Code evidence

- [`agent/env.rs`](../managym/src/agent/env.rs): `reset`, `step`,
  `observation_for_player`, `capture_decision`, `selected_revision`, `hand_counts`,
  `fork` and possible-world/search methods own single-game semantics.
- [`agent/vector_env.rs`](../managym/src/agent/vector_env.rs): `reset_each_into`,
  `reassign_roots_into`, `step_with_autoreset_on_env`, `step_one_on_env`,
  `step_isolated_into`, `replace_failed_into` and `hand_snapshots` own the batch.
  `for_each_env` currently visits slots sequentially, not in a thread pool.
- [`python/bindings.rs`](../managym/src/python/bindings.rs) and
  [`python/vector_env_bindings.rs`](../managym/src/python/vector_env_bindings.rs):
  scalar lock/Command checks; vector active-state clearing, buffer validation,
  GIL release around buffered calls, and Python result conversion.
- [`__init__.pyi`](../manabot/managym/__init__.pyi) is the installed type surface;
  [`managym/__init__.py`](../managym/__init__.py) owns the compatibility alias.
- [`env.py`](../manabot/env/env.py),
  [`vector_env.py`](../manabot/env/vector_env.py), and
  [`failures.py`](../manabot/env/failures.py): wrapper rewards, reset policies,
  tensor reuse and private failure evidence. `CapturedVector.__getattr__`
  delegates unwrapped methods; its instrumentation is specifically buffered.
- [`belief/audit.py`](../manabot/belief/audit.py): `authority_hand` establishes
  the existing name-count representation used by a belief-model client. It
  remains scalar.

## Bounded corrections and follow-ups

Implemented: complete batched hand snapshots, typed wrapper access, identity across
reset/replacement, missing vector method stubs, and corrected the guide's claim
that inactive Python reward/done buffers retain old values. The actual bindings
clear those fields; only observations remain untouched.

Larger independent follow-ups, not new parity requirements:

- **CapturedVector readable calls bypass capture bookkeeping.** Its inherited
  `reset_all`/`step` delegate directly to native; mixing them with instrumented
  buffered calls leaves `games` replay prefixes stale. Either explicitly reject
  readable mutation on this wrapper or instrument it end to end. Acceptance:
  a mixed-mode failure replays the exact native seed/setup/action prefix.
- **Same-seed capture rollover.** `CapturedVector._synchronize` detects automatic
  replacement through seed inequality; native seed succession saturates at
  `u64::MAX`. A completed game at that boundary can leave the old action prefix
  in private evidence. Use reset-generation rollover for capture bookkeeping or
  reject exhausted seed succession; acceptance must cover terminal replacement
  at the maximum seed. HandSnapshot already distinguishes these games.
- **Gym/Torch reset contract alignment.** Gym auto-reset suppresses done flags,
  while vector preserves them; Torch `reset(seed=current_base)` advances native
  succession rather than returning to that seed. Decide compatibility before
  changing either public behavior. Pin the selected behavior in wrapper tests.

No claims of throughput improvement, terminal-hand retention, GPU validation or
full scalar-helper parity follow from this change. Hand reads perform work only
on request; ordinary steps do not count or export hands.

## Focused verification

Three native debug tests cover complete/empty hands, unchanged native-seat
ownership under player control, and identity across repeated-seed resets,
reassignment and replacement. Eight Python cases pass against a wheel installed
outside the checkout: scalar parity with open/closed decklists, detached snapshots,
invalid requests, masks, failure replacement, autoreset, Torch rebuilds and an
empty hand. The guide's example runs against that installation; both import
paths and the packaged stubs expose HandSnapshot. The existing same-seed stale
capture regression also passes. Full affected suites remain gate's work.
