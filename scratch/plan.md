# Plan: remove eager auto-reset and unify slot loading (Task a2da47)

Concept review record: `scratch/concept-review.html`.

## Outcome

Every vector has one explicit slot lifecycle and one transactional slot loader:

- `step`, `step_into_buffers`, and `step_isolated_into_buffers` report a
  finished game's terminal observation, reward, flags, and winner once and park
  the slot in `Terminal` (or `Poisoned` on an isolated failure), reporting
  chooser `-1`. Nothing constructs a game inside a step.
- `load_slots_into_buffers(requests)` is the single sparse, atomic slot loader
  across `managym.VectorEnv`, `RecordedVectorEnv`, `BufferedStreams`,
  `ShardedVectorEnv`, and `TargetPracticeVector`:
  - `SlotLoad.fresh(slot, player_configs=None)` installs a fresh game using the
    slot's `next_seeds[slot]` (advancing by `stride`) and, when `player_configs`
    is given, updates `player_configs[slot]` atomically on commit (rejected if a
    roster is configured). Works uniformly on `FreshOpening`, `Playing`,
    `Terminal`, and `Poisoned` slots.
  - `SlotLoad.snapshot(slot, snapshot, options=None)` restores a detached
    branch without spending a fresh seed (rewinding `next_seeds[slot]` and
    discarding untouched recording when replacing `FreshOpening`).
- `auto_reset`, `with_auto_reset`, `replace_failed_into*`, and
  `reassign_roots_into*` are deleted everywhere.
- `open_native_streams` always returns `BufferedStreams` (single-shard or
  `ShardedVectorEnv`), and `Collector` uses one lazy batch-bounded refill model
  with zero wasted `Game::construct` on batch-end terminals.

Who notices: callers passing `auto_reset=...` get `TypeError`; callers calling
`replace_failed_into_buffers` or `reassign_roots_into_buffers` get
`AttributeError` and move to `load_slots_into_buffers([SlotLoad.fresh(...)])`;
loops stepping a finished slot without refilling get the existing
`"cannot step Terminal slot N before reset or replacement"` error. Pre-1.0, so
no deprecated shims.

## Choices made for review

1. **Fold `reassign_roots_into*` into `SlotLoad.fresh(slot, player_configs=None)`
   and delete `reassign_roots_into*`.**
   *Passed over:* Patching `reassign_roots_into` to accept parked slots.
   *Why:* Patching `reassign_roots_into` fails three concrete counterexamples:
   (a) mixed `FreshOpening` vs `Terminal`/`Poisoned` gives one method two
   opposite seed rules (rewind vs advance); (b) `RecordedVectorEnv.reassign_roots_into_buffers`
   reads `current_game_seeds()` before native runs, which is the *old* finished
   game's seed on a parked slot (corrupting journal proposals and `mixed`
   disclosure retention); (c) `action_limit` under `game_errors` leaves a slot
   `Playing`, which `reassign_roots_into` rejects (`"cannot reassign a live game"`).
   Folding into `SlotLoad.fresh(slot, player_configs=None)` removes a duplicate
   transactional loader, preserves exact adaptive seed parity (one construction
   instead of two), and brings uniform buffer validation, typed
   `ObservationCapacity`, `SlotLoadResult`, and sharded support.
2. **Expose `load_slots_into_buffers` on `BufferedStreams` / `ShardedVectorEnv`
   and make `open_native_streams` always return `BufferedStreams`.**
   *Passed over:* Returning bare `RecordedVectorEnv` for `workers == 1` and
   leaving `BufferedStreams` without `load_slots_into_buffers`.
   *Why:* One uniform transport type and slot-loading method across 1-worker and
   N-worker native streams (`BufferedStreams.install_slots` delegates to
   `load_slots_into_buffers`; `bench_sharded_collection.py` uses it directly).
   Keep the existing `workers != 1` check on adaptive collection in
   `open_collector` out of scope for this PR.
3. **Convert `TargetPracticeVector` to park terminal rows and implement
   `load_slots_into_buffers`.**
   *Passed over:* Keeping eager auto-reset in `TargetPracticeVector` behind a
   branch in `Collector`.
   *Why:* Eliminates the last dual-lifecycle branch in `Collector` in ~12 lines.
4. **Keep Torch `manabot.env.vector_env.VectorEnv`'s Gym-vector auto-reset
   contract explicitly inside the wrapper** (saving step-state buffers and
   terminal `last_info` *before* calling `load_slots_into_buffers`).
5. **Park readable `VectorEnv.step` terminals too**; readable callers restart
   with `reset_all`. No readable `load_slots` is added.

## Source of truth & invariants

- Native `SlotState` (`Unset`, `FreshOpening`, `Playing`, `Terminal`,
  `Poisoned`) in `managym/src/agent/vector_env.rs` is the sole authority for
  slot state. `FreshOpening` now has one job: permitting seed/recording rewind
  when an untouched fresh opening is replaced by `SlotLoad::Snapshot`.
- `finish_step` loses its `auto_reset` and `reset` parameters (if `terminated ||
  truncated`, set `*self = Self::Terminal`).
- `SlotLoad::Fresh(usize, Option<GameSetup>)` always draws `next_seeds[slot]`,
  advances it by `seed_stride`, validates `player_configs` length (`== 2`) and
  rejects `Some(setup)` when `!self.uniform_roster.is_empty()`, and updates
  `self.player_configs[slot]` only in the post-validation commit loop.
- `SlotLoad` exposes `.slot` (getter) and `.with_slot(slot: int) -> SlotLoad`,
  and `SlotLoadResult` exposes `.with_slot(slot: int) -> SlotLoadResult`, so
  `BufferedStreams.load_slots_into_buffers` remaps global ↔ local shard slots
  preserving exact native types and caller order.
- **Fresh load ≡ eager terminal reset** and **Fresh load on `Poisoned` ≡
  `replace_failed_into`** (verified in `vector_env/slots.rs`: same
  `replacement_root(next_seed)`, `reset_env_on_env`, roster sampler, and
  `GameRecording::default()` when `slots[slot] == Poisoned`).
- **Adaptive seed parity:** when a slot terminates under deferred reset,
  `next_seeds[slot]` is unspent (equal to the seed old eager reset + rewind
  used); `SlotLoad.fresh(slot, setup)` constructs the drawn setup once at that
  exact seed and advances `next_seeds[slot]` by `stride`.
- **Journal boundaries:** `RecordedVectorEnv.load_slots_into_buffers` records
  `self._begin_boundary(self._snapshot("reset"), "load_slots", setup_known=False)`
  before native execution and publishes `phase="ready"` after `_synchronize`
  reads exact native setups and seeds. Old journals keep their historical
  `"replace_failed"` / `"reassign_roots"` strings (`managym.replay` treats all
  non-step operations uniformly as plain labels).

## Critical edge cases verified by counterexample

1. **`MemoryStore` on parked `next_obs` rows (`collection.py` + `memory_store.py`):**
   In `complete-games` (and `exact-quota` when a stream hits quota on a terminal
   step), `_build_batch` calls `_retain_obs_rows(collector, np.arange(shape[1]))`
   while parked slots report `current_agent_indices()[row] == -1`.
   `MemoryStore.capture_live_rows` previously indexed `cursors[rows, -1]`
   (silently wrapping to seat 1!) against cleared terminal memory buffers.
   *Fix:* Reject negative seats in `MemoryStore.capture_live_rows`, and in
   `collection._capture_memory` only pass rows with `seat in (0, 1)` to
   `capture_live_rows`, returning an empty `Prefix(PrefixIdentity(f"terminal-{row}", 0, "terminal"), None)`
   for parked rows (`next_done` is True on those rows so their values never
   enter bootstrapping, while `next_obs` keeps its `(shape[1], ...)` shape).
2. **Torch `manabot.env.vector_env.VectorEnv.step` ordering:**
   `load_slots_into_buffers` overwrites `buffers["rewards"]`, `"terminated"`,
   `"truncated"` with zeros and replaces `last_info` for reloaded slots.
   *Fix:* Copy `_reward_tensor`, `_terminated_tensor`, `_truncated_tensor` and
   cache `_last_info = self._rust_env.get_last_info()` *before* calling
   `load_slots_into_buffers([SlotLoad.fresh(s) for s in done])`; then copy
   observation tensors from `_obs_cpu_views` and run `_apply_reward_policy()`.
   `get_last_info()` returns the cached terminal-step infos (and updates on
   `reset`).
3. **`action_limit` under `game_errors`:**
   `action_limit` is detected in Python while the native slot is still `Playing`
   (`acting >= 0`). Because `Collector` tracks ended rows in `_needs_refill` (or
   `_needs_assignment` / `snapshots.ready`) rather than inferring only from
   `acting < 0`, and `SlotLoad.fresh` accepts `Playing` slots, `action_limit`
   reloads cleanly in plain, adaptive, and snapshot modes.
4. **`exact-quota` terminal on the last step of a batch:**
   If stream `s` ends on its `num_steps`-th learner transition, `_settle_step`
   records `s` in `_needs_refill` and leaves it parked (zero batch-end waste);
   at the top of the next `collect`, active streams in `_needs_refill` are
   refilled while paused mid-game `Playing` streams continue undisturbed.

## Affected surfaces across layers

- **Rust (`managym/src/agent/vector_env.rs`, `vector_env/slots.rs`):**
  - Remove `VectorEnv.auto_reset`, `with_auto_reset`, `SlotState::finish_step`'s
    reset parameters, `replace_failed_into`, and `reassign_roots_into`.
  - Extend `SlotLoad::Fresh(usize, Option<GameSetup>)` and `SlotReplacement`
    (optional `player_config: Option<GameSetup>` committed to
    `self.player_configs[slot]` after all slots validate).
  - Add `SlotLoad::with_slot(&self, slot: usize) -> Self` and
    `SlotLoadResult::with_slot(&self, slot: usize) -> Self`.
  - Update Rust unit tests: rewrite eager-autoreset tests to assert terminal
    observation + explicit `SlotLoad::Fresh`; port `reassign_roots_into` tests
    (atomic validation, invalid deck/capacity rollback, recording sink
    preservation, updated `player_configs[slot]`, roster rejection) and
    `replace_failed_into` tests to `load_slots_into`.
- **PyO3 (`managym/src/python/vector_env_bindings.rs`, `managym/__init__.pyi`):**
  - Drop `auto_reset` constructor kwarg, `replace_failed_into_buffers`, and
    `reassign_roots_into_buffers`.
  - Update `SlotLoad.fresh(slot, player_configs=None)`; expose `SlotLoad.slot`,
    `SlotLoad.with_slot(slot)`, and `SlotLoadResult.with_slot(slot)`.
  - Update `.pyi` stubs.
- **Recording (`managym/recording.py`):**
  - Drop `auto_reset` parameter/attribute, `replace_failed_into_buffers`,
    `reassign_roots_into_buffers`, and the `if self.auto_reset` / `if not
    self.auto_reset` branches in `_step` and `_synchronize` (always finalize
    exposure on finished rows; `roots` comes only from reset/load calls).
  - In `load_slots_into_buffers`, use `self._snapshot("reset")` for
    `_begin_boundary` and check that `_synchronize` handles `SlotLoad.fresh`
    with updated setups via `current_player_configs()` (already automatic since
    `request.branch_root is None`).
- **Producers (`manabot/training/rollout/producers.py`):**
  - Delete `replace_failed_into_buffers`, `auto_reset` property, and
    `auto_reset` kwargs on `ShardedVectorEnv` / `open_native_streams`.
  - Add `BufferedStreams.load_slots_into_buffers(requests: Sequence[managym.SlotLoad]) -> list[managym.SlotLoadResult]`
    (validate slot bounds/uniqueness once, admit every shard before publishing shard-local loads via
    `request.with_slot(request.slot - shard.start)`, remap results back to
    global slots via `result.with_slot(offset + result.slot)` in caller order).
  - Delegate `BufferedStreams.install_slots` to `self.load_slots_into_buffers`.
  - Make `open_native_streams` always return `BufferedStreams` (single-shard
    when `workers == 1`, `ShardedVectorEnv` when `workers > 1`).
- **Collector & Memory (`manabot/training/rollout/collection.py`,
  `manabot/env/memory_store.py`, `manabot/env/target_practice.py`,
  `manabot/training/rollout/snapshot_vector.py`):**
  - `snapshot_vector.py`: drop `auto_reset=False` in `InMemorySnapshotVector`.
  - `target_practice.py`: park finished slots (`self.envs[row] = None`,
    `current_agent_indices()` returns `-1` for `None`) and implement
    `load_slots_into_buffers(requests)` for `SlotLoad.fresh`.
  - `memory_store.py`: reject negative seats in `capture_live_rows`.
  - `collection.py`:
    - Remove `auto_reset=not snapshots`, all `getattr(env, "auto_reset", True)`
      branches, `replace_failed_into_buffers` calls, and `episode.fresh_ready`
      auto-reset logic (`fresh_ready` is simply `True` after `open_collector`
      until first use, then `False`).
    - Add `_needs_refill: set[int]` on `Collector` for plain collection.
    - At top of `_collect` loop:
      - Snapshot mode: `_start_episodes` (unchanged structure, uses `install_slots`).
      - Plain mode: compute active rows needed (`~completed` for
        `complete-games`, `len(stream) < num_steps` for `exact-quota`); refill
        `rows = sorted(r for r in collector._needs_refill if needed[r])` in one
        `env.load_slots_into_buffers([SlotLoad.fresh(r) for r in rows])`,
        remove from `_needs_refill`, call `_refresh_roots`, and `_expose(..., "game_roots")`.
    - In `_settle_step`:
      - Add `ended = closed | failed` to `_needs_assignment` (adaptive),
        `snapshots.ready` (snapshot), or `_needs_refill` (plain).
      - Adaptive: `_reassign` refills non-completed rows in `_needs_assignment`
        via `env.load_slots_into_buffers([SlotLoad.fresh(r, setup) ...])`
        (and `commit_adaptive` refills batch-end `_needs_assignment` the same way).
      - Plain: leave refilling to top of next loop iteration (zero refill when
        the batch is done!).
    - In `_capture_memory`: handle parked (`seat < 0`) rows safely for `next_obs`.
- **Thin callers & scripts:**
  - `manabot/env/vector_env.py`: save step-state buffers + terminal `last_info`,
    refill done slots with `load_slots_into_buffers([SlotLoad.fresh(s)...])`,
    sync observation tensors, apply reward policy.
  - `manabot/sim/rollout.py` (`run_vector_games`): pass `active` mask for
    streams whose seat still needs games; refill finished streams whose seat
    still needs games via `load_slots_into_buffers`.
  - `scripts/bench_sharded_collection.py`: refill finished slots via
    `ShardedVectorEnv.load_slots_into_buffers`.
- **Docs & Tests:**
  - Update `docs/engine.md`, `docs/engine-python.md`, `docs/training-loop.md`,
    `docs/python-api-audit.md` (leave historical `release/v0.2.1/NOTES.md`).
  - Update tests referencing `auto_reset`, `replace_failed_into_buffers`,
    `reassign_roots_into_buffers`, or stepping past terminal without refill
    (including fake vector in `tests/training/test_game_errors.py` and readable
    step tests in `tests/academy/test_game_records.py`,
    `tests/env/test_engine_recording.py`, `tests/env/test_game_record_writer.py`).
  - Add directed tests for:
    1. `SlotLoad.fresh(slot, player_configs=setup)` atomicity, roster
       rejection, updated setup persistence, and `Playing`/`Poisoned`/`Terminal`
       acceptance (Rust + Python).
    2. Plain & adaptive Collector seed/record/statistics parity vs pinned
       pre-change values, plus native `CONSTRUCTIONS` == 0 for batch-end
       terminals.
    3. `complete-games` and `exact-quota` terminal-on-last-step with
       `decision_memory=True` (verifying `next_obs` / `MemoryStore`).
    4. Adaptive + `game_errors` hitting `action_limit` (`Playing` slot).
    5. Torch `manabot.env.vector_env.VectorEnv.step` preserving terminal reward,
       `terminated`, `winner_index` in `get_last_info()`, and replacement obs.

## Slices

1. **Native and recorded lifecycle, including the Torch consumer.** Native readable,
   strict buffered, and isolated steps park a finished slot. The sparse native
   loader admits an optional setup and publishes it with the new game.
   Recorded vectors settle the old game's evidence and read the admitted
   setup at its actual seed. Torch retains the step outcome across its explicit
   reload. Native transports now construct the same vector for every mode.

   **Sharded transaction built.** Independent shard calls could spend a seed
   on one shard before another rejected a request. The native binding borrows
   the selected vectors together, admits every destination/setup/observation,
   then publishes each replacement once. Recorded roots synchronize after the
   joint native call. Ordinary stepping still uses the existing producer pool.

   **Checked in this slice.** Four native tests passed for terminal parking,
   construction counts, fresh setup admission and recording sinks. Focused
   Python checks passed for recorded terminals, rollback and retry across
   shards, cold snapshot roots, statistics, journals and Torch outcomes.
   The engine guide's batched example loaded seeds 42 and 45 and left the
   next fresh seeds at `[42, 47]`.

2. **Next — `Collector` + `MemoryStore` + `TargetPracticeVector`:**
   Adaptive assignment, isolated recovery and snapshot installs use the new loader.
   Move plain terminal and failed-game refill to the next active collection round,
   and move target-practice refill onto
   `load_slots_into_buffers`; fix parked-row `MemoryStore` handling; update
   training tests and add parity, zero-batch-end-construction,
   `decision_memory` terminal, and adaptive `action_limit` tests.
3. **`run_vector_games`, `bench_sharded_collection.py`, and remaining caller checks**, followed by full acceptance checks.

## Acceptance check at gate

```bash
uv run --extra dev pytest \
  tests/env/test_vector_snapshot_slots.py \
  tests/env/test_fixed_seat_observations.py \
  tests/env/test_hand_snapshots.py \
  tests/env/test_vector_reducers.py \
  tests/env/test_env.py \
  tests/env/test_engine_recording.py \
  tests/env/test_game_record_writer.py \
  tests/academy/test_game_records.py \
  tests/training/test_game_errors.py \
  tests/training/test_snapshot_rollouts.py \
  tests/training/test_training_game_journals.py \
  tests/training/test_training_decks.py \
  tests/training/test_adaptive_two_seat.py \
  tests/training/test_collection.py \
  tests/training/test_memory_prefix_storage.py \
  tests/verify/test_native_buffer_capacities.py \
  tests/test_package_boundaries.py -q
CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml --test-threads 2 vector_env
git grep -nE "auto_reset|with_auto_reset|replace_failed_into|reassign_roots_into" -- managym manabot docs tests scripts
```

Expected: all tests green; grep returns nothing outside `release/`.

## Docs

Updated user-facing docs describing the intended behavior (written before implementation):

- `docs/engine-python.md`
- `docs/engine.md`
- `docs/training-loop.md`
- `docs/training.md`
- `docs/python-api-audit.md`
- `manabot/README.md`

## Demo

`uv run --extra dev pytest tests/env/test_vector_snapshot_slots.py -q -k lifecycle`
— a two-slot vector steps slot 0 to its terminal: the buffer shows the terminal
row with `terminated=1`, chooser `-1`, and `CONSTRUCTIONS` unchanged; stepping
again raises; `load_slots_into_buffers([SlotLoad.fresh(0, alt_setup)])` installs
seed 44 (`= 42 + stride`) with `alt_setup`, updates slot 0's setup for future
loads, and the game record file holds the finished game once.
