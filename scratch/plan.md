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
3. **`action_limit` under `game_errors`:**
   `action_limit` is detected in Python while the native slot is still `Playing`
   (`acting >= 0`). Because `Collector` tracks ended rows in `_needs_refill` (or
   `snapshots.ready` for a snapshot allocation) rather than inferring only from
   `acting < 0`, and `SlotLoad.fresh` accepts `Playing` slots, `action_limit`
   reloads cleanly in plain, adaptive, and snapshot modes.
4. **`exact-quota` terminal on the last step of a batch:**
   If stream `s` ends on its `num_steps`-th learner transition, `_settle_step`
   records `s` in `_needs_refill` and leaves it parked (zero batch-end waste);
   at the top of the next `collect`, active streams in `_needs_refill` are
   refilled while paused mid-game `Playing` streams continue undisturbed.

## Slices

1. **Done (7ce9093dab):** native/recorded/sharded parking and atomic
   `load_slots_into_buffers` with `SlotLoad.fresh(slot, setup)`, `with_slot`,
   `open_native_streams` always `BufferedStreams`, Torch `VectorEnv` retaining
   step outcomes across reload, eager-reset APIs deleted from managym and
   producers. Collector already uses the loader for adaptive `_reassign`,
   failed-game replacement (immediately, inside `_settle_step`) and snapshot
   installs.

2. **Done (6d3cf02810):** Collector refills only active ended rows at the top
   of each round (`_needs_refill`), adaptive commit draws after score updates,
   target practice parks and loads explicitly, parked bootstrap rows get empty
   memory prefixes; seed-parity, batch-end and `action_limit` tests added.

3. **Next — `run_vector_games` (`manabot/sim/rollout.py:178-226`) and
   `scripts/bench_sharded_collection.py:44-62`** step parked slots after a
   game ends and would raise `cannot step Terminal slot`. Refill done rows
   with `load_slots_into_buffers([SlotLoad.fresh(s) ...])` after reading
   `get_last_info()`; in `run_vector_games` refill only streams whose hero
   seat still needs games and step only live rows (or stop when none). The
   bench digest changes if refill timing moves; record it if Engine Budget
   reads this script. Then run the full acceptance check below.

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

User docs were updated before implementation (d498d114ae).

## Demo

`uv run --extra dev pytest tests/env/test_vector_snapshot_slots.py -q -k lifecycle`
— a two-slot vector steps slot 0 to its terminal: the buffer shows the terminal
row with `terminated=1`, chooser `-1`, and `CONSTRUCTIONS` unchanged; stepping
again raises; `load_slots_into_buffers([SlotLoad.fresh(0, alt_setup)])` installs
seed 44 (`= 42 + stride`) with `alt_setup`, updates slot 0's setup for future
loads, and the game record file holds the finished game once.
