# Remove eager auto-reset from vector collection (Task a2da47)

## Outcome

The Task brief calls for “one explicit slot lifecycle: `step` leaves
finished/failed slots in `Terminal`/`Poisoned`, and
`load_slots_into_buffers` refills them (with fresh games or snapshots).”

The terminal step reports its observation, reward, flags and winner once.
Parked slots report chooser `-1`. One atomic sparse load spans native,
recorded and sharded vectors, preserving seed order, roster selection,
recordings, statistics and exposure. A fresh load spends the next seed and
can update the slot's setup. A snapshot load spends no fresh seed and refunds
an untouched opening's seed. Failed loads leave slots and buffers intact.

Collection refills a finished game only when its stream needs another
decision, including after a Python `action_limit` leaves the native slot live.
The Torch Gym-vector wrapper keeps its automatic reset behavior by explicitly
loading fresh games after saving the step outputs. User docs landed in
`d498d114ae`. Design choices are in `scratch/questions.md` and the concept
review is in `scratch/concept-review.html`.

## Completed slices

1. `7ce9093dab` parks native slots and unifies fresh, failed and snapshot loads across native, recorded and sharded vectors.
2. `6d3cf02810` refills only active ended collector streams, parks target-practice terminals and supplies empty memory prefixes for parked bootstrap rows.
3. `3d27339e7a` retains terminal winners in batched evaluation, retires filled seat quotas and times benchmark refills after hashing the finished batch.

Snapshot dependency #581 merged into main at `82c5d1d1b7`. The branch
integrates the incoming collector continuation in `a6b50cd8bd`. A save
opens the fresh roots it needs for replay. The next snapshot allocation can
reuse one of those openings or replace it and recover the seed. The batch
boundary itself still leaves ended slots parked. The save/restore regression
is `test_collector_resumes_after_snapshot_branch` in
`tests/training/test_snapshot_rollouts.py`.

## Acceptance evidence

The initial Python acceptance run passed 401 cases, skipped five and failed
ten. Three failures came from a frozen scalar replay whose first recorded
action was out of range at the opening-hand choice added by `d12eb496b3`.
Adding two explicit Keeps still diverged at action 34. `669615a0e9` keeps the
frozen replay as a rejection case and records a current five-action replay
to test stale binding and typed capacity failure. `fc86c0536e` corrects six
journal expectations about the opening-hand turn and explicitly resets a
finished statistics game. All eleven cases covering these fixes passed on
focused reruns. Six rollout, benchmark and failure-capture checks passed.
The native filter passed all 26 tests, and the removed API grep returned no
matches. Results are recorded in `1eac5b227a`.

The documented mimalloc benchmark completed 2,048 decisions with digest
`abd73687fa664633fec80c624d75723e8669ad9fec85a41bd30c95243d2934b9`.

## Compression checks

All 26 native vector tests passed after consolidating step validation and slot
transitions. Focused Python checks covered recorded roots and reducers, the
durable return boundary, adaptive refills, snapshot resume, parked memory
prefixes, Torch device views and terminal reward shaping. The mixed-load
documentation example printed seeds 42 and 45, then next seeds 42 and 47.

Both profiling entry points ran with one stream. The matched corpus loaded
each fixture by snapshot, verified scalar and vector results, and reported
2, 4, 6, 2 and 33 output actions across its five cases. These were functional
checks with a single sample per case.

## Next at Gate

All implementation slices are complete. Reuse the focused results above.
The combined Python acceptance suite still needs a green run with the fixes
on the integrated branch. Include the continuation and collector checks from
[validation by change area](../docs/development.md#validation-by-change-area).

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

## Demo

`uv run --extra dev pytest tests/env/test_vector_snapshot_slots.py -q -k lifecycle`
steps one of two streams to a terminal with chooser `-1`. The completed game
and winner appear in the recording. An explicit fresh load installs seed 44
with the replacement setup, leaves the neighboring stream alone and clears
the terminal buffer. The next fresh load keeps that setup at seed 46.
