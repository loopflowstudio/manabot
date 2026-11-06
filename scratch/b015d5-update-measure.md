# b015d5 engine-measure: training update, current head (2026-11-06)

Workload: `elemental-long-183-v1` re-pinned to the current world through
`scripts/bench_training_update.py` (256-stream two-seat Elemental, collection and
learner through `execute_regime`). Recipe, world and observation version unchanged.

## Identity and limits

- Source `76e71abc3` (PR #435 head); extension `_managym.cpython-312-darwin.so`
  built 2026-11-05 22:20 from it. Apple M4 Max, 16 logical CPUs, CPU device,
  torch 2.10.0, Python 3.12.12, `threads=1`, `workers=1`.
- Shared machine, load 13-31 during these runs. Whole-update wall time is not
  comparable across these runs (115.6 vs 95.2 s/update at 256 streams). Read
  per-phase µs/decision and compare same-machine interleaved pairs only.
- No GPU. Model phases on CPU are not GPU evidence. No speed claim for GPUs.
- Probes, logs and raw output: `.lf/logs/b015d5-measure/` (gitignored, this
  checkout). `opcount*.py` count aten dispatches, `checkcost.py` times
  `PolicyInference.check`, `run_keep.py` + `digest.py` give behavior digests.

## Behavior digests (baseline)

`run_keep.py <dir> 24 3` then `digest.py <dir>`: seed 25201, 24 streams,
3 updates, CPU. Two runs gave identical results: 72 games, 14,311 decisions.

| Artifact | Digest |
|---|---|
| raw weights `self-play-raw.pt` | `76e926e3ea4fcde1` |
| EMA weights `self-play-ema.pt` | `2a31f25602a86b9c` |
| optimizer `self-play-optimizer.pt` | `05eb51c04b5e2d14` |
| game attempts (72, actions/outcomes/frames) | `cfcf3e68b8dbb6cb` |

Run ids and timing coordinates are excluded (`VOLATILE` in `digest.py`).

## Measured phases, current head, 256 streams (µs per decision)

Two unprofiled rows plus one cProfile row (cProfile inflates Python-heavy phases):

| Phase | Run a (2 updates) | cProfile run (1 update) |
|---|---|---|
| Policy inference and action selection | 1190 | 958 |
| Environment step and encoding | 99.0 | 77.4 |
|   native stepping | 64.3 | 52.6 |
|   observation encoding | 11.3 | 7.5 |
| Batch assembly | **26.9** | 25.6 |
| Rollout recording | 14.3 | 11.4 |
| Observation preparation | 11.5 | 14.2 |
| Stream bookkeeping | 8.7 | 7.9 |
| Batch transfer and behavior checks | 7.7 | 7.9 |
| Minibatches | 848 | 710 |

52,291-52,411 decisions and 257 games per update; 722 vector steps, so a step
averages ~72 active streams of 256 (the complete-games tail).
At 64 streams: assembly 19.2, recording 12.8 µs/decision.

Host-side findings (CPU work on the rented host's one thread, any GPU):

1. **Batch assembly re-copies every observation.** Recording already copies each
   step's rows (`self._buffers[key][rows]`), then `_build_batch` `np.stack`s
   ~52,000 per-row views per key into freshly zeroed `[T, E, ...]` arrays. Non-action
   input is 82,416 bytes/row (facts 24,576; each card block 13,440; relations
   12,544), so each copy is ~4.3 GB per update at 256 streams and ~1.2 GB at 64.
   The 64-stream run peaked at 2.85 GB footprint (3.20 GB max RSS,
   `/usr/bin/time -l`). Assembly is ~1.4 s/update on this laptop, mostly per-row
   overhead and fresh-page faults, both slower on rented single threads.
2. **Per-row observation dicts.** Each `_Pending` holds a dict of 22 views
   (~1.15M numpy view objects per update), part of recording.
3. `PolicyInference.check` runs three times per step (prepare, forward, catalog).
   Each pass walks parameters, buffers and modules separately: 112 µs per check
   in isolation (`checkcost.py`), ≈0.24 s/update, ≈5 µs/decision. (cProfile
   showed 0.9 s; that is inflation.)
4. `CapturedVector._step` serializes `semantic_decision_frames_json` for every
   active row before every step: 0.52 s/update, ≈10 µs/decision. The frame is
   retained failure evidence, compared on replay.

GPU-relevant counts (hardware independent, dispatches on CPU):

- One collection sampling call: 802 aten ops, 37 host-sync ops (`_local_scalar_dense`,
  `nonzero`); the forward alone is 629 ops and 17 syncs. ~722 calls per update.
  This fits the L4 receipts (4.7 s / ~700 calls ≈ 6.7 ms per call), which are
  launch- and host-bound.
- Learner forward under autograd: 7,716 ops/forward, of which 6,954 are the CPU
  GRU unrolled per time step in `SemanticCardEncoder._encode_programs` (~339 steps).
  CUDA runs that GRU as cuDNN, so the CPU count is not GPU evidence. The rest is
  ~760 ops and ~31 syncs per minibatch; CPU backward is 10,344 ops.
- Minibatch diagnostics overwrite entropy/KL/clip/loss each minibatch (only the last
  survives) and sync ~8 times per minibatch. That is small on CPU (0.9-1.1 µs/decision).

Native stepping (53-64 µs/decision) is the largest host cost but small within a
CPU-device sample (186 of ~15,000 samples, `sample-64.txt`). It belongs to the
engine budget harness (`bench_search`/`bench_steps`) and the engine Tasks. Here it
stays an unchanged control.

## Done 2026-11-06: one recorded copy per observation

`_ObservationRows` (collection.py) holds one update's non-action rows; `_Pending.obs_row`
is a store row; `RolloutBatch.obs` is the recorded rows `[N, ...]` with `obs_rows`
`[T*E]`, and `ordered_obs()` is the one `[T, E, ...]` accessor (incidents, tests).
First-update capacity comes from the closure's row floor, then 1.25x the previous rows.
Results are in `experiments/training-update-time.md`: assembly + recording falls
12-16 µs/decision in all 7 pairs (~0.6%), no whole-update or memory claim (peak is
set elsewhere). Digests: weights/EMA/optimizer match the table above; game attempts
`cb230744dcd7c850` for both base and new with `digest2.py` (also excludes
`source_commit`, `training_source_sha256`, which `digest.py` hashed, so its
attempts digest moved with any source change). Raw rows: `ab-*.txt` in the logs dir.
Checks: focused training suites pass (`test_game_errors` resource-limit replay fails
3/3 as inherited, see questions.md); `test_recovery -k real_interruption` passes.

Stopping: every remaining host lever is below the noise here (check ≈5, failure
frame ≈10 µs/decision, the frame needs review); inference and minibatches on CPU are
not GPU evidence. The named CUDA check is the next producer of evidence.

## Next phases (not active; each needs its own evidence or review)

- 1b, small and exact: one parameter/buffer/module walk per `PolicyInference.check`,
  or one full check per sampled step. ≈5 µs/decision. This is a safety contract:
  keep rejection of in-place updates, tensor replacement and mode/config change.
- Failure-capture frames (≈10 µs/decision): needs review, because the frame is
  replay evidence. Native isolated steps already return `game_error_frame` per
  failed row; the Python frame serves whole-call failures.
- GPU launch/sync reductions (inference 629 ops / 17 syncs per forward; learner ~31
  syncs per minibatch; overwritten minibatch diagnostics). Unmeasurable here. Named
  CUDA check for a real L4 run: `torch.cuda.set_sync_debug_mode("warn")` counts and
  a `torch.profiler` kernel count over updates 2-3, plus
  `time/collection/policy_inference_and_action_selection` and `time/learning/minibatches`.
- Parallel producers would overlap native stepping (~20-27% of a GPU update), but
  the recipe pins `workers: 1` and `elemental-uniform` requires one producer. That
  is a recipe decision, outside this Task.
- Native stepping: engine budget harness, owned by the engine Tasks.

Compress 2026-11-06: `_build_batch` copies `next_obs` straight from the buffers
(no `allocate(1)` template). Focused training suites: 181 passed. Realign
2026-11-06 re-ran the digest at `aa592dfa0` (no Rust change since `76e71abc3`):
raw/EMA/optimizer/attempts `76e926e3ea4fcde1` / `2a31f25602a86b9c` /
`05eb51c04b5e2d14` / `cb230744dcd7c850`, identical to base (`keep-head`).
The 256-stream timing rows were taken at `b958f3a00`; the compress change only
moves the final-observation copy and was not re-timed.

## Concept review 2026-11-06 (at `cd77e2ed1`): rollout storage

Walked: collect → learner minibatch → incident save → test reader. Phase tree,
host facts, bench and machine recommendation stand; no open product choice.

Mismatch: `RolloutBatch.obs` kept its name but changed meaning, from time-major
`[T, E, ...]` to unordered recorded rows `[N, ...]` that need `obs_rows` to mean
anything. Old readers still run, just on the wrong axes. Counterexample, run here:
`tests/model/test_layer_input.py::test_v8_recipe_collector_and_evaluation_select_saved_input`
asserts `(1, 1, 4096, 3)`, and `tests/model/test_recent_events.py` sums `events_valid`
over `(-1, -2)`, which now collapses rows, then passes recorded-row numbers to the
time-major `observations(indexes)`. Result: 8 failed, 2 passed. manabot-integration
CI runs `tests/model/`, so PR #435 would fail there. The focused training suites
missed it.

Next implement direction (follows from one owner per fact and named types):
- Replace the `obs` + `obs_rows` pair with one typed owner, e.g.
  `RecordedObservations` (`rows: dict[str, ndarray]` `[N, ...]`, `index` `[T*E]`),
  next to `action_data: PackedActions`. Give it `gather(indexes)` (time-major
  indexes in, rows out) and `ordered(shape)` (the `[T, E, ...]` copy that
  incidents and tests use). `RolloutBatch.observations()` and `ordered_obs()`
  delegate to it. Drop the field name `obs` so a stale reader fails with
  AttributeError instead of reading the wrong axes.
- Move the two `tests/model` readers to `batch.ordered_obs()` (or `observations`),
  keeping their assertions. Run all of `tests/model/` and `tests/training/` before
  gate, not only the focused suites.
- No behavior change: digests stay `76e926e3ea4fcde1` / `2a31f25602a86b9c` /
  `05eb51c04b5e2d14` / `cb230744dcd7c850`, and timing rows are not re-taken.
