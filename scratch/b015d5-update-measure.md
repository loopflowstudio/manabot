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

## Active phase: one recorded copy per observation (implement this)

Goal: delete assembly's second full copy and the per-row view dicts while the
learner gets bitwise-identical rows in the same order.

Implementation (in `manabot/training/rollout/collection.py`):

- A private per-update observation store (`_ObservationRows` or similar) holds one
  array per non-action key, `[capacity, *shape]`. Recording writes each step's rows
  with `np.take(self._buffers[key], rows, axis=0, out=store[key][n:n + k])`
  (or an equivalent single copy). It returns the base row; growth doubles
  capacity, copying used rows once. Allocate fresh per update; do not reuse across
  updates, because a returned batch may be retained (incidents, recovery) while
  the next collection runs. Reuse is a later measured choice only.
- `_Pending.obs: dict[str, np.ndarray]` becomes the store row index (int).
- `_build_batch` builds an int64 `rows` index (packed position → store row) in the
  existing time-major/packed order instead of `np.stack`, and gives `RolloutBatch`
  the store arrays trimmed to used rows (views) plus that index.
  `RolloutBatch.observations(indexes)` gathers `obs[key][rows[indexes]]`: one
  gather per key per minibatch, as today. Keep `next_obs`, `action_data`, scalars
  and `outcome_probabilities` unchanged.
- Keep one owner for the ordered `[T, E, ...]` view. `incidents.py` (failure
  dump) and tests read `batch.obs`; give them one materializing accessor (e.g. the
  existing `observations()` with no indexes, or a named method). Do not keep two
  representations.
- Snapshots are taken only at empty update boundaries; the store is empty there.
  Recovery journals, exposure counters and stats are unchanged.

Deletion targets: the per-stream, per-key `np.stack(..., out=column[:, env_index])`
loop and the `obs` zero allocation in `_build_batch`; per-row `obs` dicts in
`_Pending`; `obs_rows` dict-of-copies in `_sample_learning_rows` (replaced by the
store write).

Focused checks:
- `uv run --extra dev pytest tests/training/test_collection.py tests/training/test_batch_closure.py tests/training/test_rl_objectives.py tests/training/test_training_decks.py tests/training/test_packed_action_rollouts.py tests/training/test_naming_transitions.py tests/training/test_phase_timing.py tests/training/test_regimes.py -q`
- `uv run --extra dev pytest tests/training/test_recovery.py -q -k real_interruption`
- Digests: `run_keep.py` + `digest.py` at 24 streams × 3 updates must reproduce the
  four baseline digests above exactly.
- Python-only change: no extension rebuild, Rust tests or Engine Budget movement
  expected. Run `uv run python scripts/check_loopflow_workflows.py` only if
  workflow files change.

Measurement (same machine, interleaved base/new, read minima and per-pair ratios):
- Base worktree at `76e71abc3` (`git worktree add`); alternate base/new runs of
  `bench_training_update.py --streams 64 --updates 2` (ABBA, ≥3 pairs) and at least
  one 256-stream pair. Compare `Batch assembly` and `Rollout recording` µs/decision.
  Expect assembly to fall from 19-27 to a few µs/decision. Claim only what all pairs agree on.
- Peak memory: `/usr/bin/time -l` on the same 64-stream command, base vs new.
  Expect about −1.2 GB footprint at 64 streams (one copy of 14k × 82 KB), and
  about −4.3 GB at 256 streams if measured.
- Record results in `experiments/training-update-time.md` (fixed-hardware section)
  with the identity above. Make no GPU or whole-update claim this noise cannot support.

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
