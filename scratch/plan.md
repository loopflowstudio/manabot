# a66f13 — portable continuation for tracked self-play

## Outcome

A tracked self-play run can resume on Torch or MLX with the same policy, EMA,
Adam moments and per-parameter clocks, counts, input readings, and in-flight
native games. Restore host RNGs on every portable resume and sampling RNGs
within the same domain (`mlx`, `torch:cpu`, `torch:cuda`, `torch:mps`).
Across domains, verify the saved RNG file, reseed from the run seed and the
completed update count, and log `run/sampling_reseeded=1` on the first resumed
update.

Historical Torch object-registration saves remain readable on Torch. They open
fresh game and sampling streams and write portable state at the next save.
For pre-portable MLX saves, the recovery guide directs readers to the writer
revision. Published policy milestones and numerical incident snapshots keep
their existing contracts.

## Built

1. `fa2c52e493` — canonical weights, EMA, and Adam hooks, with per-leaf bias correction on MLX.
2. `bb0c7267b6`, `5c91b7ff7d` — one Etude callback path for both backends, with native game and RNG recovery and a temporary Etude Git pin.
3. `30db47c5c1` — tracked continuation acceptance, a smaller Metal fixture, CI wiring for the acceptance file, and training guide updates.
4. `c98aa96bab` — switched from the temporary Etude Git source to PyPI release `etude-sdk` 0.4.6 (`>=0.4.6,<0.5` for package/extras; `==0.4.6` for play runtime and release smoke); merged `main` (`82c5d1d1b7`, native snapshot self-play) in `d9e81afec3`.

The save contains `state.json` (Etude inventory and counts),
`continuation.json` (collector and host RNGs), `learner.safetensors` (canonical
raw/EMA weights, Adam state and integrity metadata), and
`sampling.safetensors` (domain-specific RNG state). The shared reader checks
the saved model, readings, observation layout, lookup buffers, file digests,
weights and optimizer state before installing learner tensors. Etude SDK 0.4.6
also supplies versioned storage listing for training-time reports.

The parent Task note recorded this direction, “Compose canonical
raw.safetensors / ema.safetensors / optimizer.safetensors cleanly under
continuation.json without rewriting CollectorSnapshot.” The branch preserves
`CollectorSnapshot` and proposes one strict learner file for review. It packs
the same canonical arrays with a single integrity manifest and avoids the
published policy writer's catalog rebinding and repeated lookup buffers.
See `scratch/questions.md` for the format and optimizer choices.

## Recorded validation

- `tests/training/test_portable_adam.py` passed on CPU, with an exact
  Torch→Torch update and cross-backend updates within float32 tolerance.
  Per-parameter clocks can lag when Torch skips a parameter with no gradient.
  MLX preserves the lag for future bias correction.
- Slice 2's parallel run passed 55 cases. The Metal safe-stop subprocess hit
  its 120 s deadline under `-n 4`, and all five safe-stop cases passed in
  292 s when run alone. Slice 3 reduced the Metal fixture to one small optimizer
  batch. That case passed in 57 s on its own.
- All seven cases in `tests/training/test_cross_backend_continuation.py`
  passed serially in 194 s against the Git pin. Each direction preserved
  native games, host RNGs, raw and EMA parameters, Adam moments and clocks,
  counts and readings at the first resumed decision. Both trained through
  update 4 and logged the reseed only on update 3.
- The same suite verified digest refusal and both expected CPU generator seeds
  with a simulated CUDA RNG file, MLX resume with Torch imports blocked,
  historical Torch state migration and MLX refusal, and Torch CPU safe-stop
  against an uninterrupted run. The CPU comparison checked host and sampling
  files byte-for-byte and all learner arrays and metadata exactly, with each
  counts digest checked against that run's counts, which include measured
  training time.
- All 42 package-boundary tests passed. Cross-backend cases used CPU.
  The Metal case checked same-domain repeatability. GPU backend switches
  remain unverified.
- Both training and play runtimes installed Etude SDK 0.4.6 from the lock on
  Apple Silicon with CPython 3.13. The isolated play install included the
  reports and evaluation extras and passed `uv pip check`. All seven
  continuation cases passed against the PyPI release in 53.77 s, and all
  39 release metadata fixtures passed.
- After merging `main` (`82c5d1d1b7`) in `d9e81afec3`, all four snapshot
  self-play cases (`tests/training/test_snapshot_rollouts.py -k self_play`,
  5.49 s) and all 42 package-boundary tests (`tests/test_package_boundaries.py`,
  10.94 s) passed.

All four implementation slices are complete; no slices remain.
