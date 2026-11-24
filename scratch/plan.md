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

The save contains `state.json` (Etude inventory and counts),
`continuation.json` (collector and host RNGs), `learner.safetensors` (canonical
raw/EMA weights, Adam state and integrity metadata), and
`sampling.safetensors` (domain-specific RNG state). The shared reader checks
the saved model, readings, observation layout, lookup buffers, file digests,
weights and optimizer state before installing learner tensors. The Etude
revision also supplies versioned storage listing for training-time reports.

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
  The Metal case checked same-domain repeatability. GPU backend switches and
  the released SDK remain unverified.

## Remaining slice — use released Etude SDK 0.4.6

Etude PR #141 merged at `af71c109` after 0.4.5. PyPI still listed 0.4.5 on
2026-11-23 and its release inventory had no 0.4.6. The dependency is the
0.4.6 release with public callbacks and versioned storage listing.
[Etude SDK on PyPI](https://pypi.org/project/etude-sdk/)

When 0.4.6 is published, remove the Git source in `pyproject.toml`, set the
base and extra requirements to `>=0.4.6,<0.5` and the `play-runtime` pin to
`==0.4.6`. Update the matching release smoke pin in
`.github/workflows/release.yml`, the metadata requirement in
`scripts/release/validate.py` and its fixtures in `tests/infra/test_release.py`,
and the requirements in `docs/development.md`. Existing experiment pins stay
as recorded.

After those edits, run `uv lock --upgrade-package etude-sdk` first. Verify
locked sync and the cross-backend acceptance suite with the PyPI SDK. Run the
release metadata fixtures with
`uv run --no-project --with pytest pytest --confcutdir=tests/infra tests/infra/test_release.py -q`.
