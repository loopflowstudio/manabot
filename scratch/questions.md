# Open questions (choices made for review)

- **etude-sdk 0.4.6 release.** Etude PR #141 (`af71c109`) merged after v0.4.5;
  PyPI latest is 0.4.5. This Task builds against Etude main through a
  temporary uv source and cannot land on PyPI pins until 0.4.6 is published.
  Publishing is outward-facing and was not done here. Needed: cut Etude
  v0.4.6 (it would also carry #142/#143).
- **Dedicated `learner.safetensors` instead of published policy containers
  (`raw.safetensors` / `ema.safetensors` + `optimizer.safetensors`).** Chosen
  because live training continuation requires strict exact-match admission,
  precision tracking, and fast saves without git subprocesses, duplicated
  lookup buffers, or `load_policy` catalog rebinding; one file also owns the
  complete cross-file integrity manifest (`counts_sha256`,
  `continuation_sha256`, `sampling_sha256`, per-buffer digests). Passed over:
  six files reusing `write_policy` for raw/EMA.
- **Torch → MLX Adam clocks.** A CPU probe with a parameter at step 1,
  global step 2, gradients +1 then −1, eps 1e-5 and rate 0.001 gave next
  weights −0.0009473588 on Torch and −0.0012619789 with the proposed MLX
  moment rescaling. Rescaling preserves the corrected moment at the save
  boundary only. Preserve the original moments and count instead. MLX can
  store a fixed clock offset because it advances every trained leaf on each
  subsequent dense optimizer update.
- **Single `sampling.safetensors` keyed by `sampling_domain` (`mlx`,
  `torch:cpu`, `torch:cuda`, `torch:mps`).** Chosen because PyTorch generator
  RNG states differ across CPU (5056 B MT19937) and CUDA/MPS (16 B Philox),
  so keying only by backend name crashes on Torch CUDA↔CPU resume. Cross-domain
  resume verifies `sampling_sha256`, reseeds deterministically from
  `(run.seed, counts.updates)`, and logs `run/sampling_reseeded=1`. Passed over:
  `sampling.<backend>.safetensors`.
- **No fallback for unmerged dba0f2 MLX `training.safetensors` saves.** Chosen
  because that format exists only on unmerged dba0f2 (PR #585); fails
  explicitly naming `training.safetensors`. Passed over: a second read-only
  loader for MLX schema 1/2.

## Surfaced while writing docs

- **Local CLI backend/device switching (`etude train --resume` vs `etude continue`):**
  `SelfPlayRecipe` declares `backend` and `device` on the recipe config, and
  `etude train RECIPE --resume` refuses any config change against the saved run.
  Switching backend or device therefore goes through
  `etude continue --run RUN --until updates=N --config '{"backend":"mlx","device":"mps"}'`
  (launched runs) or direct Python `self_play(..., backend=...)` against the same
  `ETUDE_OUT`. Confirm whether that is the intended CLI surface, or whether
  local runs also need a way to switch `backend`/`device` without `etude continue`.
- **Canonical `precision` spelling in `learner.safetensors`:**
  `SelfPlay.precision` uses `"float32" | "tf32" | "bf16"` while
  `MLXActorCritic.precision` uses `"float32" | "bfloat16"`. Record the
  `SelfPlay.precision` spelling (`"float32"`, `"tf32"`, `"bf16"`) in
  `learner.safetensors` so CUDA Torch `bf16` → MLX `bf16` passes exact precision
  admission.

## Slice 2 validation

- Compare all learner tensors and parameter clocks exactly across independently
  timed CPU runs. Each learner header binds its own `Counts.training_seconds`
  through `counts_sha256`, so the whole learner files can differ even when
  every tensor is identical. The host and sampling files can be compared as bytes.
