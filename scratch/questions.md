# Choices made without review

## Concept review (2026-11-23)

- **Portable `belief.safetensors` on both backends** (replaces kickoff's
  non-portable `.pt`/`.safetensors` split): repo check found zero frozen
  `belief.pt` artifacts or hardcoded sampler byte digests. Torch keeps legacy
  `PK\x03\x04` `.pt` read fallback by magic bytes (matching policy
  checkpoints); MLX refuses legacy `.pt` naming Torch.
- **Collapse update protocol tiers**: delete `PPOBackend`, `PPORL`,
  `CategoricalRL`, `PPOPolicy`, `PPOLearner`, `PPOAdmittedRollout`,
  `PPORolloutFile`, `ppo_rl()`, and `categorical_rl()` across Slices 1–2 so
  both backends satisfy one `UpdateBackend` / `RL`.
- **Runtime admission for `live_tokens` + `O(B * max_live^2)` MLX SDPA**:
  refuse Torch off-CUDA `live_tokens` in `backend.runtime.admit_execution`
  (not `ActorCritic.__init__`, which breaks CPU tests and `load_checkpoint`),
  and compact Q/K/V to `[B, H, max_live, d]` around fused SDPA instead of
  building an `O(B^2 * L^2)` `[T, T]` block-diagonal mask. Qualify MLX-live
  against both MLX-padded and Torch-live (CPU).

## Slice 1 (2026-11-23)

- Checked PPO gradients using Adam's first moment after one clipped step.
  On the first step it is proportional to the accumulated gradient, so the
  test can check per-tensor cosine through the actual optimizer on both
  backends, including a split MLX minibatch.

## Slice 2 (2026-11-23)

- Used MLX's built-in multihead attention for object self-attention and the
  learned critic query. It already runs fused SDPA with the checkpoint's
  separate Q/K/V projections.
- The attention-off historical mean differed from Torch by up to 0.051 before
  masking continuation context by object validity. That table row passed the
  existing forward, gradient and Adam tolerances with the mask.

## Realign after Slice 2

- Use the two declared jobs in `experiments/belief.py` for the demo. In
  Etude SDK 0.4.5, `etude/controller/local.py:train` forwards the `--config`
  override to `_prepare` for the selected job and places the declared inputs.
  Both jobs use the same fresh experiment label so the sampler declaration
  sees the policy's final checkpoint. The linked MLX run is still Slice 3
  acceptance.

## Slice 3 (2026-11-23)

- MLX computes physical log-binomial odds on the host in float64 and centers
  each conditional before sending it to the float32 correction network.
  Physical search draws use NumPy's multivariate hypergeometric distribution.
  The seeded receipts reproduced and agreed with Torch's physical log
  probabilities with known cards reserved, including the empty physical deal.
- At `a95bdce183`, the Torch-blocked local Etude run registered
  `belief-mlx/policy@0` update 1 with 246 learner transitions and SHA-256
  `25a5e7dcf51a28d03f67075352802fb4b4d918545855384c4eebe5f9e4640377`.
  The sampler histories named those policy bytes. `belief-mlx/sampler@0`
  registered its four-step checkpoint with SHA-256
  `16c5ae2e14ab3080eb2b825a6fda8dfe5592b285bfc38199bbd6b3117c7c2a2f`.
