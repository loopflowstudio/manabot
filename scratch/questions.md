# Choices made without review

## Kickoff (2026-11-23)

- Stacked on dba0f2 (`lf sync dba0f2`) because PR #585 rewrites the MLX
  learner, checkpoint and update protocol files this Task also changes.

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
