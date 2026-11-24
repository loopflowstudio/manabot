# 490ce6 — Complete remaining MLX policy architectures, belief and PPO parity

Run every architecture Torch's `ActorCritic` builds, both update rules,
live-token attention, and belief sampling and fitting through `resolve("mlx")`.
Torch remains the numerical reference. Preserve canonical checkpoint names
and use `manabot.policy.prepare.Prepared` as the only policy input.

The parent direction includes “Move live_tokens CUDA gate from
SelfPlay.valid_settings to backend admission and implement/qualify MLX packed
live-token execution over Prepared.live_index/live_offsets” and “use a
Torch-free safetensors SamplerCodec on MLX while keeping Torch .pt intact”
(Task 490ce6 comment `696d1890-ef51-43d9-8dd3-391c020a2a3c`). Both backends
write portable samplers by design. Torch retains legacy zip `.pt` reads. See
`scratch/questions.md` for that choice and the linked run evidence.

Parent PR #585 supplied exact MLX recovery, sampled health and local
evaluation. Use the shared admission and host preparation paths
(`b9b09bc6d9`). Native restores fork the decision-log lineage
(`82c5d1d1b7`), so the shared serving cache binds roots to both the viewer
prefix identity and native epoch even when branches share record nodes.

## Slices

1. **Done (`90211689b7`)** — Shared PPO update, Gray Ogre parity, and a Torch-blocked MPS policy registration.
2. **Done (`ec62204d49`)** — Remaining critic readouts, attention off, and shared categorical returns with seven architecture parity cases.
3. **Done (`a95bdce183`)** — Belief-row parity, portable samplers, legacy Torch reads, and linked Torch-blocked policy and sampler jobs.
4. **Done (`230c18b879`, evidence in `dc864d6c9f`)** — History encoders, shared GRU serving cache, packed live-token attention, backend device admission and memory warm starts.
5. **Done (`80040af539`)** — `Prepared` owns the compact attention coordinates and key mask. On Apple Silicon, 36 focused tests passed, including unequal-length parity and CPU and MPS warm starts.

## Acceptance

- The forward, gradient and clipped Adam table in
  `tests/model/test_mlx_backend.py` covers WDL, all four value aggregations,
  attention off, recent events, belief rows, decision memory (both
  encoders and both summary counts), hashed records and `world=None`.
  Live tokens match MLX padded, Torch live on CPU, and Torch padded. Use
  unequal live sequence lengths and include the neutral critic token and
  active memory gates in the comparison.
- A memory branch loads from a compatible control checkpoint and a bounded
  MLX self-play update exercises admitted memory prefixes and live-token
  collection. Cached GRU results match recomputed results across distinct
  native lifetimes with shared prefix content.
- Fixed-batch PPO `update_iteration` matches Torch on loss, per-tensor
  gradient cosine, post-step parameters, and diagnostic keys at the
  damped parity tolerances.
- The portable sampler loads across backends in both directions with
  matching `log_prob`. Legacy Torch zip samplers remain readable on Torch
  and MLX refusal names Torch. `tests/training/test_mlx_workflow.py` checks
  linked artifact admission and lineage with Torch imports blocked.
- The remaining architecture refusals are removed. Input-version refusal and
  checkpoint error context still work.

## Apple Silicon gate (run 2026-11-23 on a shared M4 Max, load average 65–105)

- Backend suite: 355 passed, 7 failed. Five `tests/belief` failures
  (`test_belief_agent`, `test_learning`, `test_player` ×2, `test_state`) fail
  identically at the merge base `82c5d1d1b7`. The two
  `test_mlx_local_evaluation_and_verification` cases ran out of game
  wall-clock budget under the load and failed the same way at the merge
  base; `[64]` passed earlier the same evening at a load near 30. Rerun both
  on a quiet Mac before landing.
- `tests/test_package_boundaries.py` needed `manabot/recipes/belief_policy.py`
  among the backend entry points; fixed.
- The retained `mlx_pilot` capture is refused (`missing writer catalog`,
  written at `b62ac3127`). A current-world capture (w64, 2 attention layers,
  4 minibatches) passes the float32 MLX parity command; bfloat16 at that
  width reaches a full-gradient cosine of 0.988 against 0.99, as
  `manabot/backends/mlx.md` already reports for w64.
- Driven by hand, all on MLX: tracked local train and resume, a resume of a
  save written by the merge base, linked belief jobs without Torch, sampler
  loads in both directions, four non-default models on Metal and CPU,
  memory warm starts and resume, and local evaluation of a WDL, value-token,
  GRU-memory policy. `scratch/qa.html` has the transcripts.

Scope ends at numerical parity. Torch resume, Etude managed evaluation,
further speed optimization and claims about playing strength are excluded.
