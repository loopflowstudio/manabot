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
evaluation. Host observation preparation (`b9b09bc6d9`) and native snapshot
rollouts (`82c5d1d1b7`) are now in the checkout. Extend the shared admission,
`prepared_chunks` and retained-prefix paths. Native restores fork the
decision-log lineage, so serving cache eligibility needs the viewer prefix
identity and native epoch even when branches share the same record nodes.

## Slices

1. **Done (`90211689b7`)** — Gray Ogre PPO runs on MLX through the shared update with `world=None`, hashed records, and historical mean. A Torch-blocked recipe registered a policy after one MPS Adam step and both backends reloaded it with identical outputs.
2. **Done (`ec62204d49`)** — Seven architecture cases matched Torch on forwards, gradients and clipped Adam weights. A collected WDL rollout trained through the shared categorical returns and the single `UpdateBackend` / `RL` surface.
3. **Done (`a95bdce183`)** — Ten focused tests passed, including belief-row forward, gradient and Adam parity, portable sampler reads across backends and legacy Torch zip reads. The linked Torch-blocked Etude jobs registered the PPO policy and its bound sampler.
4. **Next — recent events, decision memory, live tokens, and docs.**
   - Port Torch's `RecentEventEncoder` over `Prepared.events` and include
     event validity in the context mask for stream normalization.
   - Port Torch's `DecisionMemory` over prepared prefixes for GRU and
     transformer with one and four summaries. Let `PackedGRU.run` start
     from supplied states for causal DAG fold levels. Implement the memory
     gate, projection, scale and `memory_gates()` on attention layers.
     Wire `memory_roots` through `PolicyInference` and `prepare_admitted`.
     Share `CollectionMemory`'s record traversal and lifetime rules for GRU
     serving, with backend-specific tensor operations. Keep learner folds
     differentiable, cached serving roots detached and scoped to frozen
     weights, and the current last record live. Use retained prefixes without
     per-row raw padding and keep one draw per original row group.
   - In `mlx/checkpoint.py`, map the `detail_map`, `reference_map`,
     `member_map` and `record_output` sequentials and split/concat
     `.self_attn.qkv_*` like `.mha.qkv_*`. Use the shared
     `pretrained_memory_branch` check in MLX. Build a memory branch with the
     published policy's readings and copy the exact common tensors, keeping
     the newly initialized memory encoder, gates and projections.
   - Move the live-token device check from `SelfPlay.valid_settings` to
     `SelfPlayRuntime.admit_execution` before opening a run. Torch requires
     CUDA for self-play with live tokens but still supports CPU policy
     construction and checkpoint loading for reference checks. MLX admits
     CPU and MPS. Thread `live_tokens` and existing bucket options through
     MLX construction and checkpoint loading. The tracked loop already
     receives a run from `track`. Check at the recipe boundary before
     `track` and cover `self_play` and `fresh_iteration` before they resume
     state or create output.
   - In MLX attention, gather valid objects and any neutral critic token
     with `Prepared.live_index`. Keep gates, norms, projections and MLPs
     on the packed rows. Derive sequence lengths, scatter positions and
     the additive key mask on the host from `Prepared.live_offsets`.
     Around fused SDPA, scatter Q/K/V to `[B, H, max_live, d]`, gather
     packed results, and restore original object coordinates at the end.
     The attention bound is `O(B * max_live^2)`, with no device scalar
     read and no batch-wide `[T, T]` mask.
   - Remove the remaining architecture rejections. Keep input versions
     9–11 and existing input and checkpoint admission. Reconcile
     `manabot/backends/mlx.md`, `manabot/backends/README.md`,
     `docs/observation-and-model.md`, and `docs/training.md` with the
     verified entry points and limits.

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

Rebuild the Apple Silicon engine for `82c5d1d1b7` before the gate so native
snapshot and vector bindings match the checkout.

```sh
uv run --extra mlx --extra dev pytest -q \
  tests/model/test_mlx_backend.py tests/belief \
  tests/training/test_update_backends.py tests/training/test_update_backend_protocols.py \
  tests/training/test_memory_chunk_cache.py tests/training/test_memory_prefix_storage.py \
  tests/training/test_mlx_workflow.py tests/training/test_recipe_etude.py \
  tests/sim/test_local_update.py tests/sim/test_local_update_sampling.py \
  tests/test_package_boundaries.py
```

Slice 2 recorded 29 focused passes before sync. After sync, the rebuilt
engine passed the value-token case and Gray Ogre PPO parity individually.
The earlier broad MLX file run during sync did not finish with a passing
result. Its crash context was not captured in full. Gate should run the
whole file against the rebuilt engine and retain the first failing node
if the problem recurs.

Scope ends at numerical parity. Torch resume, Etude managed evaluation,
further speed optimization and claims about playing strength are excluded.
