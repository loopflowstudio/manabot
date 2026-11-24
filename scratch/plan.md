# 490ce6 — Complete remaining MLX policy architectures, belief and PPO parity

Run every architecture Torch's `ActorCritic` builds, both update rules,
live-token attention, and belief sampling and fitting through `resolve("mlx")`.
Torch remains the numerical reference. Preserve canonical checkpoint names
and use `manabot.policy.prepare.Prepared` as the only policy input.

The parent direction includes “use a Torch-free safetensors SamplerCodec on
MLX while keeping Torch .pt intact” (Task 490ce6 comment
`696d1890-ef51-43d9-8dd3-391c020a2a3c`). The portable-write design and
continued legacy Torch reads are recorded in `scratch/questions.md`.
Parent PR #585 has merged and its recovery, health, and local evaluation
work is in the checkout. Sync `474cf18dd7` also incorporated host observation
preparation from `b9b09bc6d9`. Extend the current admission and
`prepared_chunks` paths for optional inputs.

## Slices

1. **Done (`90211689b7`)** — Gray Ogre PPO runs on MLX through the shared update with `world=None`, hashed records, and historical mean. A Torch-blocked recipe registered a policy after one MPS Adam step and both backends reloaded it with identical outputs.
2. **Done (`ec62204d49`)** — Seven architecture cases matched Torch on forwards, gradients and clipped Adam weights. A collected WDL rollout trained through the shared categorical returns and the single `UpdateBackend` / `RL` surface.
3. **Next — portable belief sampler, belief rows, and Torch-free demo.**
   - Factor byte-level `encode_safetensors` and `decode_safetensors` from
     file I/O in `manabot/policy/checkpoint.py`. In
     `manabot/belief/checkpoint.py`, own the immutable `belief.safetensors`
     save and exact byte, schema, dataset, policy and world admission.
     Preserve the `manabot.autoregressive-belief-sampler/v1` format marker,
     schema and identities, hidden size, history dropout and metrics. The
     nine canonical float32 tensors are `context.{weight,bias}`, `card.weight`,
     `recurrent.{input_weight,hidden_weight,input_bias,hidden_bias}`, and
     `correction.{weight,bias}`.
   - Extend the existing `SamplerCodec` in `manabot/belief/protocol.py` to
     extract and restore canonical parameters. Delete `SamplerEncoder`
     from `BeliefTrainer`. Read legacy zip `.pt` by `PK\x03\x04` through
     the Torch codec and apply the same admission checks. MLX must name
     Torch when it refuses those bytes. Read `metrics.seed` in
     `manabot/reports/belief.py` through the admitted sampler payload so
     the report and loader use the same bytes. Preserve absent fit seeds.
   - Implement `mlx/belief.py` and `mlx/belief_fit.py` against Torch's
     physical-deal base measure and autoregressive sampler. Keep support,
     normalization and `log_prob` parity with Torch, including under MLX's
     GPU float64 limitation. Reuse host `score_sampler_arm`. Expose lazy
     `belief_fit` on both backends and `belief` on MLX.
   - In MLX policy and checkpoint loading, port the five belief embeddings
     for card, owner, zone, count and global context. Add `admit_belief`,
     read `Prepared.belief`, restore `belief_binding`, and remove the
     corresponding architecture rejection. Self-play still rejects positive
     `belief_count_buckets` because it produces no belief rows.
   - Add the backend setting to `manabot/recipes/belief_sampler.py` with
     Torch as its omitted default. Resolve it for collection, fitting and
     checkpoint I/O, and register `belief.safetensors`. Update the recipe
     consumer in `tests/training/test_recipe_etude.py` and qualify the
     linked commands below in a subprocess with Torch imports blocked.
     Add the belief row to the policy parity table and check sampler
     `log_prob` after saving on either backend and loading on the other.
     Retain a true legacy zip fixture for Torch admission and MLX refusal.
4. **Recent events, decision memory, live tokens, and docs.**
   - Port Torch's `RecentEventEncoder` over `Prepared.events` and include
     event validity in the context mask for stream normalization.
   - Port Torch's `DecisionMemory` over prepared prefixes for GRU and
     transformer with one and four summaries. Let `PackedGRU.run` start
     from supplied states for causal DAG fold levels. Implement the memory
     gate, projection, scale and `memory_gates()` on attention layers.
     Wire `CollectionMemory` through `PolicyInference` and
     `prepare_admitted` using `memory_roots`, keeping one draw per original
     row group and avoiding raw memory padding per row.
   - In `mlx/checkpoint.py`, map the `detail_map`, `reference_map`,
     `member_map` and `record_output` sequentials and split/concat
     `.self_attn.qkv_*` like `.mha.qkv_*`. Port
     `pretrained_memory_branch` warm-start from Torch's runtime.
   - Move the live-token device check from `SelfPlay.valid_settings` to
     `SelfPlayRuntime.admit_execution` before opening a run. Torch requires
     CUDA for self-play with live tokens but still supports CPU policy
     construction and checkpoint loading for reference checks. MLX admits
     CPU and MPS. Thread `live_tokens` and existing bucket options through
     MLX construction and checkpoint loading.
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

On Apple Silicon, run both jobs against a fresh experiment label. The
policy recipe's default is one PPO update. Etude delivers its final policy
to the sampler as `run.input("start")`. The sampler records
`belief.safetensors`. Verify lineage and both artifacts with Torch imports
blocked. The test can reuse the subprocess setup in
`tests/training/test_mlx_workflow.py`.

```sh
uv run --extra mlx etude train experiments/belief.py --experiment belief-mlx --job policy --config '{"backend":"mlx","device":"mps"}'
uv run --extra mlx etude train experiments/belief.py --experiment belief-mlx --job sampler --config '{"backend":"mlx"}'
```

- The forward, gradient and clipped Adam table in
  `tests/model/test_mlx_backend.py` covers WDL, all four value aggregations,
  attention off, recent events, belief rows, decision memory (both
  encoders and both summary counts), hashed records and `world=None`.
  Live tokens match MLX padded, Torch live on CPU, and Torch padded.
- Fixed-batch PPO `update_iteration` matches Torch on loss, per-tensor
  gradient cosine, post-step parameters, and diagnostic keys at the
  damped parity tolerances.
- The portable sampler loads across backends in both directions with
  matching `log_prob`. Legacy Torch zip samplers remain readable on Torch
  and MLX refusal names Torch. Both retired architecture-refusal tests
  are gone. Input-version refusal and error context still work.

Run the Apple Silicon gate with the rebuilt engine.

```sh
uv run --extra mlx --extra dev pytest tests/model/test_mlx_backend.py tests/training/test_update_backends.py tests/training/test_update_backend_protocols.py tests/belief tests/training/test_mlx_workflow.py tests/training/test_recipe_etude.py tests/sim/test_local_update.py tests/sim/test_local_update_sampling.py tests/test_package_boundaries.py -q
```

Slice 2 recorded 29 focused passes before sync. After sync, the rebuilt
engine passed the value-token case and Gray Ogre PPO parity individually.
The earlier broad MLX file run during sync did not finish with a passing
result. Its crash context was not captured in full. Gate should run the
whole file against the rebuilt engine and retain the first failing node
if the problem recurs.

Scope ends at numerical parity. Torch resume, Etude managed evaluation,
further speed optimization and claims about playing strength are excluded.
