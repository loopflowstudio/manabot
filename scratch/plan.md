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
4. **Done (`230c18b879`)** — Recent events and both decision-memory encoders run on MLX. GRU collection shares the lifetime-bound record cache with Torch. Live-token attention packs all object layers and bounds SDPA by the longest live sequence. MLX can branch from a published control with its saved column readings. Self-play admits live tokens on MLX CPU/MPS and Torch CUDA.

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

## Next — Apple Silicon gate

The local CPython 3.13 extension was rebuilt from this checkout after test
collection found the previous binary lacked `managym.SlotLoad`. The five new
architecture cases passed the existing forward, gradient and clipped Adam
tolerances. A bounded PPO update branched from a Torch control with legacy
readings, trained nonzero gates and reloaded them on MLX CPU and MPS.
The full gate and the captured parity check in `docs/development.md` remain.

```sh
uv run --extra mlx --extra dev pytest -q \
  tests/model/test_mlx_backend.py tests/belief \
  tests/training/test_update_backends.py tests/training/test_update_backend_protocols.py \
  tests/training/test_memory_chunk_cache.py tests/training/test_memory_prefix_storage.py \
  tests/training/test_memory_collection_fold.py tests/training/test_memory_learner_fold.py \
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
