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

## Next — Apple Silicon gate

Run the full backend suite and the captured parity check required by
[`docs/development.md`](../docs/development.md#validation-by-change-area).
The local CPython 3.13 extension was rebuilt from this checkout after the
previous binary lacked `managym.SlotLoad`. The earlier broad MLX file run
during sync did not finish with a passing result. Use verbose output to
retain the first failing test or the last test started if the process exits.

```sh
uv run --extra mlx --extra dev --extra artifacts pytest -v \
  tests/model/test_mlx_backend.py tests/belief \
  tests/training/test_update_backends.py tests/training/test_update_backend_protocols.py \
  tests/model/test_torch_backend_protocol.py tests/model/test_torch_container_load.py \
  tests/model/test_prepare_rows.py \
  tests/training/test_memory_chunk_cache.py tests/training/test_memory_prefix_storage.py \
  tests/training/test_memory_collection_fold.py tests/training/test_memory_learner_fold.py \
  tests/training/test_mlx_workflow.py tests/training/test_recipe_etude.py \
  tests/sim/test_local_update.py tests/sim/test_local_update_sampling.py \
  tests/test_package_boundaries.py
```

The artifact manifest lists the retained MLX pilot parity subset. Fetch it
and check its world admission before using it as gate evidence. If admission
fails, use a bounded current-world capture through
`manabot/backends/torch/capture.py` and record the old capture's refusal.

```sh
doppler run --project manabot --config prd -- uv run --extra artifacts \
  python -m manabot.data fetch experiments/data/mlx_pilot/capture
uv run --extra mlx python -m manabot.backends.parity \
  experiments/data/mlx_pilot/capture/initial.pt --capture experiments/data/mlx_pilot/capture
```

Scope ends at numerical parity. Torch resume, Etude managed evaluation,
further speed optimization and claims about playing strength are excluded.
