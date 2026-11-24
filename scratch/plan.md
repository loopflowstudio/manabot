# 490ce6 — Complete remaining MLX policy architectures, belief and PPO parity

Stacked on dba0f2 (PR #585: exact MLX recovery, unified `StepRecorder` /
sampled health, MLX local eval). `lf sync dba0f2` ran at kickoff; nothing in
this plan duplicates #585. If #585 merges first, `lf sync main`.

## Outcome

Every `ModelConfig` the Torch `ActorCritic` builds, both update rules
(`damped_policy` and `ppo`), live-token attention, and belief sampling/fitting
run on `resolve("mlx")` with Torch as the numerical reference — behind **one**
uniform `UpdateBackend` surface and **one** portable safetensors container for
both policies and belief samplers.

On Apple Silicon, a person can train the belief pilot end to end without
importing Torch:

```sh
uv run --extra mlx etude train manabot/recipes/belief_policy.py --config '{"backend":"mlx","device":"mps"}' --until updates=1
uv run --extra mlx etude train manabot/recipes/belief_sampler.py --config '{"backend":"mlx"}'
```

The first trains Gray Ogre PPO (default `ModelConfig`: `world=None`, hashed
records, `historical_mean`) on MLX and exports a portable policy; the second
collects histories with it and records `belief.safetensors`, a portable sampler
checkpoint admitted by bytes, schema and world in either MLX or Torch.
`require_supported` in `manabot/backends/mlx/checkpoint.py` no longer refuses
any architecture; only the input-version check (9–11) remains.

Done when:

- the demo above runs with `torch` absent from `sys.modules` (asserted by a
  test running the recipes' train functions in a subprocess with Torch imports
  blocked, as #585 does for local eval);
- `tests/model/test_mlx_backend.py`'s forward/gradient/Adam parity test is a
  table over architecture variants and passes for every row: `categorical_wdl`,
  each `value_aggregation` (`masked_mean`, `historical_mean`, `value_token`,
  `attention_readout`), `attention_on=False`, `recent_events`, belief rows,
  decision memory (`gru` and `transformer`; summaries 1 and 4), hashed and
  `world=None`, and live tokens (MLX-live == MLX-padded and MLX-live ==
  Torch-live CPU);
- PPO `update_iteration` on MLX matches Torch on one fixed batch (loss,
  per-tensor gradient cosine, post-step parameters, diagnostics keys) at the
  same tolerances as the damped parity check;
- a belief sampler saved by Torch loads in MLX (and vice versa) with matching
  `log_prob`, while legacy `.pt` samplers still load in Torch and are cleanly
  refused by MLX naming Torch;
- `test_unimplemented_layouts_are_refused_by_name` and
  `test_a_backend_without_a_categorical_critic_is_refused` are deleted, not
  edited.

## Source of truth

- Architecture: Torch `ActorCritic` (`manabot/backends/torch/policy.py` and its
  private modules). MLX modules mirror it one to one so canonical parameter
  names map through `checkpoint.py` unchanged (`mlx.md` "Layout mapping"). New
  Torch-side names get MLX mappings, never renames.
- Host input: `manabot.policy.prepare.Prepared` is the only policy input.
  Belief rows (`Prepared.belief`), decision-memory prefixes, recent-event rows,
  hashed/catalog routes and `live_index`/`live_offsets` are already
  host-prepared; MLX adds no second preparation.
- Updates are written once over a single `UpdateBackend`
  (`manabot/training/updates/backend.py`), with no capability tiers or
  `isinstance` refusal gates once MLX supports both rules and both critics.
- Shared host algorithms (`transition_gae`, `categorical_lambda_returns`,
  `selected_magnitudes` in `manabot/policy/returns.py`; `AdmittedRows`
  admission, `PrefixObservation` trim, and one-ahead `prefetch_minibatches` in
  `manabot/training/rollout/admission.py`; safetensors bytes codecs in
  `manabot/policy/checkpoint.py`) live once in framework-free modules.

## Choices made for review (concept review, 2026-11-23)

1. **Portable `belief.safetensors` container on both backends** (passed over:
   non-portable `.pt` on Torch vs `.safetensors` on MLX). Zero frozen
   `belief.pt` artifacts or hardcoded sampler digests exist in the repository.
   One portable container matches policy checkpoints, enables cross-backend
   Torch↔MLX sampler loading and parity tests, and removes the split
   `SamplerEncoder`/`SamplerCodec` half-codecs. Torch keeps legacy `PK\x03\x04`
   `.pt` read fallback by magic bytes.
2. **Collapse update protocol tiers into one `UpdateBackend` / `RL`** (passed
   over: keeping `PPOBackend`, `PPORL`, `CategoricalRL`, `ppo_rl`,
   `categorical_rl`). Once both backends implement PPO and categorical WDL,
   those 7 sub-protocols and 2 runtime `isinstance` gates distinguish nothing.
3. **Runtime device admission for `live_tokens` + `O(B * max_live^2)` compacted
   SDPA in MLX** (passed over: refusing off-CUDA in `ActorCritic.__init__`,
   which breaks `load_checkpoint(..., live_tokens=True)` and 6 CPU test suites;
   and `[T,T]` block-diagonal masking in MLX, which is `O(B^2 * L^2)`).

## Design by area

### 1. Remaining protocol collapse and shared returns

Slice 1 rewrote PPO on the damped rule's shape, moved the `PrefixObservation`
trim and one-ahead `prefetch_minibatches` into `AdmittedRows`, folded
`standardized_advantages`/`minibatches` into `RL`/`AdmittedRollout` and deleted
the `PPO*` protocols and `ppo_rl()`. What remains:

- move `categorical_lambda_returns` (the `reverse_waves` / `bootstrap_cuts`
  recursion in `manabot/backends/torch/rl.py`) into `manabot/policy/returns.py`
  in NumPy; both `torch/rl.py` and `mlx/rl.py` wrap it thinly;
- after MLX gains categorical WDL, fold `CategoricalRL` into `RL` and delete
  `CategoricalRL`, `categorical_rl()` (callers in `updates/targets.py`) and
  `test_a_backend_without_a_categorical_critic_is_refused`.

### 2. Live tokens: runtime admission and `O(B * max_live^2)` MLX attention

- **Execution admission (`SelfPlay` + `backend.runtime`)**:
  - remove `if self.live_tokens and self.device != "cuda"` from
    `SelfPlay.valid_settings` (`manabot/training/loops/self_play.py`);
  - add `admit_execution(device: str, precision: Precision, live_tokens: bool) -> None`
    on `SelfPlayRuntime` (checked in `self_play` / `create` before opening a
    run): Torch refuses `live_tokens=True` when `device != "cuda"`
    (`"live tokens require CUDA on the torch backend"`), leaving
    `ActorCritic(..., live_tokens=True)` and `load_checkpoint(..., live_tokens=True)`
    on CPU intact for `.to("cuda")` and all existing CPU live-token tests; MLX
    accepts `live_tokens=True` on both `mps` and `cpu`.
  - Update `MLXBackend.load_policy`, `MLXBackend.load`, `mlx/checkpoint.py:assemble`,
    and `mlx/runtime.py:create_policy` to accept and forward `live_tokens`
    (and `bucketed_shapes` on `load`).
- **MLX packed attention (`manabot/backends/mlx/policy.py`)**:
  - mirror Torch's `_attend_live` and `forward_live`: gather present objects
    (plus optional neutral `value_token`) using `Prepared.live_index` into
    packed `[L, D]`. Run `memory_gate`, `perspective`, `norm1`,
    `query_proj`/`key_proj`/`value_proj`, `out_proj`, `norm2`, `mlp`, and
    `final_norm` on `[L, D]` with zero absent-slot compute;
  - around `mx.fast.scaled_dot_product_attention` only, compute on host from
    `Prepared.live_offsets` (zero device sync): `lengths = np.diff(offsets)`,
    `max_live = int(lengths.max())`, flat scatter positions into
    `[B * max_live]`, and `[B, 1, 1, max_live]` additive key mask (`0` /
    `-inf`). Scatter packed Q/K/V `[L, H, d]` to `[B, H, max_live, d]`, run
    fused SDPA, and gather back to `[L, H, d]`. Complexity is
    `O(B * max_live^2)` (linear in `B`, strictly <= padded `O(B * N^2)`).
    Scatter final `[L, D]` back to `[B, S, D]` with zero padding.
- **Qualification gate**: parity row checks MLX-live == MLX-padded, MLX-live ==
  Torch-live (CPU), and MLX-live == Torch-padded.

### 3. Complete MLX policy, collection, runtime, and checkpoint parity

Every remaining `ModelConfig` variant mirrors Torch `ActorCritic` in
`manabot/backends/mlx/policy.py`, closing all hidden scalar/catalog assumptions
outside `require_supported`:

- **`categorical_wdl`**:
  - 3-logit `value_head` output; set `self.value_semantic = "signed_outcome"`
    for `categorical_wdl` (else `None`).
  - Match Torch's split between `forward_distribution(obs)` (raw `[B,1]` or
    `[B,3]` logits) and `__call__(obs)` (expected signed outcome `[B]` via
    `P(win) - P(loss)` for `categorical_wdl`, `[B]` scalar otherwise). Update
    `evaluate`, `evaluate_probabilities`, `win_probabilities`, and
    `SelectionPredictor` to consume `__call__`'s `[B]` values.
  - Update `manabot/backends/mlx/collection.py` (`CollectionScope.sample`) to
    compute `outcomes = softmax(value)`, `value = outcomes[:, 2] - outcomes[:, 0]`,
    and return `outcome_probabilities` when `value_kind == "categorical_wdl"`,
    mirroring `torch/collection.py:118-121`.
  - Add `expected_outcome`, `outcome_probabilities`, `categorical_lambda_returns`
    (wrapping `policy/returns.py`), `categorical_value_loss`, `selected_rows`,
    `selected_moves`, and `magnitude_quantile` to `manabot/backends/mlx/rl.py`.
  - Update `manabot/backends/parity.py:minibatch_loss` to set
    `categorical=policy.config.value_kind == "categorical_wdl"`.
- **`value_aggregation`** (`historical_mean` landed in Slice 1):
  - `value_token` (learned trailing token through attention; neutral perspective
    and zero memory gate), `attention_readout` (`AttentionReadout` with one
    learned query + `mha`, built last as in Torch).
- **`attention_on=False`**:
  - omit `self.attention` and `self.extra_attention`; skip attention and column
    trimming while preserving `final_norm` and `value_token` behavior.
- **`recent_events`**:
  - port `torch/_recent_events.py` (`RecentEventEncoder`) over `Prepared.events`;
    include `events.validity.any(-1)` in `stream_norm` context presence mask.
- **Belief rows**:
  - port the five belief embeddings (`belief_card_embedding`,
    `belief_owner_embedding`, `belief_zone_embedding`,
    `belief_count_embedding`, `belief_global_embedding`), `admit_belief`
    (`BeliefPolicy` conformance), and restore `policy.belief_binding` in
    `MLXBackend.load_policy`.
- **Decision memory**:
  - extend `PackedGRU.run` with optional `initial: mx.array | None = None` for
    causal DAG fold levels;
  - port `torch/_decision_memory.py` (`DecisionMemory` for both `gru` and
    `transformer`, 1 or 4 summaries, `memory_gate` + `memory_projection` +
    `memory_gate_scale` on attention layers, real `memory_gates()`, and
    `CollectionMemory` wired into `PolicyInference` and `prepare_admitted`
    with `memory_roots`); MLX collection keeps one sampling call per row group
    and never pads raw memory per row;
  - extend `mlx/checkpoint.py`: add `detail_map|reference_map|member_map|record_output`
    to `_SEQUENTIAL` / `_MLX_SEQUENTIAL`, and split/concat `.self_attn.qkv_*`
    alongside `.mha.qkv_*` (matching `manabot/policy/checkpoint.py`);
  - add `pretrained_memory_branch` warm-start support in `mlx/runtime.py`
    matching `torch/runtime.py:95-120`.

### 4. Belief: one portable container (`belief.safetensors`), unified codec, MLX sampler

- **Recipes**:
  - `manabot/recipes/belief_policy.py`: `PolicyConfig` inherits from
    `SelfPlayRecipe` (`manabot/recipes/builders.py`), gaining `backend` (with
    `exclude_if` default `"torch"`) and `mlx` != `cuda` validation without
    duplication; `train` resolves `config.backend` and passes `backend=` to
    `self_play`.
  - `manabot/recipes/belief_sampler.py`: `BeliefConfig` adds
    `backend: BackendName = Field(default="torch", exclude_if=lambda v: v == "torch")`;
    `train` resolves `config.backend`, writes `run.out / "belief.safetensors"`,
    and passes `backend=backend.belief_fit` and `codec=backend.belief` (no
    hard `manabot.backends.torch.belief_fit` import).
- **Framework-free safetensors bytes helpers (`manabot/policy/checkpoint.py`)**:
  - factor `encode_safetensors(arrays, metadata) -> bytes` and
    `decode_safetensors(data) -> tuple[dict[str, np.ndarray], dict[str, str]]`
    from `write_safetensors` / `read_safetensors` (zero new dependencies).
- **Unified portable sampler container (`manabot/belief/checkpoint.py`, `protocol.py`, `sampling_fit.py`)**:
  - `manabot/belief/checkpoint.py` owns encoding and decoding the portable
    `belief.safetensors` container: JSON metadata holds `format`
    (`"manabot.autoregressive-belief-sampler/v1"`), `schema`,
    `schema_identity`, `dataset_identity`, `policy_identity`,
    `world_identity`, `hidden_size`, `history_dropout`, and `metrics`;
    tensor payload holds the 9 float32 parameters under canonical names
    (`context.{weight,bias}`, `card.weight`,
    `recurrent.{input_weight,hidden_weight,input_bias,hidden_bias}`,
    `correction.{weight,bias}`).
  - `SamplerCodec` (`manabot/belief/protocol.py`, implemented by both
    `backend.belief` modules) provides `sampler_parameters(model) -> dict[str, np.ndarray]`,
    `restore_sampler(schema, hidden_size, history_dropout, parameters)`, and
    optional `decode_legacy_sampler(data)` (Torch decodes `PK\x03\x04` `.pt`
    bytes and maps `state_dict` to canonical names; MLX raises `ValueError`
    naming the Torch backend when given `PK\x03\x04` bytes).
  - Delete `SamplerEncoder` from `BeliefTrainer`; `save_belief_sampler` and
    `load_belief_sampler` both take `codec: SamplerCodec` (`backend.belief`).
  - Expose lazy `belief_fit` property on both `TorchBackend` and `MLXBackend`
    (and `belief` on `MLXBackend`).
  - Update `manabot/reports/belief.py:185` to read `metrics.seed` via
    `read_sampler_metadata` / `decode_sampler_payload` instead of `torch.load`.
- **MLX belief implementation**:
  - `manabot/backends/mlx/belief.py`: `physical_deal_log_prob`,
    `sample_physical_deal`, `AutoregressiveBeliefSampler` (same architecture
    and physical-deal base measure as `torch/belief.py`; note MLX GPU has no
    float64 so base logits/log_prob run on CPU or float32 — verify exact
    parity with Torch `log_prob`), `PhysicalDistribution`,
    `sampler_parameters`, `restore_sampler`, `freeze_sampler`, and legacy `.pt`
    refusal naming Torch.
  - `manabot/backends/mlx/belief_fit.py`: `evaluate_physical_sampler`,
    `evaluate_sampler`, `fit_belief_sampler` (mirroring `torch/belief_fit.py`,
    reusing host `score_sampler_arm`). Register `"belief"` and `"belief_fit"`
    in `manabot/backends/mlx/__init__.py`.

## Absent and error states

- A checkpoint whose architecture MLX now supports loads; an input version
  outside 9–11 still raises `UnsupportedPolicyInput`/`UnsupportedPolicy`.
- `live_tokens=True` with Torch off CUDA fails at self-play runtime admission
  (`backend.runtime.admit_execution`) naming CUDA and Torch — never at
  `ActorCritic.__init__` (CPU tests and `load_checkpoint` keep working) and
  never at `SelfPlay.valid_settings`; with MLX it runs on both `mps` and `cpu`.
- A legacy Torch sampler `.pt` (`PK\x03\x04`) handed to MLX fails with a
  message naming the Torch backend (never a parse error); handed to Torch it
  decodes via the legacy fallback and enforces the exact byte/schema/world
  checks. A `belief.safetensors` loads cleanly on either backend.
- `belief_count_buckets > 0` in self-play still refuses (self-play produces no
  belief inputs) — unchanged.

## Exclusions

- MLX speed optimization beyond avoiding the `O(B^2 * L^2)` live-token bug and
  retaining packed/chunked execution; parity is numerical only, not a
  learning-quality claim.
- Torch resume, evaluation via Etude managed agents, anything in #585.

## What goes

- `CategoricalRL`, `categorical_rl()`,
  `test_a_backend_without_a_categorical_critic_is_refused` (Slice 2).
- Torch `rl.categorical_lambda_returns` duplicate wave recursion (moves into
  `manabot/policy/returns.py`).
- `SelfPlay.valid_settings`' hardcoded live-token CUDA check (moves to
  `backend.runtime.admit_execution`).
- `SamplerEncoder` on `BeliefTrainer`, the hard `manabot.backends.torch.belief_fit`
  import in `manabot/recipes/belief_sampler.py`, and raw `torch.load` in
  `manabot/reports/belief.py`.
- Each `require_supported` reason as its slice lands, and
  `test_unimplemented_layouts_are_refused_by_name`.
- Docs: `manabot/backends/mlx.md`, `manabot/backends/README.md`,
  `docs/observation-and-model.md`, and `docs/training.md` updated to reflect
  full MLX parity, runtime live-token admission, and `belief.safetensors`.

## Slices

1. **Done (90211689b7)** — Gray Ogre PPO on MLX through the shared update,
   `world=None`/hashed records, `historical_mean`; the Torch-blocked recipe
   test registered a portable checkpoint after one MPS Adam step, and both
   backends reloaded it with identical logits and value.
2. **Next — critic and attention variants, then final protocol collapse**:
   - Move `categorical_lambda_returns` to `manabot/policy/returns.py`; add
     categorical WDL + selection helpers to `mlx/rl.py`.
   - Separate `MLXActorCritic.forward_distribution` vs `__call__`, fix
     `evaluate`/`evaluate_probabilities`/`win_probabilities`/`value_semantic`,
     add `categorical_wdl` handling to `mlx/collection.py` and
     `backends/parity.py:minibatch_loss`.
   - Fold `CategoricalRL` into `RL`; delete `CategoricalRL`, `categorical_rl()`,
     and `test_a_backend_without_a_categorical_critic_is_refused`.
   - Implement `value_token`, `attention_readout` (built last), and
     `attention_on=False` in `mlx/policy.py`.
   - Convert `test_mlx_backend.py` parity test into a parametrized table over
     architecture variants.
3. **Portable belief sampler container, belief rows, and Torch-free demo**:
   - Factor `encode_safetensors` / `decode_safetensors` in
     `manabot/policy/checkpoint.py`.
   - Unify `belief.safetensors` in `manabot/belief/checkpoint.py` with
     `SamplerCodec` on `backend.belief`, lazy `backend.belief_fit` on both
     backends, legacy `.pt` read fallback on Torch, and `reports/belief.py`
     updated off raw `torch.load`.
   - Implement belief rows, `admit_belief` (`BeliefPolicy`), and
     `belief_binding` restoration in MLX; implement `mlx/belief.py` and
     `mlx/belief_fit.py`; wire `belief_sampler.py`.
   - Add cross-backend Torch↔MLX sampler parity test, legacy `.pt` refusal
     test on MLX, and the Torch-blocked subprocess belief demo test.
4. **Recent events, `O(B * max_live^2)` live tokens, decision memory, docs**:
   - Move `live_tokens` off-CUDA check from `SelfPlay.valid_settings` to
     `backend.runtime.admit_execution`; implement `O(B * max_live^2)` compacted
     SDPA live-token path in `mlx/policy.py` and qualify against both MLX-padded
     and Torch-live (CPU).
   - Port `recent_events` and `decision_memory` (`gru` with `initial` state on
     `PackedGRU.run`, `CollectionMemory` in `PolicyInference`, `memory_gates`,
     `pretrained_memory_branch`, and `transformer` with `.self_attn.qkv_*` and
     `_SEQUENTIAL` mappings).
   - Remove remaining `require_supported` architecture checks and delete
     `test_unimplemented_layouts_are_refused_by_name`.
   - Update `mlx.md`, `backends/README.md`, `docs/observation-and-model.md`,
     and `docs/training.md`.

## Acceptance check at gate

```sh
uv run --extra mlx --extra dev pytest tests/model/test_mlx_backend.py tests/training/test_update_backends.py tests/training/test_update_backend_protocols.py tests/belief tests/training/test_mlx_workflow.py tests/training/test_recipe_etude.py tests/sim/test_local_update.py tests/sim/test_local_update_sampling.py tests/test_package_boundaries.py -q
```

All pass, the parity table has every architecture row above, cross-backend
sampler parity passes, and the Torch-blocked belief demo test passes.

## Open choice for Slice 3's demo

`manabot/recipes/belief_sampler.py` reads `run.input("start")`, supplied only
by a `TrainingJob(..., start=policy)` declaration (`experiments/belief.py`), so
the Outcome's standalone `etude train .../belief_sampler.py` line cannot run as
written. Options: (a) run `experiments/belief.py` per job with `--config`;
(b) keep the sampler two-job-only and show `collect_belief` + `train_belief`
in Python (what the docs show); (c) add an optional start-checkpoint path to
`BeliefConfig`. Slice 3 takes (a) if `etude train --job` accepts `--config`,
else (b), and records the choice in `scratch/questions.md`.
