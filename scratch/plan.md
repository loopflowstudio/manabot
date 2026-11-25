# MTG-433: one policy input, one transport

Main supports policy input v11 only, and an observation reaches a policy only
as a native `managym.ObservationBatch` (or an `ObservationSelection` referencing
one or more native batches). Inputs v9/v10, the dense dictionary transport, and
the dict-masquerading wrappers built to bridge them (`ObservationSelection` as a
`Mapping`, `PrefixObservation`, `CompactObservation`, `select_rows`) are deleted.

## User-visible outcome

- `ObservationConfig(policy_input_version=9 | 10)`, a v9/v10 checkpoint, replay
  or saved regime raise `UnsupportedPolicyInput`: "… v0.2.5 is the last release
  supporting v9-v10, and ac601834d is the last revision supporting v1-v8".
  An omitted version still means v1 in `from_checkpoint`.
- `managym` no longer exports `encode_observation_into`,
  `encode_observations_into`, `input_v10_state`, `legacy_card_lists`,
  `Env.encode_observation(_into)`, `VectorEnv.set_buffers` /
  `write_observations_for_players`, `RolloutPool.set_buffers` / `encode_active` /
  `step_active`, `ObservationLayout.allocate` / `axis_extents` /
  `object_references`, or `observation_layout(legacy=…, turn_state=…)`.
  `ObservationLayout(policy_input_version=…)` accepts only 11.
- `managym.ObservationBatch` adds validated `to_bytes()` / `from_bytes(data)` so
  offline backend captures, parity checks, and probes persist and reload native
  CSR batches through the one admission/preparation path without breaking the
  CSR structural/join certificate invariant.
- `manabot.env.env.Env.reset/step/step_semantic/scenario_refresh` return the
  native viewer `managym.Observation` directly. Policies encode at the point
  they prepare: `managym.encode_observation_batch([obs], space.layout, overflow)`.
- Training, Academy, evaluation, search, belief, probes, and backend parity
  behave as before on v11. `tests/model/test_pinned_pilot.py` still replays
  `v11/pilot.safetensors` with its recorded action ranks and probability tolerance.

## Experience and data model across layers

One representation owns each stage; nothing re-encodes into or masquerades as a
dense dictionary:

1. **Viewer observation (`managym.Observation`)**: returned by `managym.Env` and
   `manabot.env.env.Env` (`reset`, `step`, `step_semantic`, `scenario_refresh`).
   Scripted, heuristic, random, LLM, and search players consume it directly with
   zero tensor encoding. `Env.last_raw_obs` remains as an alias to avoid
   unrelated churn across ~35 test/verify readers.
2. **Native encoded batch (`managym.ObservationBatch`)**: produced only by
   `managym.encode_observation_batch`, `VectorEnv.observation_rows()` /
   `observations_for_players()`, `RolloutPool.encode_active_batch()` /
   `step_active_batch()`, and validated `ObservationBatch.from_bytes(data)`.
   Every `ObservationBatch` in Python certifies its layout (v11), CSR row
   offsets, capacities, and native joins. Live encoding pays zero validation
   cost; `from_bytes` validates once in Rust on load.
   - Decision-memory blanking ablation uses `batch.blank_decision_memory()`
     everywhere (`eval/agents.py`, `sim/search/flat_mc.py`,
     `backends/torch/verify_decision_memory.py`); `manabot.env.decision_memory.blank_memory`
     is deleted.
3. **Row selection & interned memory (`ObservationSelection`, `RetainedMemory`)**:
   - `RetainedMemory(prefixes: tuple[Prefix, ...], capacities: MemoryCapacities)`
     replaces `PrefixObservation(dict[str, np.ndarray])` and `CompactObservation`.
     It is a plain frozen dataclass with capacity invariants and `.select(indexes)`.
   - `ObservationSelection(space, coordinates, memory: RetainedMemory | None = None)`
     drops `Mapping[str, np.ndarray]` inheritance, `.export()`, and
     `.materialize()`. Explicit `.project(key) -> np.ndarray` remains for
     inspection and tests, so no hot path can accidentally trigger dense array
     scatters via `obs[key]`.
   - `type PolicyObservation = managym.ObservationBatch | ObservationSelection`
     lives in `manabot.policy.spans`; `ObservationBuffers` and `select_rows` are
     deleted.
   - `MemoryStore.capture` accepts `PolicyObservation` directly (reading via
     `ObservationSelection.live_memory`), removing dense float32 buffer slicing.
4. **Host admission (`manabot.policy.admission.admit`)**:
   - `admit_native` is renamed to `admit` (steady-state name; no `native_`
     transition qualifier).
   - Signature: `admit(obs: PolicyObservation, spec: Spec, *, bucketed: bool, belief: BeliefRows[np.ndarray] | None = None) -> ObservationSpans`.
   - `admission.beliefs(belief, spec.model)` validates `BeliefRows` directly:
     requires `belief is not None` iff `spec.model.belief_count_buckets > 0`.
     `BeliefTensorView.rows(batch_size=1) -> BeliefRows[np.ndarray]` replaces
     `BeliefTensorView.observation()` and the six `"belief_*"` dict keys.
   - `check_memory_contract(spans, spec)` validates `RetainedMemory` (when
     interned) or native CSR `decision_*` tables on `ObservationSpans`.
   - Dense validators (`check`, `legal_mask`, old `admit`, `facts`,
     `shared_effects`, `relations`, `permanent_links`, `actions`, `recent_events`,
     dense `admit_memory`/`live_memory`) are deleted.
5. **Host preparation (`manabot.policy.prepare.prepare`)**:
   - Consumes `ObservationSpans` only, reading directly from `RowSpans` with no
     intermediate `dense = {}` scatter dict, no `native is None` branches, no
     `BucketedObservation`, and no v9/v10 (`stack is not None`,
     `"player_turn_state" in obs`) branches.
   - Accepts explicit `belief: BeliefRows[np.ndarray] | None = None`.
   - **Critical fix**: `used_columns`, `bucketed_columns`, and `live_tokens`
     include `belief.validity` and the belief global token when `belief is not None`
     (today the `ObservationSpans` path only iterates `(*OBJECT_KEYS, "stack")`,
     which would silently drop all belief tokens from attention at inference).
   - `reference_distribution(obs: PolicyObservation | ObservationSpans, kind: str)`
     computes `"uniform"` from `action_counts` and `"action_type_uniform"`
     directly from packed CSR `actions` rows (`[K, width]`), eliminating the
     per-minibatch `[B, max_actions, width]` scatter in PPO and damped-policy
     updates.
6. **Backends, collection, probes, and tests**:
   - `BackendPolicy.prepare(obs: PolicyObservation, *, inference=False, belief=None)`
     and `prepare_admitted(obs: PolicyObservation, *, inference=False, needs=None, memory_roots=None, belief=None)`.
   - `ActorCritic` and `MLXActorCritic` drop `stack_tokens`, `turn_state`,
     `space.one_token` forks, and dense admission/bucketing fallbacks.
   - `RolloutBatch` owns `recorded: RecordedObservations` (`rows: ObservationSelection`,
     no duplicate `memory`/`memory_capacities`), `next_obs: ObservationSelection`,
     and `action_data: PackedActions` (no `.fields` dict, non-optional
     `.selection: ObservationSelection`). `ActionRows.observation` is typed
     `ObservationSelection`.
   - `TargetPracticeVector` implements `set_observation_layout` +
     `observation_rows()` via `encode_observation_batch`, so `open_collector`
     has one buffer path (`RecordedSpans` + `step_buffers`).
   - **Counterfactual/invariance probes**:
     - `combat_probe.combat_samples` calls `policy.prepare(selection)` once per
       chunk and perturbs `Prepared.permanents` via `dataclasses.replace` +
       `read_numeric` (1× prepare instead of 5×, zero dense dict export).
     - `action_cases.identity_removed` perturbs `Prepared` via `replace`.
     - `choice_inputs.reorder_offers` and `reorder_objects` permute
       `managym.Observation` (`action_space.actions` or the 4 card/permanent
       lists) and re-encode with `encode_observation_batch`, testing the real
       native encoder and deleting `layout.object_references`.
   - **Offline persistence**:
     - `backends/torch/capture.py` and `backends/parity.py` save/load
       `obs.batch.to_bytes()` (plus `MemoryArchive` when `obs.memory` is
       present) instead of dense padded `.npz` dicts.
     - `verify/decision_memory_probe.py` and `backends/torch/decision_memory_probe.py`
       store `ObservationSelection` (with `RetainedMemory`) on `ProbeRow` and
       persist via `ObservationBatch.to_bytes()` + `MemoryArchive`.

## Findings that shaped this design

1. **Five live consumers and four offline/probe consumers had no non-dense path.**
   - Scalar `Env` (`manabot/env/env.py`) and scalar players (`eval/agents.py`,
     `sim/player.py`, `sim/anchors.py`, `sim/search/flat_mc.py`,
     `env/opponents.py`, `verify/util.py`, `training/rollout/selection_data.py`).
   - Belief-enabled models (`belief/runtime.py`, `belief/agent.py`,
     `belief/encoding.py`, `belief/likelihood.py`), where `admit_native` always
     raised on `beliefs({}, model)` AND `prepare.py`'s `ObservationSpans`
     attention-compaction loops omitted belief columns.
   - `TargetPracticeVector` (`env/target_practice.py`), `manabot/env/vector_env.py`,
     and `LocalUpdateTeacher(native_encoder=False)`.
   - Offline/probe consumers: `backends/torch/capture.py` + `backends/parity.py`,
     `verify/decision_memory_probe.py` + `backends/torch/decision_memory_probe.py`,
     `training/combat_probe.py`, and permutation/ablation tests
     (`choice_inputs.py`, `action_cases.py`, `choice_corpus.py`,
     `learning_regression.py`, `tests/model/prepared_routes.py`).
2. **The batch encoder does not depend on the dense encoder.** `PackedRows::append`
   (`managym/src/agent/observation_rows.rs:256`) calls `encode_rows(obs, config, writer, None)`.
   `write_dense_row` stays for `decision_memory::EncodedMemory`. Hidden
   dependency: `prepare_slot_buffers` (`python/vector_env_bindings.rs:982-999`)
   uses dense `encode` as a slot admission check and becomes a scratch
   `PackedRows::append`.
3. **`InputLayout` has only two variants** (`OneToken`, `Legacy`), so the enum,
   config field, and `turn_state` flag disappear entirely along with
   `input_v10_state`, `TurnStateReads`, and `RequiresInputV10`.
4. **Old recorded observation JSON is not policy input.** `PermanentData::legacy_card`,
   `SpellData::{from_legacy_card,legacy_card}`, `Observation::historical()`, and
   `upgrade_historical` stay (`observation_json.rs`). Only `legacy_input.rs`,
   `legacy_card_lists`, and `card_records` (`observation.rs:2585`) go.
5. **Engine Budget and wire schemas are unaffected**: `bench_steps` and
   `bench_search` never encode; `protocol/*.schema.json` is unaffected.
6. **The executed frontier and batched baseline used v9.** Their fresh
   recipes now select v11. Frontier evaluation still checks the saved
   checkpoint's capacities and admits the historical v9 layout.
7. **The removal condition was waived.** See questions 1 for the decision.

## Choices made for review

- **Add validated `ObservationBatch.to_bytes()` / `from_bytes()` in Rust.**
  Passed over: retaining dense dict admission for `parity.py`/`decision_memory_probe.py`
  (keeps 420 lines of dense validators and dual `prepare.py` branches alive) or
  serializing `Prepared` (breaks cross-backend/bucketed parity).
- **Drop `Mapping` from `ObservationSelection`; replace `PrefixObservation` /
  `CompactObservation` / `select_rows` with `RetainedMemory`.** Passed over:
  keeping the dict facade (hides per-minibatch `[B, 512, 23]` scatters in
  `reference_distribution` and preserves empty-dict subclasses).
- **Rename `admission.admit_native` → `admission.admit`.** Passed over: keeping
  the transitional `_native` qualifier once it is the only admission function.
- **Scalar `Env` (`reset`, `step`, `step_semantic`, `scenario_refresh`) returns
  `managym.Observation`.** Passed over: returning a one-row `ObservationBatch`
  (forces encoding on heuristic/LLM/random/search decisions) or keeping the
  `(obs, obs)` 2-tuple on `scenario_refresh`. `Env.last_raw_obs` stays as an
  alias. `RandomMatchupPlayer` (`sim/anchors.py`) reads `len(obs.action_space.actions)`
  with `np.arange(count)` to preserve uniform draws; frozen historical
  registrations retain their recorded digests.
- **Move `combat_probe` and `identity_removed` onto `Prepared`, and
  `reorder_offers`/`reorder_objects` onto `managym.Observation` +
  `encode_observation_batch`.** Passed over: keeping synthetic dense mutation
  plumbing and `layout.object_references`.
- **Belief rows travel explicitly as `BeliefRows[np.ndarray] | None`** via
  `BeliefTensorView.rows(batch_size)`. `ViewerDecision.observation` is
  `managym.Observation`.
- **`manabot.env.vector_env.VectorEnv` is deleted**; tests move to
  `RecordedVectorEnv` or scalar `Env`, or go when testing dense views.
- **`native_encoder` is removed from `LocalUpdateTeacher`**; recorded specs with
  `"native_encoder": false` are refused naming `v0.2.5`.
- **`experiments/tla_frontier.py` and `batched_baseline.py` move to v11** (same
  capacities); `best_player_on_main.py` is deleted with the house line in its
  report; `vector_statistics_cost.py` migrates if tiny, else retires the same way.
- **Old search receipts holding dense buffers** (`managym/replay.py:488-493, 651-663`)
  are refused with a typed error naming `v0.2.5`.
- **After retirement**, document `v0.2.5` as the release for v9/v10.
- `observations_for_players` layout override: questions 11.

## Source of truth

`managym::agent::observation_encoder::encode_rows` → `PackedRows` →
`managym.ObservationBatch` (live or validated `from_bytes`) →
`ObservationSelection` (with optional `RetainedMemory`) →
`manabot.policy.admission.admit` → `ObservationSpans` →
`manabot.policy.prepare.prepare` → `Prepared`.
`manabot.infra.config.SUPPORTED_POLICY_INPUTS == (11,)` is the one version gate;
native `ObservationLayout` and `ObservationBatch.from_bytes` enforce the same.

## Affected consumers

| Consumer | Change |
| --- | --- |
| Self-play collection (`training/rollout/*`) | `ObservationRows`, dense action packing, `allocate_vector_buffers`, `set_buffers`, `PrefixObservation`, `CompactObservation`, `select_rows` deleted; `RecordedSpans` + `RetainedMemory` + `ObservationSelection` only; `reference_distribution` reads CSR `actions` |
| `training/combat_probe.py` | 1× `policy.prepare(selection)` + `replace(prepared, permanents=...)` via `read_numeric` |
| `training/rollout/selection_data.py`, `training/state/game_errors.py` | `ObservationBatch` + `reference_distribution(batch, ...)` |
| Backend capture & parity (`backends/torch/capture.py`, `backends/parity.py`, `scripts/profile_mlx_learner.py`) | Save/load `ObservationBatch.to_bytes()` / `from_bytes()` (+ `MemoryArchive` when memory is present) |
| Decision-memory probes (`verify/decision_memory_probe.py`, `backends/torch/decision_memory_probe.py`, `verify/decision_memory.py`, `backends/torch/verify_decision_memory.py`) | `ObservationSelection` + `RetainedMemory` + `ObservationBatch.blank_decision_memory()` / `to_bytes()` |
| Evaluation samplers (`eval/agents.py`), Etude game children | `encode_observation_batch` + `blank_decision_memory()` + `admission.admit` + `prepare`; stays Torch-free (`tests/test_package_boundaries.py`) |
| Academy / `sim/policy_provider.py` | Already batch; dense else-branch deleted |
| Search (`sim/guidance.py`, `search/puct.py`, `flat_mc.py`, `local_update.py`, `rollout.py`) | `Mapping` inputs deleted; root observation is `managym.Observation`; `blank_decision_memory()` native |
| Belief (`manabot/belief/*`, backends, `policy/prepare.py`) | `BeliefTensorView.rows()` → explicit `belief: BeliefRows | None`; fix `used_columns` / `bucketed_columns` / `live_tokens` attention loops to include belief tokens |
| Verify (`verify/util.py`, `choice_inputs.py`, `choice_corpus.py`, `action_cases.py`, `record_coverage.py`, `learning_regression.py`) | Permutations on `managym.Observation` + `encode_observation_batch`; `identity_removed` on `Prepared`; read fields via `ObservationSelection.project` |
| Torch and MLX `policy.py` | Dict branches, `one_token`/`turn_state`/`stack_tokens` forks deleted; `belief=` parameter added |
| `managym/recording.py`, `replay.py`, `__init__.pyi` | Dense wrappers/stubs deleted; `ObservationBatch.to_bytes`/`from_bytes` stubs added |
| `scripts/pinned_pilot.py`, `scripts/bench_*.py` | `PILOTS = {"v11": …}`, `u1952`/`BEST_PLAYER` removed; benches on `RecordedVectorEnv`, dense parity mode deleted |
| Docs | User-facing docs drafted ahead of build: `docs/engine.md`, `docs/engine-python.md`, `docs/observation-and-model.md`, `docs/development.md`, `docs/training-loop.md`, `docs/experiment-metrics.md`, `managym/README.md`, `manabot/README.md`, `manabot/policy/README.md`, `manabot/backends/README.md`, `manabot/backends/mlx.md`. Remaining Pass 4 internal/release docs: `AGENTS.md`, `.lf/qa/*.md`, 0.3.0 release notes entry |

## Absent and error states

- v9/v10 anywhere (config, checkpoint, replay, Academy catalog,
  `ObservationLayout`, `ObservationBatch.from_bytes`) → `UnsupportedPolicyInput`
  / native `ValueError` naming `v0.2.5`. Academy lists such files as unsupported.
- Corrupt/truncated/out-of-bounds `ObservationBatch.from_bytes` → native
  `ValueError`; never constructs a partial or uncertified `ObservationBatch`.
- Belief-enabled model without `BeliefRows`, or belief-free model with
  `BeliefRows` → `ValueError` in `admission.beliefs`.
- Capacity overflow → `managym.ObservationCapacity` from
  `encode_observation_batch`, as today.
- Historical observation JSON → still read; encoding one still raises
  `HistoricalObservation` without offering v9/v10.

## Exclusions

Other 0.3.0 aliases (`WORLD_VERSION`, `manabot.env.failures` aliases,
`TrainingGames`, `CheckpointVillain`, `InMemorySnapshotVector`, `opponent_policy`,
`.card` getters, `manabot.managym`); frozen evidence (regime JSON, conformance
files, protocols, `release/v0.2.*` notes, experiment reports' recorded text);
the version bump to 0.3.0 itself.

## What goes

Rust: `agent/legacy_input.rs`; `InputLayout`, `turn_state`, `input_v10_state`,
`TurnStateReads`, `RequiresInputV10`; `EncodedObservation(Mut)`, `encode`,
`allocate`, `encode_into`, `encode_into_incremental`, `encode_into_memory`,
`validate_buffer_len`, the `RowWriter` impl and `values()` for
`EncodedObservationMut`; `ObservationLayout.allocate`, `python/layout_extents.rs`,
`Join`/`joins` in `observation_extents.rs`; in `python/vector_env_bindings.rs`
`ObservationBuffers`, `set_buffers`, `write_observations_for_players`,
`registered_layout`, `input_buffer_config`, `check/read_observation_buffers`,
`Destination::Dense`, `SendObservationFieldSlices`, RolloutPool
`set_buffers`/`encode_active`/`step_active`, `encode_observations_into`
(~1,000 lines); in `python/bindings.rs` `numpy_array`, `encoded_to_dict`,
`Env.encode_observation(_into)`, free `encode_observation_into`,
`legacy_card_lists`, `input_v10_state`; `Env::encode_observation(_into)`,
`VectorEnv::write_observations_for_players`; `card_records`;
`bench_observation.rs` (rewritten over `PackedRows` or deleted).

Python: `manabot/env/legacy_input.py`, `manabot/env/vector_env.py`,
`tests/fixtures/dense_collector.py`; `ObservationBuffers`,
`ObservationSpace.{allocate,encoder,encode,contains,one_token,turn_state}`,
`ObservationEncoder`, `allocate_vector_buffers`, `encode_observation`;
`ObservationRows`, `PackedActions.fields`; `PrefixObservation`,
`CompactObservation`, `select_rows`, `blank_memory`, `ObservationSelection`
`Mapping` inheritance / `.export()` / `.materialize()`; `BucketedObservation`,
`row_needs`, `bucket_sizes`, `bucket_observation`; `admission.{check,legal_mask,
facts,shared_effects,relations,permanent_links,actions,recent_events}` (and old
dense `admit`; `admit_native` renamed to `admit`); dense `admit_memory` /
`live_memory` / `MemoryStore` dict slicing; `prepare()` intermediate `dense = {}`
and `native is None` / `stack is not None` / `player_turn_state` branches;
`BeliefTensorView.observation()`; `DEPRECATED_POLICY_INPUTS`,
`SAVED_POLICY_INPUT_VERSION`; `experiments/best_player_on_main.py`.

Tests and fixtures: `tests/verify/test_input_v10.py`,
`tests/model/test_bucketed_shapes.py`, `tests/fixtures/pinned_pilot/{v9,u1952}/`,
dense-only tests in `tests/verify/test_native_buffer_capacities.py`,
`test_native_observation_contract.py`, `tests/env/test_named_observation_layout.py`,
`test_vector_snapshot_slots.py`, v9 halves of `tests/model/test_stack_tokens.py`;
Rust `input_v10_carries_turn_state_that_v9_refuses`, `legacy_card_index`, Legacy
rows of layout tests.

Must survive: v11 pilot recording bytes; `Prepared` values for v11; recorded
JSON readers; `write_dense_row`; `reference_distribution` (rewritten on CSR);
`bucket`, `BucketSizes`, `object_trimming`, `sizes_of`; `admit` (renamed from
`admit_native`), `check_memory_contract`, `binding`, vocabularies; checkpoint
`input_readings`/`legacy_readings` (readings, not input versions).

## Passes

Done:

- **0. Belief on native batches** (`993ac80296`): explicit `BeliefRows`;
  compact attention keeps belief rows and the global token (questions 12).
- **Docs drafted ahead of build** (`0bb2c15c12`); every claim must be backed by
  code before gate.
- **1. Native batch bytes and capture/parity** (`565258570e`, `0cb9a72532`):
  64 decisions archive to 130,397 bytes vs 15,808,000 dense; MLX float32 parity
  max weight diff 3.0e-08. Pass 4 must keep the object-join descriptions the
  archive uses when deleting the dense extent scanner.
- **2a. Scalar games, evaluation, collection, retained memory**
  (`0369f18b7a`..`fab0d064aa`). Broad six-suite run: 2,107 passed, 16 skipped,
  27 failed; 24 reproduce on the starting revision (conditional-search fixtures
  14, automatic game capture 3, belief fixtures 4, PUCT compatibility, recording
  export, MLX 256-wide eval timeout); the other 3 passed together in 261.52 s
  after Pass 2b. Paired-deal fixture allowance raised to 240 s (new sources play
  2,644 vs 1,683 decisions on the same deals).
- **2b. Native v11 consumers and rollout inspection**
  (`6d07c41230`..`60b8347589`): counterfactuals on `Prepared`, permutations
  re-encoded natively, decision-memory verification on packed rows, packed
  benchmarks (questions 15, 16).
- **3. Policy input v11 only** (`008ddbf3a1`): strength comparison waived
  (questions 1). Config, checkpoint, replay, Academy catalog, native layout and
  archive admission refuse v9/v10 naming v0.2.5; legacy input modules, layout
  selectors, v9/u1952 pilots and the Torch vector wrapper removed. The v11
  pilot replayed 71 decisions; its 235 probabilities matched the
  pre-retirement sources (`a9b519e647`) exactly and the committed recording to
  9.24e-7 (check is 1e-5 plus chosen-action rank). The archived v9 frontier
  evaluation is reproducible at `14a168309f`. Historical v4 eligibility
  captures: questions 17.

- **3b. Dense Python admission, preparation and rollout** (`5f66e3e99c`,
  `d752654715`, `1daaf1ce9b`): `admit_native` is `admit`; `prepare`,
  admission and bucketing read `ObservationSpans` only; `ObservationRows`,
  `PrefixObservation`, `CompactObservation`, `select_rows`, dense
  `MemoryStore` capture, `PackedActions.fields` and the `ObservationSpace`
  dense helpers (`allocate`, `encode`, `encoder`, `contains`,
  `encode_observation`, `allocate_vector_buffers`, `ObservationBuffers`) are
  deleted. Retained-memory chunk featurization keeps its padded row-dict
  featurizer (`_prepare_memory_rows`); that is retained-prefix storage, not
  transport. Tests read encodings through `tests/fixtures/observation_rows.py`
  (`ObservationSelection.project`), which matched the dense encoder exactly
  over 120 live decisions with decision memory. Taught: no `manabot` caller
  reaches the native dense entry points. Only `SnapshotVector` and
  `BufferedStreams` still implement `write_observations_for_players`, because
  the runtime-checkable vector protocol names it and `RecordedVectorEnv`
  supplies `observations_for_players` only through `__getattr__`; Pass 4
  changes that protocol together with `recording.py`. Tests: ~110 files move
  to native batches (`d752654715`..`e14c986b64`); tests of deleted dense
  validation are deleted. Scoped runs: model 673 passed, 1 strict xfail
  (questions 18: native memory details emit program tokens absent from the
  record vocabulary, which dense lookup had hidden); training/sim 823 passed,
  21 failed, all failing at `95fa222b07`; env/verify/semantic/academy/eval
  1517 passed, 50 failed, all failing at `95fa222b07`. Native-transport
  tests stay on `space.layout.allocate` + `step_buffers` for Pass 4.

Remaining:

**4. Native dense transport deleted + docs/guards.** The Rust list in "What
goes"; slot admission via scratch `PackedRows`; Rust tests read rows through
one `inspect.rs` reader over `encode_rows`; `recording.py`, `replay.py`, stubs;
remaining dense tests (native-transport tests kept running on
`space.layout.allocate` in 3b); the `write_observations_for_players` protocol
methods in `SnapshotVector`/`BufferedStreams`; decide questions 11. Debug
`cargo nextest run --profile ci` (`CARGO_BUILD_JOBS=2`, `--test-threads 2`)
and clippy. Then `AGENTS.md`, `.lf/qa/*.md`, 0.3.0 release-notes entry, and a
boundary test asserting removed names are absent.
