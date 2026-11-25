# MTG-435: remove deprecated manabot shims, aliases, CLI commands and legacy `.pt` loaders

One PR on `0.2.x` main that deletes every manabot compatibility path tagged
for 0.3.0, plus the two read-only legacy `.pt` loaders, leaving single-owner
steady-state persistence, belief-admission, and replay contracts. No version
bump here.

Scope sources: the Task brief, and the steer of 2026-11-24 (comment
`5dbae184`), which supersedes the brief's "keep read-only legacy `.pt`
save/belief loaders":

> Updated scope per Jack Heart: also delete BOTH read-only legacy .pt loaders
> in this Task: 1. Legacy .pt tracked-training save loader (restore_legacy,
> legacy_file, legacy_parameters, legacy_adam, and the
> learner.pt/optimizer.pt/ema.pt branch ...). Refuse legacy learner.pt saves
> clearly (pointing to v0.2.5 or warm-starting from the published policy)
> instead of loading them. 2. Legacy .pt belief sampler loader
> (decode_legacy_sampler and the PK\x03\x04 branch ...). Only
> belief.safetensors is accepted. Keep published policy checkpoint .pt reading
> (manabot/backends/torch/checkpoint.py / manabot.backends.convert) untouched
> because experiments/mini_baseline_backends.py and
> scripts/profile_mlx_learner.py still reference initial.pt.

## User-visible outcome

- Importing a removed module or name fails with `ImportError` /
  `AttributeError` instead of a `DeprecationWarning`.
- `manabot export-game` is no longer a command (Typer "No such command").
- `etude train ... --resume` on a save holding `learner.pt` fails before
  model/reading diffing or tensor loading, on both Torch and MLX, with one
  message: resume it with manabot 0.2.5, or warm-start a new run from the
  run's published policy. Original save bytes and live tensors stay untouched.
- A belief sampler whose bytes are a Torch zip (or any non-safetensors payload)
  is refused on both backends with `ValueError("unknown sampler checkpoint
  format")`; only the `belief.safetensors` container loads.
- Nothing changes for current recipes, current saves, published policy
  checkpoints (`.pt` and `.safetensors`), recorded games, or Academy opening
  files earlier releases wrote.

## Steady-state model across layers

1. **Single-owner save admission and non-optional resumed host state
   (`manabot/training/state/continuation.py`,
   `manabot/training/loops/self_play.py`,
   `manabot/training/state/backend.py`,
   `manabot/backends/{torch,mlx}/state.py`).**
   - `continuation.py` owns save-format admission via a read-only helper
     `admit_save_format(folder) -> dict[str, Any]`:
     1. If `(folder / "training.safetensors").exists()`, raise
        `ValueError("training.safetensors: unsupported pre-portable MLX save")`.
     2. Read `record = json.loads((folder / "state.json").read_bytes())`.
     3. If `"learner" in record["files"]` or `(folder / "learner.pt").exists()`,
        raise `ValueError("state.json: this save holds learner.pt, written
        before portable saves; resume it with manabot 0.2.5, or start a new run
        warm-started from the run's published policy")`.
     4. Return `record`.
   - `self_play.resume_training`'s `restore(folder)` callback executes a strict
     three-phase read-before-mutate sequence without inspecting save files
     itself:
     1. `record = admit_save_format(folder)` (refuses legacy Torch and
        pre-portable MLX saves first, even if `counts.model` or
        `counts.readings` is absent or changed);
     2. `resume_model(counts.model, settings.model)` and `readings =
        resume_readings(counts, learner.agent.readings,
        learner.agent.observation_space)` (runs before `counts_sha256` and
        before tensor loading so edited/drifted portable saves still produce
        the semantic model/readings refusal without mutating live weights);
     3. `restored = restore_continuation(folder, ..., record=record)`
        (validates `updates`, `counts_sha256`, portable `expected_files`,
        headers, buffers, parameters, Adam, `HostContinuation`, and sampling,
        then copies into live `learner`, `ema`, and `adam` last).
     (`restore_continuation` calls `admit_save_format(folder)` if `record is
     None` so direct callers also enforce format admission.)
   - `RestoredContinuation.host` becomes required (`HostContinuation`, not
     `HostContinuation | None = None`): the `host=None` half-state existed only
     for legacy `.pt` saves and as a fresh-run sentinel. `resume_training`
     returns `tuple[Counts, RestoredContinuation | None]` (`None` when fresh;
     `RestoredContinuation` when resumed). `self_play` branches once on
     `restored is not None`.
   - `Continuation.restore_legacy` is removed from `training/state/backend.py`,
     along with `restore_legacy`, `legacy_file`, `legacy_parameters`,
     `legacy_adam` (and now-unused `Iterator`, `contextmanager`, `Path`,
     `pickle`) from `backends/torch/state.py` and `restore_legacy` (and `Path`)
     from `backends/mlx/state.py`.

2. **Single-format belief sampler admission (`manabot/belief/checkpoint.py`,
   `manabot/belief/protocol.py`, `manabot/backends/{torch,mlx}/belief.py`).**
   - Delete `decode_legacy_sampler` from `SamplerCodec`,
     `backends/torch/belief.py` (dropping unused `io`, `Any`, and inlining
     single-caller `canonical_sampler_parameters` if no other caller remains),
     and `backends/mlx/belief.py`.
   - Delete the `if data.startswith(b"PK\x03\x04"):` branch in
     `read_belief_sampler`.
   - Wrap `decode_safetensors(data)` and metadata `json.loads` in
     `try ... except (ValueError, TypeError, KeyError, AttributeError,
     struct.error) as error: raise ValueError("unknown sampler checkpoint
     format") from error` (matching `continuation.py:read_tensors`), so Torch
     zips, truncated files, and malformed headers never leak raw decode
     exceptions.

3. **One recording module (`managym.recording`), one arena-trace module
   (`manabot.sim.replay`), one replay runtime check (`managym.replay`).**
   - Move `record_actions` (dropping its dead `option_counts` parameter, whose
     only caller was `TrainingGame.export`) into `manabot/sim/replay.py`,
     beside `replay_environment`, `write_trace`, `read_trace`, and
     `replay_games`, and delete `manabot/sim/recording.py` completely (removing
     the duplicate `recording.py` filename across `managym` and `manabot`).
   - Update `scripts/release/academy_replay_smoke.py` to import `GameInputs,
     PlayerSetup` from `managym.recording` and `record_actions, write_trace`
     from `manabot.sim.replay`.
   - Delete `manabot/env/replay.py` and `manabot/env/failures.py`'s
     `runtime_identities`. In `manabot/sim/replay.py`, collapse the duplicate
     `require_record_runtime` alias to one `from managym.replay import
     require_replay_runtime`. In `manabot/training/state/game_errors.py`,
     import `require_replay_runtime` from `managym.replay`.
   - In `manabot/env/failures.py`, delete `__getattr__`,
     `TrainingExposureReceipt`, `NativeTrainingGame`, `NativeTrainingTrace`,
     and all imports made dead by those removals. Reference
     `recording.GameInputs` and `recording.GameRecord` via the existing
     `from managym import recording` import (do NOT import bare `GameRecord` or
     `GameInputs` at module scope, which would leave `hasattr(failures,
     "GameRecord") == True`).

## What goes

Callers were traced with `git grep` and an AST probe of every
`from <module> import <name>` resolving through a `__getattr__` fallback.

| Remove | Callers & CI to convert |
|---|---|
| `manabot/env/card_capture.py`, `manabot/infra/journal.py`, `manabot/sim/search/flat.py` (whole modules) | none in code; drop `manabot/sim/search/flat.py` from `.github/ci/select-tests.sh:59` |
| `manabot/env/replay.py` (whole module) | `manabot/sim/replay.py` (collapse `require_record_runtime` alias to `managym.replay.require_replay_runtime`), `manabot/training/state/game_errors.py:29` → `managym.replay.require_replay_runtime`; update `managym/replay.py:406` error string |
| `manabot/sim/recording.py` (whole module: `TrainingGames`, `record_games`, `TrainingGame`, `_read_receipt`; move `record_actions` without `option_counts` to `manabot/sim/replay.py`) | `scripts/release/academy_replay_smoke.py:19` → `from manabot.sim.replay import record_actions, write_trace`; remove `tests/sim/test_recording.py` from `.github/workflows/manabot-integration.yml:68` |
| `manabot/env/failures.py`: module `__getattr__`, `runtime_identities`, `TrainingExposureReceipt`, `NativeTrainingGame`, `NativeTrainingTrace`, and dead imports (`Counter`, `cache`, `hashlib`, `platform`, `np`, `warnings`, `CapturedCards`, `_PlayerSetup`, `_game_disclosure`, `_jsonl`, `ConfigDict`, `Field`, `model_validator`, `Literal`); switch `_GameInputs`/`_GameRecord` annotations to `recording.GameInputs`/`recording.GameRecord` | `scripts/release/academy_replay_smoke.py:18` → import `GameInputs`, `PlayerSetup` from `managym.recording` |
| `manabot/cli.py` `export-game` (and unused `warnings` import) | none |
| `manabot/sim/table_bots.py` `CheckpointVillain` (and unused `Sequence`, `warnings`, `PolicyRunner`, `PolicySpec` imports) | tests below |
| `manabot/reports/self_play.py` `SelfPlayPoint.position` (and unused `Literal`, `warnings`, `SEAT_LABELS` imports) | none |
| `manabot/training/updates/damped_policy.py` `compute_targets`, `DampedTargets` (and unused `dataclass`, `warnings` imports) | `tests/training/test_credit_assignment.py` (5 calls) → `compute_batch_targets` (`value_targets` replaces `values`) |
| `manabot/training/rollout/snapshot_vector.py` `InMemorySnapshotVector` (and any imports it leaves unused) | one test |
| `manabot/backends/torch/learner.py` module `__getattr__` (and unused `warnings` import) | one test; rename `_StepRecorder` to plain `StepRecorder` now that nothing shadows it |
| `manabot/backends/{torch,mlx}/rl.py` `damped_policy_loss`, `sys`, and `typing_extensions.deprecated` import | `tests/training/test_snapshot_rollouts.py:972,1066` → `objective.damped_policy_loss(backend.rl, ...)` |
| Legacy save loader: `restore_legacy`, `legacy_file`, `legacy_parameters`, `legacy_adam` in `backends/torch/state.py`; `restore_legacy` in `backends/mlx/state.py`; `Continuation.restore_legacy` in `training/state/backend.py`; `"learner" in files` load branch in `training/state/continuation.py`; `learner.pt` model-settings fallback in `training/loops/self_play.py:1050` | replaced by `admit_save_format` refusal + non-optional `RestoredContinuation.host` above |
| Legacy belief loader: `decode_legacy_sampler` in `backends/{torch,mlx}/belief.py` and `SamplerCodec`; `PK\x03\x04` branch in `belief/checkpoint.py` | `tests/belief/test_sampling_report.py::_checkpoint` → write through `save_belief_sampler`; wrap safetensors decode errors as `ValueError("unknown sampler checkpoint format")` |

`--bucketed-shapes`: no CLI alias exists (`git grep 'bucketed-shapes'` finds
only the v0.2.3 release note). Nothing to remove. `bucketed_shapes` the
setting is current and stays.

## Tests removed and converted

Tests removed because the behavior is gone:
- `test_captured_vector_deprecated_alias_warns_and_resolves` (the
  `failures.CapturedVector` half; the `managym.recording.CapturedVector` half
  stays)
- `test_portable_record_names_warn_before_removal`
- `test_retained_checkpoint_loader_accepts_omitted_setup`
- `test_deprecated_snapshot_factory_uses_native_deferred_slots`
- The `StepRecorder` legacy-import assertion in `test_learning_pieces.py:206`
- The `backend.rl.damped_policy_loss` warning assertion in
  `test_update_backends.py:132`
- `test_legacy_torch_save_requires_torch` (folded into the unified refusal test
  on Torch and MLX below)
- `test_replay_runtime_admission` and
  `test_replay_requires_source_only_for_automatic_records` in
  `tests/training/test_automatic_game_capture.py` (duplicated by
  `tests/env/test_engine_recording.py:73-95` once the wrapper is gone)
- `tests/sim/test_recording.py` (delete file and remove from
  `.github/workflows/manabot-integration.yml:68`; update comment at
  `tests/academy/test_configured_launch.py:526`)

Tests converted or added:
- `CheckpointVillain` in `tests/academy/test_watch.py`,
  `tests/model/test_policy_input_rejection.py`, `tests/model/test_world.py`:
  move to Academy's `PolicyRegistry(providers={"manabot":
  provider}).register(PolicySpec("manabot", path, ...))` with the player
  configs (`test_watch.py:305` already shows the accepting case).
- `test_legacy_torch_save_migrates` in
  `tests/training/test_cross_backend_continuation.py` becomes
  `test_legacy_torch_save_is_refused`: the `legacy_save` fixture writes a real
  `learner.pt` save through Etude object registration (and also verifies the
  refusal fires even when `counts.model` / `counts.readings` are absent), then
  `--resume` raises the new refusal message on Torch and (when installed) MLX,
  leaving original save bytes, `state/current.json`, and live weights
  untouched.
- `test_legacy_sampler_zip_requires_torch` in
  `tests/belief/test_sampling_report.py` becomes
  `test_legacy_sampler_zip_is_refused`: writes a Torch `.pt` zip payload,
  passes its real SHA-256 so byte-identity admission passes, and asserts
  `ValueError("unknown sampler checkpoint format")` on Torch and (when
  installed) MLX.
- `tests/training/test_mlx_continuation.py:132`: update for `resume_training`
  returning `RestoredContinuation | None` (`assert restored is not None and
  restored.sampling is not None`).
- Add a concise `record_actions` round-trip and unoffered-action refusal test
  in `tests/sim/test_replay.py`.

## Decisions made for review

Recorded alongside alternatives in `scratch/concept-review.html`:

1. **Single ordered save-admission sequence + non-optional
   `RestoredContinuation.host` (2a).** `admit_save_format` in `continuation.py`
   refuses `training.safetensors` and `learner.pt` before `resume_model` /
   `resume_readings`, which still run before `counts_sha256` and tensor
   loading. `RestoredContinuation.host` is required and `resume_training`
   returns `RestoredContinuation | None`. Passed over: keeping `host: None`
   for a deleted legacy path, or running `restore_continuation` wholesale
   before `resume_readings` (which breaks edited-readings refusal tests and
   atomic weight admission).
2. **`record_actions` moves to `manabot/sim/replay.py`; `manabot/sim/recording.py`
   is deleted (1a).** `record_actions` builds Command traces for `write_trace`,
   already depends on `sim/replay.py`, and was never a documented public API;
   deleting `manabot/sim/recording.py` eliminates the duplicate `recording.py`
   filename across `managym` and `manabot`. Passed over: keeping a 1-function
   `manabot/sim/recording.py`.
3. **Legacy belief zips get the generic `"unknown sampler checkpoint format"`
   refusal (3a).** Wrapping `decode_safetensors` + JSON decoding catches
   `struct.error` / `UnicodeDecodeError` / `JSONDecodeError` / `KeyError` /
   `TypeError` / `AttributeError` and raises `ValueError("unknown sampler
   checkpoint format")` with no `PK\x03\x04` check left in code. Passed over:
   keeping a `PK\x03\x04` branch for a zip-specific error message.
4. **Legacy source admission is deleted with `manabot/env/replay.py` (4a).**
   Its `source_sha256` comparison cannot pass for records written by earlier
   revisions. Callers move to `managym.replay.require_replay_runtime`;
   `failures.runtime_identities` is deleted; `managym/replay.py:406` and
   `docs/engine-python.md:367` stop claiming legacy verification requires
   manabot. Passed over: moving an unpassable wrapper into `failures.py`.

## Exclusions

- `manabot.managym` alias (MTG-295) stays.
- Published policy checkpoint `.pt` reading stays (`backends/torch/checkpoint.py`,
  `manabot.backends.convert`).
- `manabot train` stays as its already-removed migration tombstone (exit 2).
- v9/v10 inputs, dense observation transport, `ObservationSpace.encode` /
  `encoder` / `ObservationEncoder`, `WORLD_VERSION`, the `academy` package
  alias, `PlayerConfig.disclosure`, `managym.recording.CapturedVector` and
  every other managym or Academy alias: sibling Tasks.
- `TrainingBots.verify_arena` stays: Academy still opens command traces and
  games that `export-game` wrote in earlier releases.
- Release notes and `release/` are frozen; `experiments/protocols/*.json`
  source listings are frozen evidence.

## Docs

Updated in place to state the steady state (dropping the "deprecated, removed
in 0.3.0" sentences and legacy-loader fallbacks):

- `AGENTS.md`: `managym.replay.matches_retained_error`.
- `manabot/README.md`: removed `record_games(...)` / `manabot export-game`
  deprecation sentence.
- `manabot/backends/README.md`: legacy Torch/MLX saves refused on both
  backends, belief samplers safetensors-only, dropped deprecated `StepRecorder`
  and `backend.rl.damped_policy_loss` notes, clarified `.pt` policy vs save/sampler
  handling.
- `manabot/backends/mlx.md`: legacy Torch saves and `.pt` belief samplers
  refused on both backends.
- `managym/README.md`: earlier exported training games without offering
  `manabot export-game`.
- `managym/academy/README.md`: earlier exported training games without naming
  `manabot.sim.recording`.
- `docs/academy.md`: earlier exported training games without offering
  `record_games(...)`, `TrainingGames`, or `manabot export-game`.
- `docs/training.md`: `self_play` snippet for `resume_training` returning
  `RestoredContinuation | None` with non-optional `host`, safetensors-only
  belief samplers, recovery table and refusal messages for `learner.pt` and
  `training.safetensors`, `managym.replay`.
- `docs/training-loop.md`: dropped `InMemorySnapshotVector.factory` and legacy
  Torch `.pt` save stream sentences.
- `docs/engine-python.md`: legacy source-digest receipts require
  `--allow-changed-runtime` or the original revision rather than manabot.
- `.lf/qa/training.md`: `learner.safetensors` beside `continuation.json` and
  legacy `learner.pt` save refusal QA step on both backends.

## Acceptance check at gate

```bash
uv run python - <<'EOF'
import importlib, warnings
warnings.simplefilter("error")
for gone in ("manabot.env.card_capture", "manabot.env.replay",
             "manabot.infra.journal", "manabot.sim.recording",
             "manabot.sim.search.flat"):
    try: importlib.import_module(gone)
    except ModuleNotFoundError: continue
    raise SystemExit(f"{gone} still imports")
checks = {
    "manabot.env.failures": (
        "CapturedVector", "GameRecord", "GameInputs", "runtime_identities",
        "TrainingExposureReceipt", "NativeTrainingGame", "NativeTrainingTrace",
    ),
    "manabot.sim.table_bots": ("CheckpointVillain",),
    "manabot.training.updates.damped_policy": ("compute_targets", "DampedTargets"),
    "manabot.training.rollout.snapshot_vector": ("InMemorySnapshotVector",),
    "manabot.backends.torch.rl": ("damped_policy_loss",),
    "manabot.backends.mlx.rl": ("damped_policy_loss",),
    "manabot.backends.torch.state": ("restore_legacy", "legacy_file", "legacy_parameters", "legacy_adam"),
    "manabot.backends.mlx.state": ("restore_legacy",),
    "manabot.backends.torch.belief": ("decode_legacy_sampler",),
    "manabot.backends.mlx.belief": ("decode_legacy_sampler",),
}
for module, names in checks.items():
    try:
        loaded = importlib.import_module(module)
    except ModuleNotFoundError:
        if ".mlx." in module:
            continue
        raise
    assert not [n for n in names if hasattr(loaded, n)], (module, names)
import manabot.managym  # retained alias
from manabot.sim.replay import record_actions  # moved from deleted sim/recording.py
print("ok")
EOF
uv run manabot export-game x --out y; test $? -eq 2
git grep -n -E "0\.3\.0" -- manabot ':!manabot/env/legacy_input.py' ':!manabot/env/observation.py'
```

Expected: `ok`; Typer exits 2 with "No such command"; the grep lists only
dense-transport / v9-v10 comments and `manabot/README.md:126`
(`PlayerConfig.disclosure`), `semantic/text_observation.py:54` — the sibling
Tasks' rows — and nothing from "What goes".

Then the consumers:

```bash
uv run --extra dev pytest tests/test_package_boundaries.py tests/sim tests/belief \
  tests/env/test_captured_vector_bookkeeping.py tests/env/test_engine_recording.py \
  tests/training/test_automatic_game_capture.py tests/training/test_cross_backend_continuation.py \
  tests/training/test_mlx_continuation.py tests/training/test_self_play_loop.py \
  tests/training/test_credit_assignment.py tests/training/test_snapshot_rollouts.py \
  tests/training/test_update_backends.py tests/training/test_learning_pieces.py \
  tests/training/test_game_errors.py tests/model/test_world.py \
  tests/model/test_policy_input_rejection.py tests/academy/test_watch.py \
  tests/academy/test_checkpoint_catalog.py tests/academy/test_configured_launch.py \
  -q -W error::DeprecationWarning:manabot
uv run python scripts/release/academy_replay_smoke.py --help
```

plus the repository's lint/type configuration, and
`docs/development.md#validation-by-change-area` for the touched packages.

## Passes

1. **Complete.** Removed legacy training and belief loaders and moved replay admission to managym. Torch ↔ MLX portable resume, rejection with missing settings/readings and intact weights/save bytes, sampler reports, and both replay CLIs passed.
2. **Next. Shims, aliases, CLI, and `sim/recording.py` removal** — move
   `record_actions` to `manabot/sim/replay.py`, delete `manabot/sim/recording.py`,
   `manabot/env/card_capture.py`, `manabot/infra/journal.py`,
   `manabot/sim/search/flat.py`, and remove all deprecated aliases/commands in
   the table above alongside their caller, CI, and test conversions.
3. **Docs, CI check, and acceptance gate.**

## Done when

The gate script prints `ok`, `manabot export-game` is unknown, a legacy
`learner.pt` save is refused before model/reading diffing or tensor loading
with the message naming 0.2.5 and warm-starting, a zip belief sampler is
refused on both backends as `"unknown sampler checkpoint format"`, the consumer
tests pass with manabot `DeprecationWarning`s as errors, CI workflow paths are
clean, and no doc offers a removed name.
