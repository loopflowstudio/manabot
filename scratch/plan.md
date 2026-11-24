# a66f13 — one portable continuation for tracked self-play

Stacked on dba0f2 (PR #585). Base `0c1a7a1a77`.

## Outcome

A tracked self-play run writes the same four files whichever backend runs it,
and resumes on either backend. Start on CUDA Torch, stop, continue on Apple MLX
(or the reverse, or move Torch between CUDA/MPS/CPU): learner weights, EMA
weights, Adam moments and step counts, cumulative counts, input readings and
in-flight native games (`continuation.json`) come back intact. Within the same
sampling domain (`mlx`, `torch:cpu`, `torch:cuda`, `torch:mps`), collector and
framework RNGs restore bit-exact too; across domains, sampling reseeds
deterministically from `(run.seed, counts.updates)` and records
`run/sampling_reseeded=1` on the first resumed update.

Old Torch saves (`learner.pt` / `optimizer.pt` / `ema.pt` written by Etude
object registration) still resume on Torch, read-only, with fresh game and
sampling streams; their next save writes the canonical portable layout.

Published policy milestones (`out/policies/NNNNNNNN-{raw,ema}.safetensors`) stay
completely separate from internal training recovery (`out/state/save-NNNNNNNN/`):
milestones are forward-compatible evaluation artifacts with `world_binding`,
`checkpoint_source` and catalog rebinding; training saves are strict
exact-match continuation snapshots.

## One save folder (four files, four owners)

Written by `export_continuation(folder, ...)` registered through Etude's
`run.resume(export=, restore=, extra=Counts())` (Etude PR #141, public API).
Every backend writes the exact same four filenames:

| File | Owner | Content |
|---|---|---|
| `state.json` | Etude | Clocks (`at`), `wall_ms`, `seed`, `commit`, custom `files` inventory, `extra` (`Counts`: updates, transitions, optimizer_updates, model, readings, snapshot_bank) |
| `continuation.json` | `manabot/training/state/continuation.py` (dba0f2, unchanged) | `HostContinuation` (schema 2): `CollectorSnapshot` (live game roots/prefixes, `learned_game_rows`, `opponent_rng`), minibatch/Python/NumPy RNG |
| `learner.safetensors` | `manabot/training/state/continuation.py` + `optimizer.py` (pure NumPy via `manabot/policy/checkpoint.py` `write_safetensors`/`read_safetensors`) | Canonical float32 `raw/<canonical>`, optional `ema/<canonical>`, `adam/step` (`int64` scalar), and when `step > 0`, `adam/exp_avg/<canonical>` & `adam/exp_avg_sq/<canonical>` for every parameter with `param_steps[name] > 0`. Its `__metadata__` is the single continuation manifest: `schema="canonical-v1"`, `backend`, `precision`, `has_ema`, `param_steps` (JSON map `canonical -> int`), `observation` config JSON, `buffers` (JSON map `name -> [dtype, shape, sha256]`), `counts_sha256`, `continuation_sha256`, `sampling_sha256`, `sampling_domain`. |
| `sampling.safetensors` | `manabot/training/state/continuation.py` calling backend hooks (pure NumPy safetensors; add `"U32": np.uint32`, `"U16"`, `"U64"` to `_DTYPES` in `manabot/policy/checkpoint.py`) | Backend-private sampling RNG arrays: MLX writes `rng.collector`, `rng.mlx` (`uint32 (2,)`); Torch writes `rng.collector`, `rng.torch.cpu`, plus `rng.torch.cuda` (on CUDA) or `rng.torch.mps` (on MPS) (`uint8 1-D`). |

### Choices made for review

- **Dedicated `learner.safetensors` instead of reusing the published policy
  container (`raw.safetensors` / `ema.safetensors` + `optimizer.safetensors`).**
  Chosen because training continuation and published policy milestones have
  opposite contracts: `write_policy` runs `git` via `checkpoint_source()`,
  builds `world_fields()`, duplicates megabytes of lookup buffers across raw
  and EMA, and round-trips `backend.load_policy` (whose `read_container`
  performs forward-compatible catalog rebinding rather than exact-match
  verification). One `learner.safetensors` writes weights, EMA, Adam and a
  single complete integrity manifest (including `precision`, `buffers` digests
  that still name the first differing buffer key, `counts_sha256`,
  `continuation_sha256` and `sampling_sha256`) in one fsync and one upload
  without git subprocesses or model reconstruction. Passed over: six files with
  published policy containers for raw/EMA and integrity digests overloaded onto
  `optimizer.safetensors`.
- **Per-parameter `param_steps: dict[str, int]` instead of scalar `step` +
  `unstepped: frozenset[str]`.** Chosen because Torch's conditional encoders
  (`SharedEffectEncoder` when `rows is None`, `FactEncoder` when
  `rows.programs is None`) leave `p.grad is None` on minibatches without active
  effects or fact programs. PyTorch `torch.optim.Adam` skips `state[p]["step"]`
  and moment decay on `None` gradients, so a parameter can legitimately hold
  nonzero moments with `0 < param_step < counts.optimizer_updates`. Recording
  `param_steps` (`0 <= param_steps[name] <= step`, where `0` means unstepped
  and omits zero moment tensors) makes Torch → Torch 100% bit-exact, MLX →
  Torch exact. MLX stores the difference between the global step and each
  younger parameter's step on that parameter's optimizer leaf. Bias correction
  uses its own clock, including its first gradient after import. Report the
  number of clocks aligned with the source in `run/adam_steps_aligned`.
- **One filename `sampling.safetensors` keyed by `sampling_domain` instead of
  `sampling.<backend>.safetensors`.** Chosen because PyTorch `torch.Generator`
  RNG state is incompatible across Torch devices (5056-byte MT19937 on CPU vs
  16-byte Philox on CUDA/MPS). Keying compatibility by `sampling_domain`
  (`"mlx"` across CPU/MPS; `"torch:cpu"`, `"torch:cuda"`, `"torch:mps"` for
  Torch) keeps one backend-independent 4-file save layout, prevents Torch
  CUDA→CPU crashes, and verifies `sampling_sha256` uniformly on every resume.
  Passed over: backend-suffixed filenames.
- **No fallback for unmerged dba0f2 MLX `training.safetensors` saves.** Fails
  explicitly naming `training.safetensors`. Passed over: a second read-only
  loader for unmerged MLX schema 1/2.

## Adam representation and parameter mapping

Canonical parameter names reuse the existing linear bijections:
`canonical_name`/`torch_name` (`manabot/policy/checkpoint.py`) and
`mlx_parameters`/`canonical_from_mlx` (`manabot/backends/mlx/checkpoint.py`).
Because both maps only rename, split and concatenate, and Adam moments are
elementwise, `exp_avg` and `exp_avg_sq` pack/split identically to weights (in
Torch `in_proj_weight` is one tensor so Q/K/V share one `param_step`; in MLX
fresh parameters share the global `step`; imported parameters may be younger).

Both learners use the same Adam constants: `betas=(0.9, 0.999)`, `eps=1e-5`,
bias correction enabled; learning rate is not continuation state (set every
update by `set_learning_rate`).

Invariants:
- When `step == 0` (e.g. update 0 stop or all-skipped update): `param_steps == {}`,
  `exp_avg == {}`, `exp_avg_sq == {}`. Torch keeps `optimizer.state == {}`; MLX keeps its scalar step and learning rate without moments.
- When `step > 0`: `param_steps.keys()` equals the model's trainable canonical
  parameter names; `0 <= param_steps[name] <= step`; `exp_avg` and `exp_avg_sq`
  have exact keys `{k for k, s in param_steps.items() if s > 0}`, float32,
  matching parameter shapes, all finite.
- Torch `load_adam` restores `state[param]["step"]` as a **CPU 0-D `float32`
  tensor** (`torch.tensor(float(p_step), dtype=torch.float32)`) and moves
  `exp_avg` / `exp_avg_sq` to `param.device` as `float32`; parameters with
  `p_step == 0` stay absent from `optimizer.state`.
- Torch `load_policy_arrays` copies tensors **in place** (`param.copy_`) so
  `learner.optimizer` keeps valid references to `learner.agent.parameters()`
  and training mode stays intact.

## Shared/backend split

- `manabot/policy/checkpoint.py`: add `"U16"`, `"U32"`, `"U64"` to `_DTYPES` so
  pure-NumPy `write_safetensors`/`read_safetensors` supports MLX `uint32` RNG
  keys alongside Torch `uint8` RNG tensors.
- `manabot/training/state/optimizer.py` (framework-free, NumPy only):
  `AdamState` dataclass (`step: int`, `param_steps: dict[str, int]`,
  `exp_avg: dict[str, np.ndarray]`, `exp_avg_sq: dict[str, np.ndarray]`),
  validation (`admit_adam` checking keys, shapes, float32, finiteness, step
  bounds against expected parameter shapes and `counts.optimizer_updates`).
- `manabot/training/state/continuation.py` (framework-free):
  keeps `HostContinuation`, `write_host`, `read_host`, `capture_host`,
  `restore_host_rng`, and adds:
  - `RestoredContinuation` dataclass (`host: HostContinuation | None`,
    `sampling: dict[str, np.ndarray] | None`, `sampling_reseeded: bool`,
    `adam_steps_aligned: int`).
  - `buffer_digests(buffers)` → `{name: [dtype, shape, sha256]}` (exact-match
    comparison that names the first differing buffer key on refusal).
  - `export_continuation(folder, *, backend_name, state, learner, ema, counts,
    settings, collector, rng)` — writes `continuation.json`,
    `sampling.safetensors`, then `learner.safetensors` with all digests bound.
  - `restore_continuation(folder, *, state, learner, ema, counts, settings)` —
    dispatches between `learner.safetensors` (canonical), `learner.pt` +
    `optimizer.pt` (legacy Torch via `state.restore_legacy`), `training.safetensors`
    (explicit refusal), or missing/corrupt save; validates EVERY header, digest,
    buffer, parameter, Adam moment, `HostContinuation`, and compatible
    `sampling.safetensors` **before** mutating any live object.
- `manabot/training/state/backend.py` (`Continuation` protocol):
  replaces `restore(run, ...)` with pure array/state hooks:
  - `policy_arrays(policy) -> dict[str, np.ndarray]` (reused by `checkpoint.py`)
  - `policy_buffers(policy) -> dict[str, np.ndarray]` (reused by `checkpoint.py`)
  - `load_policy_arrays(policy, arrays: Mapping[str, np.ndarray]) -> None`
  - `adam_state(learner, *, step: int) -> AdamState`
  - `load_adam(learner, state: AdamState) -> int` (returns `adam_steps_aligned`)
  - `sampling_domain(device: str) -> str`
  - `sampling_state(session: CollectionSession) -> dict[str, np.ndarray]`
  - `validate_sampling(device: str, arrays: Mapping[str, np.ndarray]) -> None`
  - `restore_sampling(session: CollectionSession, arrays: Mapping[str, np.ndarray]) -> None`
  - `restore_legacy(folder: Path, learner: SelfPlayLearner, ema: BackendPolicy | None) -> None`
- `manabot/training/loops/self_play.py`:
  owns `run.resume(export=, restore=, extra=Counts())` ONCE for both backends
  in two clean phases:
  1. Before `self_play_games`: call `run.resume(export=unopened_export,
     restore=restore, extra=Counts())`. Inside `restore(folder)`: validate
     `resume_model` and `resume_readings` against `counts` (preserving legacy
     fallback to `policies/00000000-raw.safetensors` when `counts.model is None`),
     then call `restore_continuation(...)` and hold the resulting
     `RestoredContinuation`. After `run.resume` returns, set `counts.model` and
     `counts.readings` (no-op on canonical resume; populates fresh and legacy
     runs). `unopened_export` raises `RuntimeError` if ever called before the
     collector opens.
  2. Inside `with self_play_games(..., snapshot=host.collector) as collector:`:
     if `restored.host is not None`, call `restore_host_rng(restored.host, rng)`;
     if `restored.sampling is not None`, call
     `state.restore_sampling(collector.numerics, restored.sampling)`. Then
     register the live exporter via `run.save(export=export)` (which saves
     update 0 immediately when `should_stop` is set, and otherwise installs
     `export` for due saves and planned stops inside `run.steps(...)`). On the
     first resumed update, log `run/sampling_reseeded=1` if reseeded and
     `run/adam_steps_aligned=<n>` if `> 0`.
- `manabot/backends/torch/state.py`: implements the `Continuation` hooks +
  `restore_legacy` (`torch.load(..., map_location="cpu", weights_only=True)`,
  reading all `.pt` files into memory before mutating any live object).
- `manabot/backends/mlx/state.py`: implements the `Continuation` hooks; refuses
  `restore_legacy` with an actionable message; removes `from etude.train.state
  import read_state` and all `training.safetensors` code. Incident capture
  (`capture_incident`) keeps its own `training_arrays` helper untouched.

## What goes

- `training.safetensors` (MLX schema 1/2), `verify_header`, `sampling_digest`,
  MLX `binding`, `StagedContinuation`, and `from etude.train.state import
  read_state` in `manabot/backends/mlx/state.py` and tests.
- Torch `restore()` registering `learner`/`optimizer`/`ema` objects with Etude.
- `Continuation.restore(run, ...)` protocol method.
- `test_tracked_state_refuses_a_different_framework` (inverted: cross-backend
  resume now continues).

Survives unchanged: `continuation.json` / `CollectorSnapshot`; numerical
incident capture (`capture_incident`, `.pt` / `.safetensors`);
`manabot/training/loops/minimal.py` (standalone Torch tutorial loop).

## Absent and error states

- **Fresh run:** Etude skips `restore`; `counts.model`/`counts.readings` set
  from the freshly built policy; first `run.save` writes all four files.
- **Same sampling domain resume:** weights, EMA, Adam, counts, live games, host
  RNGs and backend sampling RNG restore bit-exact (on CPU float32, safe-stop
  matches uninterrupted bit-for-bit on both Torch and MLX).
- **Different sampling domain resume (Torch ↔ MLX or Torch CUDA ↔ CPU/MPS):**
  `sampling.safetensors` digest is still verified (tampering fails closed), its
  tensors are ignored, sampling reseeds deterministically from
  `(run.seed, counts.updates)`, and `run/sampling_reseeded=1` is logged on the
  first resumed update. On Torch → MLX with `param_steps[k] < step`,
  per-parameter clocks are restored with their original moments and `run/adam_steps_aligned` is logged.
- **Legacy Torch save (`learner.pt` / `optimizer.pt` / optional `ema.pt`):**
  on Torch, loaded read-only through `restore_legacy` with fresh games/RNGs
  (next save writes the canonical layout); on MLX, refused with `"this save was
  written by Torch object registration; resume once on Torch to write a portable
  save"`.
- **Pre-unification MLX save (`training.safetensors`):** refused explicitly on
  both backends.
- **Any contract or integrity failure** (missing/extra key, shape, dtype,
  nonfinite, `step != counts.optimizer_updates`, invalid `param_steps`, digest
  mismatch on `counts`/`continuation`/`sampling`, `precision` mismatch, `has_ema`
  mismatch, buffer/model/readings mismatch, invalid `HostContinuation` or
  `sampling.safetensors`): raises `ValueError` naming the file and key **before**
  mutating any live object or writing `state/current.json` or milestones.

## Etude dependency

PR #141 (`af71c109`) is on Etude main after v0.4.5; PyPI latest is 0.4.5.
Landing needs `etude-sdk` 0.4.6 published and `pyproject.toml` pinned to
`>=0.4.6,<0.5` (and `==0.4.6` in `play-runtime`). Slice 1 needed no SDK change. Slices 2–3 use
Etude `af71c109` via a temporary uv source; Slice 4 swaps to the PyPI pin once
0.4.6 is cut (flagged in `scratch/questions.md`).

## Acceptance check

`uv run --extra dev pytest tests/training/test_cross_backend_continuation.py -q`:

1. Torch tracked run, 2 updates, safe stop → MLX resume: MLX learner/EMA
   canonical parameters, Adam `step`/moments, counts and readings match the
   saved files; `run/sampling_reseeded=1` logged on update 3; resumed run
   trains to 4 updates and publishes.
2. MLX → Torch, same assertions (including exact `param_steps` restoration).
3. Torch CUDA/MPS/CPU sampling-domain mismatch (e.g. simulated `torch:cuda`
   domain resumed on `torch:cpu`): refuses tampered `sampling.safetensors`,
   reseeds cleanly when intact, logs `run/sampling_reseeded=1`.
4. MLX save/resume in a subprocess with `sys.modules["torch"] = None`.
5. Legacy Torch object-registration fixture (`learner.pt`/`optimizer.pt`/`ema.pt`)
   resumes on Torch (next save writes canonical layout) and refuses with the
   actionable message on MLX.
6. Torch CPU safe-stop matches uninterrupted Torch CPU bit-for-bit across
   `continuation.json`, `learner.safetensors`, and `sampling.safetensors`.

Plus updated existing tests:
- `tests/training/test_mlx_continuation.py`
- `tests/training/test_self_play_loop.py` (update the two direct
  `run.resume(learner=..., optimizer=..., ema=...)` inspection blocks)
- `tests/training/test_snapshot_rollouts.py` (update line 1832 inspection block)
- `tests/training/test_numerical_health.py` (read `learner.safetensors` +
  `sampling.safetensors`)
- `tests/test_package_boundaries.py`

## Demo

Run `bounded_self_play` on Torch to 2 updates in a tmp `ETUDE_OUT`, then the
same run with `--backend mlx` to 4; `ls` of the save folder shows the four
files (`state.json`, `continuation.json`, `learner.safetensors`,
`sampling.safetensors`) and the MLX run continues at update 3 with
`run/sampling_reseeded=1`.

## Slices

1. **Built — portable learner, EMA and Adam.** Torch and MLX exchange canonical arrays and per-parameter clocks. CPU tests passed for Torch → Torch (exact), Torch → MLX and MLX → Torch (next update within float32 tolerance), unsigned safetensors, invalid moments and the existing MLX damped-update parity check.
2. **Next — unified continuation & loop wiring.** Add `export_continuation` and
   `restore_continuation` in `manabot/training/state/continuation.py`, sampling
   hooks + `restore_legacy` in both backends, update `Continuation` in
   `training/state/backend.py`, wire `run.resume(export=, restore=, extra=)`
   in `manabot/training/loops/self_play.py`, remove `training.safetensors` and
   `etude.train.state.read_state` from `backends/mlx/state.py`. Update
   `test_mlx_continuation.py`, `test_self_play_loop.py`,
   `test_snapshot_rollouts.py`, and `test_numerical_health.py` to pass against
   the 4-file canonical layout.
3. **Cross-backend, cross-device, Torch safe-stop, and legacy tests + docs.**
   Add `tests/training/test_cross_backend_continuation.py` (acceptance 1–6) and
   verify docs against the implementation. Verify `tests/test_package_boundaries.py`.
4. **Etude 0.4.6 pin** once released; drop the temporary uv source.

## Docs written ahead of implementation

Updated in place:
- `docs/training.md` (run directory layout, `## Recovery`, `### What resume does not promise`, `### Portable learner continuation`)
- `manabot/backends/README.md` (MLX summary row, collector RNG, `### Continuing a run`, checkpoint/continuation split)
- `manabot/backends/mlx.md` (`### Tracked training and recovery`, precision resume contract)
- `docs/training-loop.md` (snapshot-bank recovery with canonical continuation)
- `manabot/README.md` (Etude SDK 0.4.6 minimum)

### Questions surfaced while writing docs

- **Local CLI backend/device switching (`etude train --resume` vs `etude continue`):**
  `SelfPlayRecipe` declares `backend` and `device` on the recipe config, and
  `etude train RECIPE --resume` refuses any config change against the saved run.
  Switching backend or device therefore goes through
  `etude continue --run RUN --until updates=N --config '{"backend":"mlx","device":"mps"}'`
  (launched runs) or direct Python `self_play(..., backend=...)` against the same
  `ETUDE_OUT`. Confirm whether that is the intended CLI surface, or whether
  local runs also need a way to switch `backend`/`device` without `etude continue`.
- **Canonical `precision` spelling in `learner.safetensors`:**
  `SelfPlay.precision` uses `"float32" | "tf32" | "bf16"` while
  `MLXActorCritic.precision` uses `"float32" | "bfloat16"`. Record the
  `SelfPlay.precision` spelling (`"float32"`, `"tf32"`, `"bf16"`) in
  `learner.safetensors` so CUDA Torch `bf16` → MLX `bf16` passes exact precision
  admission.
