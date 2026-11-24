# MLX backend (Apple Silicon pilot)

`manabot.backends.mlx` runs a saved `ActorCritic` policy, and the damped
update that trains it, in [MLX](https://ml-explore.github.io/mlx/) on Apple
Silicon. It loads
the policy files training writes ([policy checkpoints](README.md#policy-checkpoints))
and uses PyTorch as its numerical reference
([parity check](README.md#parity-check)). The ordinary pilot gates timing on
those checks. The [MPS retest](../../experiments/reports/mps_retest.md) explicitly
measures approximate half-precision arms too; it reports failing tolerances
alongside timing and does not treat execution as numerical equivalence. Local
evaluation and baseline verification can also run through MLX in explicit CPU
float32 without importing PyTorch; PyTorch remains the default for evaluation
and the numerical parity reference.

```python
from manabot.backends import resolve

backend = resolve("mlx")
policy, obs_space = backend.load_policy(".runs/example/self-play-raw.safetensors")
policy.cache_programs()  # inference with fixed weights: encode the catalog once
logits, values = policy.evaluate(policy.prepare(observation, inference=True))
```

`observation` is the dictionary of numpy buffers `VectorEnv` already produces.
`prepare` checks it as the PyTorch policy does and builds the same `Prepared`
batch (`manabot.policy.prepare`): every integer lookup (definition rows, record
vocabulary, legal actions, packed GRU schedules, shared-effect routes) happens
on the host, so the MLX graph holds only gathers, scatters and arithmetic.
`inference=True` drops the object and fact
columns no row uses, exactly as PyTorch CPU inference does. `evaluate` returns
NumPy logits `[B,A]` (-1e8 off the legal support) and values `[B]`, like the
PyTorch policy; calling `policy(batch)` returns MLX arrays, with values `[B,1]`.

`backend.load_policy` reads only the container. An older `.pt` policy raises
`NotAContainer`, saying to
[convert it](README.md#convert-an-older-pt-checkpoint) first:

```sh
uv run python -m manabot.backends.convert .runs/example/checkpoint.pt
```

## Install

The `mlx` extra is optional and installs only on macOS with Apple Silicon:

```sh
uv sync --extra mlx
```

Importing `manabot.backends.mlx` without it raises `ImportError` naming the
extra. It is the only package that imports MLX, and no other manabot module
imports it, eagerly or lazily (`tests/test_package_boundaries.py` enforces this).

## What the backend covers

MLX supports tracked and fresh scalar damped self-play with catalog record ids, table
attention and a masked-mean critic. Inputs v9 through v11 use the same saved
column readings as Torch. The model supports stream normalization, focus and
policy FiLM, and v11 stack and known-library rows. Collection and learning
support bucketed shapes and bounded execution chunks. The baseline, local and Elemental recipes accept `backend="mlx"` for training and recovery.

Run one fresh baseline iteration on Apple Silicon with the recipe’s model,
learning rule and both-seat routing. Give each run a new output directory.

```python
from pathlib import Path

from manabot.backends import resolve
from manabot.recipes.baseline import BaselineConfig, settings
from manabot.training.loops.self_play import fresh_iteration

recipe = settings(BaselineConfig(device="mps")).model_copy(
    update={"streams": 2, "games_per_stream": 1, "bucketed_shapes": True}
)
result = fresh_iteration(
    recipe, backend=resolve("mlx"), seed=19,
    out=Path(".runs/mlx-baseline"),
)
print(result.checkpoint, result.counts.learner_transitions)
```

The ordinary self-play loop and `fresh_iteration` share policy creation,
collection, learning, EMA and milestone export. The output directory holds
`policies/00000001-raw.safetensors` and, when averaging is enabled,
`policies/00000001-ema.safetensors`. Both readers can load either file.
The returned policies are the live sources of those exports. Frozen-choice inspection
reads them on MLX using the NumPy corpus and its offer and object permutations.

### Tracked training and recovery

Run the bounded local recipe through one update on MLX, then continue the same
run from its saved update boundary:

```sh
uv run --extra mlx etude train manabot/recipes/local.py --config '{"backend":"mlx","device":"mps"}' --until updates=1
uv run --extra mlx etude train manabot/recipes/local.py --resume --until updates=2
```

Each attempt writes raw and EMA policy exports under `out/policies/`
(`00000001-raw.safetensors`, `00000001-ema.safetensors`, and after resume
`00000002-raw.safetensors`, `00000002-ema.safetensors`) and commits an atomic
update save selected by `out/state/current.json`. A stop before the first update
saves update 0. Both backends write the following files under
`out/state/save-NNNNNNNN/`.

```text
out/state/save-00000001/
  state.json            Etude clocks, seed, commit, file inventory and Counts
  continuation.json     framework-free host state (stream roots/prefixes, host and minibatch RNGs)
  learner.safetensors   canonical float32 raw/EMA weights, Adam state and integrity manifest
  sampling.safetensors  backend sampling RNG arrays keyed by sampling_domain
```

- `continuation.json` (framework-free host state, schema 2): global-slot stream
  state (0-action roots for `complete-games`, committed action prefixes for
  `exact-quota`, parked fresh roots for snapshot vectors), `learned_game_rows`,
  the opponent RNG when `behavior="random"`, the minibatch RNG, and Python/NumPy
  RNGs;
- `learner.safetensors` (`schema="canonical-v1"`): canonical float32 raw and
  optional EMA weights, Adam `step` and moments, per-parameter `param_steps`,
  `backend`, `precision`, `has_ema`, observation config, lookup-buffer digests,
  and SHA-256 digests binding `Counts`, `continuation.json` and
  `sampling.safetensors`;
- `sampling.safetensors`: backend sampling RNG arrays (`rng.collector`,
  `rng.mlx` for `sampling_domain="mlx"`; `rng.collector`, `rng.torch.cpu`, and
  optional `rng.torch.cuda` / `rng.torch.mps` for Torch domains).

Because both backends write this layout, a launched Torch run can continue on
MLX when the model and precision match and the parent's worker pool can run
MLX on the requested device.

```sh
uv run --extra mlx etude continue --run RUN --until updates=4 --config '{"backend":"mlx","device":"mps"}'
```

Before mutating weights or opening native streams, resume validates model
settings, saved column readings, precision, EMA presence, lookup-buffer digests,
schema, the `Counts`/`continuation.json`/`sampling.safetensors` digests,
parameter and Adam names, dtypes, shapes, finiteness and `param_steps` bounds.
It restores weights, EMA and Adam state, reopens streams directly into their
committed roots or replayed action prefixes (verifying every slot's native state
digest, chooser and decision-memory cursors, including across changed worker
counts), and installs host RNGs. Prefix replay rebuilds native game statistics
and preserves any recorded policy probabilities and values. Policy cutoff counts
and durable game records resume with live play.

Within the same `sampling_domain` (`"mlx"` across CPU/MPS), `sampling.safetensors`
restores collector and MLX RNGs too: on CPU float32, stopping between updates
(via `--until` or `ETUDE_STOP_FILE`) and resuming reproduces the next update's
native decisions, work counters, weights, EMA and Adam arrays identically. Each
attempt measures its own elapsed time. On Metal, stream keys, slots, action
traces, state digests and restored arrays match
identically, while subsequent GPU arithmetic follows Metal's float32/bf16
repeatability.

Across sampling domains (Torch → MLX or MLX → Torch), resume still verifies
`sampling_sha256` (tampering fails closed), ignores the incompatible generator
arrays, reseeds sampling deterministically from `(run.seed, counts.updates)`,
and logs `run/sampling_reseeded=1` on the first resumed update. On Torch → MLX,
a conditional parameter can have fewer Adam updates than the global step.
MLX restores its original moments and uses its own update count for bias
correction, including on its first gradient. The first resumed update reports
the number of these parameters in `run/adam_steps_aligned`.

Legacy Torch saves (`learner.pt` / `optimizer.pt` / `ema.pt`) are refused on MLX
with `"this save was written by Torch object registration; resume once on Torch
to write a portable save"`. Pre-unification MLX `training.safetensors` saves are
refused explicitly on both backends. A missing file, digest mismatch or
nonfinite Adam state fails admission before the loop installs learner tensors.
A corrupt action prefix or incompatible stream geometry fails during collector
reconstruction, before the loop can update `state/current.json` or publish a
milestone. Portable policy exports under `out/policies/`
remain separate and can start a fresh learner on either backend.

A numerical failure ends the attempt. Under `numerical-incident/`, the JSON
record names the invariant (`nonfinite_gradients` before Adam,
`nonfinite_parameters_after_step` after Adam), attempted update and last policy
exports. `tensors.safetensors` retains bounded evidence from up to 16 rollout
rows within 32 MiB. `state.safetensors` holds the failure-time model, Adam, EMA
and RNG state when their tensors fit within 512 MiB.

### Local evaluation and baseline verification

Score an exported policy against an anchor or another policy through MLX on CPU
without importing PyTorch:

```sh
uv run --extra mlx python -m manabot.eval.local \
  --backend mlx \
  --checkpoint ~/.etude/track/local/out/policies/00000002-ema.safetensors \
  --opponent heuristic --deal-seed 101 --out .runs/local-mlx-check --seconds 300
```

```text
completed: 1/1 complete deals; 4/4 games; evidence: .runs/local-mlx-check/panel
score: 0.500 (1 deal block; uncertainty unavailable from a single deal); local-check-not-admission
```

In Python, `manabot.eval.evaluation.evaluate_local(..., backend_name="mlx")`
runs the same scalar CPU game children through MLX, and
`manabot.recipes.baseline.evaluate` uses `config.backend`:

```python
from pathlib import Path

from manabot.recipes.baseline import BaselineConfig, evaluate

report = evaluate(
    BaselineConfig(backend="mlx", device="mps", updates=2),
    Path("~/.etude/track/baseline/out").expanduser(),
    seed=0,
)
print(report.learning.verdict, report.learning.gain)
```

Both resolve MLX with explicit `precision="float32"` on CPU so width ≥ 256
checkpoints do not switch to `bfloat16` during CPU verification. `backend_name`
is execution-only and does not enter `EvaluationRequest` or `EvaluationKey`, so
registration digests and existing evaluation directories stay unchanged.

Fresh construction and checkpoint loading both call `require_supported`.
It reports unsupported settings together, including belief inputs, decision
memory, recent events, other critic layouts and disabled attention.

The modules of `policy.py` mirror the PyTorch model one to one, so a parameter
path in MLX names the same component as in the torch state dict:

- object embeddings and the perspective vector;
- the catalog GRU and fact programs;
- shared-effect routing;
- the attention stack;
- action heads, name candidates and the critic.

Two departures from the PyTorch forward are exact up to float rounding. The
catalog GRU always encodes every definition, where PyTorch learning encodes
only the rows a batch references; the others never reach the loss either way.
And the clipped surrogate's likelihood ratio is float32, where PyTorch uses
float64 (the GPU has none). Ordinary values can differ by rounding; a
float32 ratio can also overflow sooner, which the numerical checks refuse.

## Layout mapping

`checkpoint.py` maps the container's canonical parameters to MLX's layout:

- **Attention.** The canonical packed in-projection `qkv_weight` `[3·d, d]` and
  `qkv_bias` `[3·d]` (PyTorch's `in_proj_*`) split, in that order, into
  `query_proj`, `key_proj` and `value_proj` of MLX's `MultiHeadAttention`.
- **GRU.** `PackedGRU` holds `Wx`, `Wh`, `input_bias` and `hidden_bias`, with
  gates ordered reset, update, new. All four tensors copy unchanged. Keeping
  both biases preserves their independent Adam moments and their separate
  contributions to the global gradient norm used for clipping.
- **Containers.** `nn.Sequential` children `N` become `layers.N`.

Every canonical parameter maps to MLX arrays, or loading fails naming the key.

## Learner

The backend carries no loss of its own. `manabot.backends.mlx.rl` has the
torch backend's functions that the scalar damped update calls, over MLX
arrays, and
`manabot.backends.mlx.learner.Learner` differentiates whatever loss an update
hands it, clips the global gradient norm and applies Adam (eps 1e-5, bias
correction). The damped update runs on it unchanged:

```python
import manabot.backends.mlx as backend
from manabot.recipes.builders import move_learning
from manabot.training.updates.damped_policy import update_damped_iteration

learner = backend.learner.Learner(policy, minibatch_rows=64)
diagnostics = update_damped_iteration(
    backend, learner, batch, move_learning(), iteration=1
)
```

`batch` is a `RolloutBatch` as torch collection writes it. Check diagnostics
against the torch backend on the intended batch, weights and precision using
the parity command below. Returns and advantage selection run on the host in NumPy
(`manabot.policy.returns`), and the surrogate ratio is float32
([what differs](README.md#algorithms-are-written-once)).

`Learner.step(..., recorder=observer)` accepts the shared `StepRecorder`
protocol (`manabot.training.updates.health.StepRecorder`). On sampled steps (the
first step of iteration 1 and every 25th iteration under `OptimizerHealth`),
MLX retains zero-copy pre-step parameter references, computes per-leaf squared
sums on device in float32 alongside the finite check, and reduces them in host
float64 to record pre/post-clip gradient norms, critic readout (`value_head.`,
`value_readout.`) and optional `value_token` norms, missing/zero/nonzero/frozen
tensor counts, and absolute and relative L2 parameter displacement. MLX returns
zero gradients for unused trainable leaves, which count as zero tensors. A
missing tensor means the gradient tree has no entry for a trainable parameter.
The recorder keeps the gradient sample if Adam produces a nonfinite parameter.
Unsampled steps skip the measurement reductions and weight snapshots.
Disabling sampling preserves the collected actions and learned weights.

The step runs eagerly: packed GRU schedules and legal-action counts change
with every batch, so a compiled graph would retrace per call; only the GRU cell
is compiled. Collection samples once per group across execution chunks.
Portable exports contain the policy’s float32 parameters and lookup buffers.

Results, machine and cost are in
[the MLX pilot report](../../experiments/reports/mlx_pilot.md).

## Compute precision

A policy built without a precision follows the
[one width rule](../../docs/training.md#reduced-precision) every GPU backend
shares: bfloat16 at `hidden_dim` 256 or more (`BF16_MIN_HIDDEN_DIM`), float32
below it (`manabot.backends.mlx.arrays.default_precision`). An explicit
precision always wins. Fresh and tracked self-play use `SelfPlay.precision`,
including when starting from a portable checkpoint. An explicit precision on
`resolve("mlx", precision=...)` must agree with the recipe. Resuming any tracked
run requires the precision recorded in `learner.safetensors`.

```python
from manabot.backends import resolve
from manabot.backends.mlx.learner import Learner

backend = resolve("mlx", precision="bfloat16")
policy, obs_space = backend.load_policy(".runs/example/checkpoint.safetensors")
learner = Learner(policy, minibatch_rows=64)  # inherits the policy's precision
# Alternatively: Learner(policy, minibatch_rows=64, precision="float32")
```

`MLXActorCritic`, `assemble` and `build` also accept `precision`:
`"float32"` or `"bfloat16"`; omitted, each policy takes its width's
default. The parity command always names its precision (float32 unless
`--precision` says otherwise). It is an execution option, not a change
to the checkpoint's stored float32 weights. `policy.set_precision(...)` changes
both inference and learning and invalidates cached catalog encodings.

Weights and floating activations use the selected compute dtype inside the
forward; gradients return through those casts to float32 master weights.
Loss arithmetic, global clipping and Adam state remain float32; bfloat16 keeps
float32's exponent range, so no loss scaling is used. A nonfinite objective or
gradient refuses the step before changing parameters or Adam state; it never
silently skips a step. Logits, values, log-softmax and sampled action log probabilities
are float32 for both behavior and learner evaluation. Both paths use
`rl.policy_logs` and `rl.action_logs`, promoting before normalization.

Public forwards select precision automatically. Collection holds
`policy.forward_precision()` across all draws from the frozen policy and
reuses its encoded catalog. Leaving collection restores the master weights
and clears the catalog before learning. Custom inference loops can use the
same context manager while their weights are fixed.

Check a checkpoint and real capture with the selected dtype:

```sh
uv run --extra mlx python -m manabot.backends.parity capture/initial.pt --capture capture --backend mlx --precision bfloat16
```

| Precision | Legal logits, max absolute | Values, max absolute | Loss, relative | Full-gradient cosine |
|---|---:|---:|---:|---:|
| bfloat16 | 0.001 | 0.01 | 1% | ≥ 0.99 |

The bound was calibrated on the retained real-input w384/w768 checks in
the [MPS retest](../../experiments/reports/mps_retest.md), source `140eec28d`. It
remains a numerical gate; passing those cases does not establish general
equivalence or equivalent optimization trajectories.
Half still requires a finite clipped Adam step and reports the minimum
per-tensor cosine and the maximum post-step parameter difference. It does not
apply float32's 1e-6 post-step bound: the retained half deltas exceed it.
Float32 keeps its existing per-tensor cosine and post-step gates unchanged.

Run this preflight on the intended weights and inputs: half support, default
or explicit, means the API executes that dtype, not that an input passes its
gate or trains equivalently. Float32 passed each real-capture check below. Half did not:

- Two-row captures: bfloat16 failed once at 1.07% relative loss
  difference against 1%, and passed on one later capture.
- Four 8-row minibatches at w64l2 and w384l8: bfloat16 at w64 reached a
  full-gradient cosine of 0.9315 against ≥ 0.99; bfloat16 at w384 failed only
  the relative-loss gate, on near-zero losses.

The [report's QA table](../../experiments/reports/mps_retest.md#later-small-capture-qa-2026-11-13)
has each measurement. Tolerances are unchanged, and the relative-loss gate has
no absolute floor.
