# MLX backend (Apple Silicon pilot)

`manabot.backends.mlx` runs a saved `ActorCritic` policy, and the damped
update that trains it, in [MLX](https://ml-explore.github.io/mlx/) on Apple
Silicon. It loads
the policy files training writes ([policy checkpoints](README.md#policy-checkpoints))
and uses PyTorch as its numerical reference
([parity check](README.md#parity-check)). The ordinary pilot gates timing on
those checks. The [MPS retest](../../experiments/reports/mps_retest.md) explicitly
measures approximate half-precision arms too; it reports failing tolerances
alongside timing and does not treat execution as numerical equivalence. Evaluation uses PyTorch.

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

Run the bounded local recipe through one update, then continue the same run.

```sh
uv run --extra mlx etude train manabot/recipes/local.py --config '{"backend":"mlx","device":"mps"}' --until updates=1
uv run --extra mlx etude train manabot/recipes/local.py --resume --until updates=2
```

Etude stages `training.safetensors` and `state.json` in one save and commits
the folder through `state/current.json`. It restores raw and EMA weights,
Adam moments, learning rate, step and cumulative counts. Restore checks the
model settings, saved column readings, lookup tables, backend, schema and
array names, dtypes and shapes before loading the weights. Game streams and
sampling start again at the saved update. Portable policy exports can start
a fresh learner on either backend.

A numerical failure ends the attempt. Under `numerical-incident/`, the JSON
record names the invariant, attempted update and last policy exports.
`tensors.safetensors` retains bounded evidence from up to 16 rollout rows
within 32 MiB. `state.safetensors` holds the failure-time model, Adam, EMA
and RNG state when their tensors fit within 512 MiB.

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

The MLX learner encodes the catalog definitions referenced by each prepared
batch. With bucketed shapes it selects those definitions after trimming the
observation and runs attention over the object columns used by at least one
row. It restores the bucket's object coordinates before the critic and action
focus lookups. Inference can cache the full catalog while the weights are fixed.

The clipped surrogate computes its likelihood ratio in float32 on MLX and
float64 on PyTorch. The Apple GPU lacks float64. Ordinary ratios can differ
by rounding, and the numerical checks reject a float32 ratio that overflows.

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
(`manabot.policy.returns`), the surrogate ratio is float32, and the sampled
gradient detail of `OptimizerHealth` is absent
([what differs](README.md#algorithms-are-written-once)).

The step runs eagerly: packed GRU schedules and legal-action counts change
with every batch, so a compiled graph would retrace per call; only the GRU cell
and its reverse are compiled. Each `PackedGRU` gathers and projects its tokens
once, and its recurrence is an `mx.custom_function` with a hand-written VJP:
autograd never traces the per-step slices, whose whole-input zero cotangents
would make the backward quadratic in the number of steps. Collection samples once per group across execution chunks.
Portable exports contain the policy’s float32 parameters and lookup buffers.

### Profile captured learner minibatches

Replay the learner arrays and initial policy from a Torch capture on the Apple GPU.
The first minibatch warms the path. Each subsequent batch contributes one sample
to the reported medians.

```sh
uv run --extra mlx python -m scripts.profile_mlx_learner capture --bucketed --precision bfloat16 --out profile.json
```

Omit `--bucketed` for padded shapes. Precision defaults to float32. The output
records admission and host preparation, full loss and gradients, and clipping
plus Adam. Two controls replay the same loss at the same weights. The first
reuses the catalog table. The second also reuses the attention and MLP outputs,
passing gradients through to the object inputs. Differences between controls
estimate the cost of the skipped work. Each sample records its maximum control
loss difference. Repeated bfloat16 forwards can vary at fixed weights, so the
bfloat16 controls use the existing 1% relative loss tolerance. Only the full
gradient advances Adam.

Results, machine and cost are in
[the MLX pilot report](../../experiments/reports/mlx_pilot.md).

## Compute precision

A policy built without a precision follows the
[one width rule](../../docs/training.md#reduced-precision) every GPU backend
shares: bfloat16 at `hidden_dim` 256 or more (`BF16_MIN_HIDDEN_DIM`), float32
below it (`manabot.backends.mlx.arrays.default_precision`). An explicit
precision always wins. Fresh and tracked self-play use `SelfPlay.precision`,
including when starting from a portable checkpoint. An explicit precision on
`resolve("mlx", precision=...)` must agree with the recipe. Resuming a tracked
MLX run requires the precision in its training save.

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
