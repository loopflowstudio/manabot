# MLX backend (Apple Silicon pilot)

`manabot.backends.mlx` runs a saved `ActorCritic` policy, and the damped
update that trains it, in [MLX](https://ml-explore.github.io/mlx/) on Apple
Silicon. It loads
the shared checkpoint container ([backends](README.md#convert-a-pytorch-checkpoint))
and uses PyTorch as its numerical reference
([parity check](README.md#parity-check)). The ordinary pilot gates timing on
those checks. The [MPS retest](../../experiments/mps_retest.md) explicitly
measures approximate half-precision arms too; it reports failing tolerances
alongside timing and does not treat execution as numerical equivalence. Training, evaluation and the CLI
still run the PyTorch model; nothing in the PyTorch path imports MLX.

```sh
uv run python -m manabot.backends.convert .runs/example/checkpoint.pt
```

```python
from manabot.backends.mlx.checkpoint import load_mlx_policy

policy, obs_space = load_mlx_policy(".runs/example/checkpoint.safetensors")
policy.cache_programs()  # inference with fixed weights: encode the catalog once
logits, values = policy.evaluate(policy.prepare(observation, inference=True))
```

`observation` is the dictionary of numpy buffers `VectorEnv` already produces,
admitted as for the PyTorch policy; `prepare` (`manabot.policy.prepare`) does
every integer lookup (definition rows, record vocabulary, legal actions, packed
GRU schedules, shared-effect routes) on the host, so the MLX graph holds only
gathers, scatters and arithmetic. `inference=True` drops the object and fact
columns no row uses, exactly as PyTorch CPU inference does. `evaluate` returns
NumPy logits `[B,A]` (-1e8 off the legal support) and values `[B]`, like the
PyTorch policy; calling `policy(batch)` returns MLX arrays, with values `[B,1]`.

## Install

The `mlx` extra is optional and installs only on macOS with Apple Silicon:

```sh
uv sync --extra mlx
```

Importing `manabot.backends.mlx` without it raises `ImportError` naming the
extra. It is the only package that imports MLX, and no other manabot module
imports it, eagerly or lazily (`tests/test_package_boundaries.py` enforces this).

## What the backend covers

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

`batch` is a `RolloutBatch` as torch collection writes it. The same batch and
weights give the same diagnostics as the torch backend within the parity
bars. Returns and advantage selection run on the host in NumPy
(`manabot.policy.returns`), the surrogate ratio is float32, and the sampled
gradient detail of `OptimizerHealth` is absent
([what differs](README.md#algorithms-are-written-once)).

The step runs eagerly: packed GRU schedules and legal-action counts change
with every batch, so a compiled graph would retrace per call; only the GRU cell
is compiled. The backend does not write checkpoints or collect; training
continuity stays with the PyTorch learner.

Results, machine and cost are in
[the MLX pilot report](../../experiments/mlx_pilot.md).

## Compute precision

Float32 is the default. Both half dtypes are supported explicitly on the policy
and learner; this does not enable an MLX training CLI or change Torch admission.

```python
from manabot.backends.mlx.checkpoint import load_mlx_policy
from manabot.backends.mlx.learner import Learner

policy, obs_space = load_mlx_policy(
    ".runs/example/checkpoint.safetensors", precision="bfloat16"
)
learner = Learner(policy, minibatch_rows=64)  # inherits the policy's precision
# Alternatively: Learner(policy, minibatch_rows=64, precision="float16")
```

`MLXActorCritic`, `assemble`, `build` and `MLXBackend` also accept `precision`:
`"float32"`, `"bfloat16"` or `"float16"`. It is an execution option, not a change
to the checkpoint's stored float32 weights. `policy.set_precision(...)` changes
both inference and learning and invalidates cached catalog encodings.

Weights and floating activations use the selected compute dtype inside the
forward; gradients return through those casts to float32 master weights.
Loss arithmetic, global clipping and Adam state remain float32. Float16 uses
fixed loss scaling of 128; a nonfinite objective or gradient refuses the step
before changing parameters or Adam state. It does not silently skip a step or
adapt the scale. Logits, values, log-softmax and sampled action log probabilities
are float32 for both behavior and learner evaluation. Both paths use
`rl.policy_logs` and `rl.action_logs`, promoting before normalization.

Public forwards select precision automatically. A fixed-weight inference loop
may hold `with policy.forward_precision():` across calls to reuse cast weights;
close it before differentiating or stepping. `Learner.loss` clears inference
catalog caches before differentiation, so catalog parameters still learn.

Check a checkpoint and real capture with the selected dtype:

```sh
uv run --extra mlx python -m manabot.backends.parity capture/initial.pt --capture capture --backend mlx --precision bfloat16
```

| Precision | Legal logits, max absolute | Values, max absolute | Loss, relative | Full-gradient cosine |
|---|---:|---:|---:|---:|
| bfloat16 | 0.001 | 0.01 | 1% | ≥ 0.99 |
| float16 | 0.0001 | 0.002 | 0.2% | ≥ 0.999 |

These bounds cover the retained real-input w384/w768 checks in the
[MPS retest](../../experiments/mps_retest.md), source `140eec28d`; they are
measured-domain numerical gates, not a claim about optimization trajectories.
Half still requires a finite clipped Adam step and reports the minimum
per-tensor cosine and the maximum post-step parameter difference. It does not
apply float32's 1e-6 post-step bound: the retained half deltas exceed it.
Float32 keeps its existing per-tensor cosine and post-step gates unchanged.

Run this preflight on the intended weights and inputs. The bounded real-capture
check of the supported API passed float32 and float16 but rejected bfloat16
at 1.07% relative loss difference against its unchanged 1% gate. That
counterexample is retained in the report. A fresh capture after the native
sync passes all three dtypes; support does not mean every input passes the
numerical gate.
