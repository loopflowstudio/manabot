# MLX backend (Apple Silicon pilot)

`manabot.backends.mlx` runs a saved `ActorCritic` policy, and its PPO learner
step, in [MLX](https://ml-explore.github.io/mlx/) on Apple Silicon. It loads
the shared checkpoint container ([backends](README.md#convert-a-pytorch-checkpoint))
and must agree with PyTorch numerically before it is used for any timing
([parity check](README.md#parity-check)). Training, evaluation and the CLI
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
And the PPO surrogate's likelihood ratio is float32, where PyTorch uses
float64 (the GPU has none); the two differ only for |log ratio| > 88.

## Layout mapping

`checkpoint.py` maps the container's canonical parameters to MLX's layout:

- **Attention.** The canonical packed in-projection `qkv_weight` `[3·d, d]` and
  `qkv_bias` `[3·d]` (PyTorch's `in_proj_*`) split, in that order, into
  `query_proj`, `key_proj` and `value_proj` of MLX's `MultiHeadAttention`.
- **GRU.** MLX's `GRU` layout holds `Wx`, `Wh`, one input bias `b` and a hidden
  bias `bhn` for the new gate only. Both order gates reset, update, new, so the
  weights copy unchanged; the reset and update hidden biases fold into `b`,
  which is exact because those gates add both biases before the sigmoid, and
  the new gate's stays as `bhn`.
- **Containers.** `nn.Sequential` children `N` become `layers.N`.

Every canonical parameter maps to MLX arrays, or loading fails naming the key.

## Learner step

`manabot.backends.mlx.learner.MLXLearner` computes the same damped PPO loss as
`manabot.training.updates.damped_policy` over one minibatch (its terms are in
`rl.py`), clips the global gradient norm and applies Adam with the learner's
settings. It runs eagerly: packed GRU schedules and legal-action counts change
with every batch, so a compiled graph would retrace per call; only the GRU cell
is compiled. It exists to measure the step and does not write checkpoints;
training continuity stays with the PyTorch learner.

Results, machine and cost are in
[the MLX pilot report](../../experiments/mlx_pilot.md).
