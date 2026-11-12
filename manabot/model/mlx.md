# MLX policy and learner (Apple Silicon pilot)

`manabot.model.mlx` runs a saved `ActorCritic` policy, and its PPO learner step,
in [MLX](https://ml-explore.github.io/mlx/) on Apple Silicon. It reads the same
checkpoints the PyTorch model writes and must agree with them numerically before
it is used for any timing. Training, evaluation and the CLI still run the
PyTorch model; nothing in the PyTorch path imports MLX.

```sh
uv run --extra mlx python -m manabot.model.mlx.convert .runs/example/checkpoint.pt
```

```text
read   .runs/example/checkpoint.pt  (180 tensors)
wrote  .runs/example/checkpoint.mlx.safetensors
```

```python
from manabot.model.mlx.checkpoint import load_mlx_policy

policy, obs_space = load_mlx_policy(".runs/example/checkpoint.mlx.safetensors")
policy.cache_programs()  # inference with fixed weights: encode the catalog once
logits, values = policy(policy.prepare(observation, inference=True))
```

`load_mlx_policy` validates the same world binding as
`manabot.model.checkpoint.load_checkpoint` and fails with the same errors; a
refusal names the revision that wrote the original torch checkpoint.
`observation` is the dictionary of numpy buffers `VectorEnv` already produces,
admitted as for the PyTorch policy; `prepare` does every integer lookup
(definition rows, record vocabulary, legal actions, packed GRU schedules,
shared-effect routes) on the host, so the MLX graph holds only gathers,
scatters and arithmetic. `inference=True` drops the object and fact columns no
row uses, exactly as PyTorch CPU inference does. Outputs are MLX arrays:
logits `[B,A]` (-1e8 off the legal support) and values `[B,1]`, where the
PyTorch policy returns values `[B]`; reshape before comparing the two.

## Install

The `mlx` extra is optional and installs only on macOS with Apple Silicon:

```sh
uv sync --extra mlx
```

Importing `manabot.model.mlx` without it raises `ImportError` naming the extra.
No other manabot module imports it, eagerly or lazily
(`tests/test_package_boundaries.py` enforces this).

## What the port covers

The modules mirror the PyTorch model one to one, so a parameter path in MLX
names the same component as in the torch state dict:

- object embeddings and the perspective vector;
- the catalog GRU and fact programs;
- shared-effect routing;
- the attention stack;
- action heads, name candidates and the critic.

It covers what the w160l8 regime uses: a catalog world with catalog record
ids, table attention, a masked-mean scalar critic and policy input v9. A
checkpoint that enables anything else (belief inputs, policy history, the
pooled token stream, another critic, FiLM, input v10) is refused at conversion
with the unsupported field named, rather than run partially. Checkpoints
of the `local` recipe use FiLM and are refused.

Two departures from the PyTorch forward are exact up to float rounding. The
catalog GRU always encodes every definition, where PyTorch learning encodes
only the rows a batch references; the others never reach the loss either way.
And the PPO surrogate's likelihood ratio is float32, where PyTorch uses
float64 (the GPU has none); the two differ only for |log ratio| > 88.

## Checkpoint conversion

The converter writes one [safetensors](https://huggingface.co/docs/safetensors)
file of float32 arrays. Its metadata holds the checkpoint's model, observation
and world configuration, its architecture receipt and the `source` record of
the revision that wrote it, as JSON strings, and the torch file it was
converted from under `converted_from`. Three layouts differ between the frameworks
and are mapped explicitly:

- **Attention.** PyTorch's `nn.MultiheadAttention` packs query, key and value
  in-projections into one `in_proj_weight` `[3·d, d]` and `in_proj_bias`
  `[3·d]`. MLX's `MultiHeadAttention` holds separate `query_proj`, `key_proj`
  and `value_proj`; the converter splits the packed tensors in that order.
- **GRU.** PyTorch stores `weight_ih`, `weight_hh`, `bias_ih` and `bias_hh`;
  MLX's `GRU` layout holds `Wx`, `Wh`, one input bias `b` and a hidden bias
  `bhn` for the new gate only. Both order gates reset, update, new, so the
  weights copy unchanged; the converter folds the reset and update hidden
  biases into `b`, which is exact because those gates add both biases before
  the sigmoid, and keeps the new gate's as `bhn`.
- **Containers.** `nn.Sequential` children `N` become `layers.N`.

Integer buffers (catalog programs, record vocabulary, naming joins) are stored
under `buffer/<torch name>`. Every torch tensor maps to exactly one MLX array
or buffer, or conversion fails naming the key; nothing is dropped silently.

## Parity check

```sh
uv run --extra mlx python -m manabot.model.mlx.parity capture/initial.pt --capture capture
```

```text
minibatch   0  B  46  logits 5.8e-09  values 2.4e-07  loss 1.4e-07  min cosine 1.000000 (...)
...
forward    logits  max |Δ| 7.7e-09  (tol 1e-4)  ok
forward    values  max |Δ| 3.6e-07  (tol 1e-4)  ok
collection logits  max |Δ| 7.0e-09  (tol 1e-4)  ok
collection values  max |Δ| 3.3e-07  (tol 1e-4)  ok
learner    loss    rel Δ  0.0000%  (tol 0.1%)  ok
learner    grads   min cosine 0.999940 over 201 tensors (tol 0.999)  ok
one step   params  max |Δ| 9.4e-05, update min cosine 0.944017 (informative)
parity: pass
```

The check runs the PyTorch model in float32 on CPU as the reference, with the
training code's own loss, and the MLX port on the same inputs: a capture
directory written by `experiments/mlx_pilot.py capture` (real learner
minibatches with their targets and settings, and real collection calls). It
compares logits on legal actions and values (absolute difference), the PPO
loss (relative difference) and each parameter's gradient (cosine
similarity). Tolerances are fixed: logits and values within 1e-4, loss within
0.1%, every gradient cosine at least 0.999. A tensor whose gradient norm is
below 1e-6 of the global norm in both frameworks counts as zero: the
policy-score bias and every attention key bias have analytically zero
gradients (softmax ignores a shift shared by all its inputs), so both sides
hold rounding noise. The check exits 1 naming the worst component when any
comparison fails, and 2 when the checkpoint or capture cannot be read or the
checkpoint is refused; a timing taken from a model that fails it is not
reported.

## Learner step

`manabot.model.mlx.learner.MLXLearner` computes the same damped PPO loss as
`manabot.training.learning.damped_policy` over one minibatch, clips the global
gradient norm and applies Adam with the learner's settings. It runs eagerly:
packed GRU schedules and legal-action counts change with every batch, so a
compiled graph would retrace per call; only the GRU cell is compiled. It exists to measure the step and does
not write checkpoints; training continuity stays with the PyTorch learner.

Results, machine and cost are in
[the MLX pilot report](../../experiments/mlx_pilot.md).
