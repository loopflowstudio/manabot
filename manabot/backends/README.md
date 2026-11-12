# Framework backends

A backend runs the manabot policy forward and its PPO learner step in one
array framework. Backends share the framework-free contract in
[`manabot.policy`](../policy/README.md): host preparation of observation
batches, the checkpoint container and the `BackendPolicy`,
`BackendLearner` and `Backend` protocols. Training, evaluation and the CLI
run the PyTorch backend, which is also the reference every other backend is
checked against.

| Backend | Status | Guide |
| --- | --- | --- |
| `torch` | the reference: training, evaluation, the CLI and Academy | [below](#the-torch-backend) |
| `mlx` | policy and learner step, parity-checked; measurement only | [mlx.md](mlx.md) |

## Layout

Every backend is one package, `manabot/backends/<name>/`, with at least these
files; a backend may add private modules that its `policy.py` uses (torch's
`live_tokens.py`):

| File | Holds |
| --- | --- |
| `__init__.py` | the import guard naming the optional extra; nothing else |
| `arrays.py` | conversion to and from NumPy, the float dtype, the device |
| `policy.py` | the forward over a `Prepared` batch: logits and values |
| `rl.py` | the loss terms of `manabot.training.updates` over the framework's arrays |
| `learner.py` | the optimizer, gradient step and precision |
| `checkpoint.py` | the map from canonical names to the backend's layout, loading, and `BACKEND` |

`BACKEND` in `checkpoint.py` implements `manabot.policy.protocol.Backend`, the
one object `parity` needs. A backend imports its own framework,
`manabot.policy` and the configuration and observation modules; it never
imports another backend (`tests/test_package_boundaries.py` enforces it).
Nothing outside `manabot/backends/` imports a backend other than torch, and
outside `manabot/backends/torch/` only the modules the boundary test lists by
name import torch; that list shrinks as training and evaluation become
backend-generic.

## The torch backend

| Module | Names |
| --- | --- |
| `manabot.backends.torch.policy` | `ActorCritic` |
| `manabot.backends.torch.rl` | `transition_gae`, `standardized_advantages`, `expected_outcome`, `categorical_lambda_returns`, `clipped_surrogate`, `clip_fraction`, `support_logs`, `reverse_kl`, `squared_value_loss`, `categorical_value_loss`, `entropy`, `selection_mask`, `selected_mean`, `NumericalError` |
| `manabot.backends.torch.learner` | `Learner`, `gradient_step`, `update_ema`, `StepRecorder`, `forward_precision`, `apply_matmul_precision` |
| `manabot.backends.torch.arrays` | `from_numpy`, `to_numpy`, `flat_numpy`, `device_of` |
| `manabot.backends.torch.checkpoint` | `load_checkpoint`, `save_checkpoint`, `assemble`, `BACKEND` |

```python
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.backends.torch.learner import Learner

agent, obs_space = load_checkpoint(".runs/example/checkpoint.pt")
learner = Learner(agent, "cpu", minibatch_rows=64)
```

`load_checkpoint` reads a `.pt` file or a container, telling them apart by
their first bytes; `save_checkpoint` writes `.pt` until the container carries
every configuration the torch policy runs (belief inputs, history, FiLM and
the rest refused below).

`BACKEND` is torch's `manabot.policy.protocol.Backend`, NumPy in and out,
beside the native torch API; its parameters and gradients keep canonical
names. Until `ActorCritic` reads `Prepared` indices, its `prepare` builds the
shared `Prepared` and keeps the admitted torch observation on it (`Retained`),
so parity can run torch against its own torch reference.

`rl.py`, `learner.py` and `arrays.py` import nothing else from manabot at
runtime, so another role reuses them unchanged.

## Convert a PyTorch checkpoint

```sh
uv run python -m manabot.backends.convert .runs/example/checkpoint.pt
```

```text
read   .runs/example/checkpoint.pt  (180 tensors)
wrote  .runs/example/checkpoint.safetensors
```

The converter reads the `.pt` file with PyTorch and writes the shared
container beside it: one [safetensors](https://huggingface.co/docs/safetensors)
file of float32 parameters under canonical names, the integer lookup buffers
(catalog programs, record vocabulary, naming joins) under `buffer/<name>`, and
metadata holding the checkpoint's model, observation and world configuration,
its architecture receipt and the `source` record of the revision that wrote it,
as JSON strings, with the torch file under `converted_from`. Canonical names
are the PyTorch state-dict names with two framework idioms renamed: a GRU holds
`input_weight`, `hidden_weight`, `input_bias` and `hidden_bias` (gates reset,
update, new), and attention's packed in-projection is `qkv_weight` /
`qkv_bias` (query, key, value). Every torch tensor becomes one parameter or
buffer, or conversion fails naming the key.

Loading a container validates the same world binding as
`manabot.backends.torch.checkpoint.load_checkpoint` and fails with the same errors; a
refusal names the revision that wrote the original torch checkpoint. A file
that is not a container (a `.pt`, or a conversion without the metadata) raises
`NotAContainer`, saying to convert the `.pt`; `convert` refuses a `.safetensors`
input. Backends
cover what the w160l8 regime uses: a catalog world with catalog record ids,
table attention, a masked-mean scalar critic and policy input v9. A checkpoint
that enables anything else (belief inputs, policy history, another critic,
FiLM, input v10) is refused with every unsupported
field named (`UnsupportedPolicy`), rather than run partially. Checkpoints of
the `local` recipe use FiLM and are refused.

## Parity check

```sh
uv run --extra mlx python -m manabot.backends.parity capture/initial.pt --capture capture
```

```text
mlx minibatch   0  B  46  logits 5.8e-09  values 2.4e-07  loss 1.4e-07  min cosine 1.000000 (...)
...
mlx    forward logits                           max |Δ| 7.7e-09 (tol 1e-4)  ok
mlx    forward values                           max |Δ| 3.6e-07 (tol 1e-4)  ok
mlx    collection logits                        max |Δ| 7.0e-09 (tol 1e-4)  ok
mlx    collection values                        max |Δ| 3.3e-07 (tol 1e-4)  ok
mlx    learner loss                             rel Δ 0.0000% (tol 0.1%)  ok
mlx    gradient ...                             min cosine 0.999940 over 201 tensors (tol 0.999)  ok
mlx    one step                                 params max |Δ| 9.4e-05, update min cosine 0.944017 (informative)
parity: pass (mlx)
```

The check runs the PyTorch model in float32 on CPU as the reference, with the
training code's own loss, and every installed backend on the same inputs: a
capture directory written by `experiments/mlx_pilot.py capture` (real learner
minibatches with their targets and settings, and real collection calls). Each
backend loads the reference's weights through the canonical names and is
compared in its own parameter layout. The check compares logits on legal
actions and values (absolute difference), the PPO loss (relative difference)
and each parameter's gradient (cosine similarity). Tolerances are fixed:
logits and values within 1e-4, loss within 0.1%, every gradient cosine at
least 0.999. A tensor whose gradient norm is below 1e-6 of the global norm in
both frameworks counts as zero: the policy-score bias and every attention key
bias have analytically zero gradients (softmax ignores a shift shared by all
its inputs), so both sides hold rounding noise. The check exits 1 naming each
failing backend's worst components, and 2 when the checkpoint or capture
cannot be read, the checkpoint is refused or no backend is installed; a
timing taken from a backend that fails it is not reported.
