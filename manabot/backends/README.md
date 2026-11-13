# Framework backends

A backend runs the manabot policy forward and the learner step of its
updates in one array framework. Backends share the framework-free contract in
[`manabot.policy`](../policy/README.md): host preparation of observation
batches, the checkpoint container and the `BackendPolicy`,
`BackendLearner` and `Backend` protocols. Training, evaluation and the CLI
run the PyTorch backend, which is also the reference every other backend is
checked against.

| Backend | Status | Guide |
| --- | --- | --- |
| `torch` | the reference: training, evaluation, the CLI and Academy | [below](#the-torch-backend) |
| `mlx` | policy and the damped update, parity-checked; measurement only | [mlx.md](mlx.md) |

## Layout

Every backend is one package, `manabot/backends/<name>/`, with at least these
files; a backend may add private modules that its `policy.py` uses (torch's
`live_tokens.py`):

| File | Holds |
| --- | --- |
| `__init__.py` | the import guard naming the optional extra, and the files below as attributes loaded on first use (`backend.rl`) |
| `arrays.py` | conversion to and from NumPy, the float dtype, the device |
| `policy.py` | the forward over a `Prepared` batch: logits and values |
| `rl.py` | the policy terms, losses, returns and selection the updates call, over the framework's arrays |
| `learner.py` | `Learner`: the optimizer, the differentiated loss, the guarded step and precision |
| `rollout.py` | `AdmittedRollout`: a collected batch's observations as the policy reads them |
| `checkpoint.py` | the map from canonical names to the backend's layout, loading, and `BACKEND` |

`BACKEND` in `checkpoint.py` implements `manabot.policy.protocol.Backend`, the
one object `parity` needs. A backend imports its own framework,
`manabot.policy` and the configuration and observation modules; it never
imports another backend (`tests/test_package_boundaries.py` enforces it).
Nothing outside `manabot/backends/` imports a backend other than torch, and
outside `manabot/backends/torch/` only the modules the boundary test lists by
name import torch; that list shrinks as training and evaluation become
backend-generic.

## Algorithms are written once

An update or a loop is written once and takes the backend as an argument:
the backend's module itself, not a class to subclass or a configuration
string. The maintained updates in `manabot/training/updates/` take it first:

```python
import manabot.backends.torch as backend

diagnostics = update_damped_iteration(
    backend, learner, batch, move_learning(), iteration=1
)
```

The damped update runs on torch and on MLX. The same batch and the same
weights give the same diagnostics on both, within the
[parity check](#parity-check)'s bars
(`tests/training/test_update_backends.py` runs it on every installed
backend):

```python
import manabot.backends.mlx as backend

learner = backend.learner.Learner(policy, minibatch_rows=64)
diagnostics = update_damped_iteration(
    backend, learner, batch, move_learning(), iteration=1
)
```

Inside an update, array math goes only through the backend's modules. The
Protocols in `manabot/training/updates/backend.py` state this surface
(`Backend` for the damped update; `PPOBackend`, torch only, adds what PPO
calls; `CategoricalRL`, torch only, the categorical critic), and
`tests/training/test_update_backend_protocols.py` checks every installed
backend against them:

- `backend.arrays`: moving host rows in and out (`from_numpy`, `to_numpy`)
  and `float32` for a forward output.
- `backend.rl`: the policy terms (`legal_mask`, `policy_logs`,
  `action_logs`, `reference_logs`, `damped_policy_loss`), the losses
  (`clipped_surrogate`, `reverse_kl`, `squared_value_loss`,
  `categorical_value_loss`, `entropy`, `clip_fraction`), returns
  (`transition_gae`, `standardized_advantages`, `categorical_lambda_returns`,
  `outcome_probabilities`, `expected_outcome`), selection (`selected_rows`,
  `selection_mask`, `selected_mean`), `constant` for a value with no gradient
  path, and the refusals (`require`, `require_finite`, `float64_operand`,
  `NumericalError`).
- `backend.learner`: its `Learner` holds the policy and optimizer, sets the
  learning rate (`set_learning_rate`), differentiates a loss (`loss`) and
  steps (`step`); `forward_precision` is the context a forward runs in,
  `no_grad` the one for constants and `check_objective` the loss's refusal.
- `backend.rollout`: `AdmittedRollout(agent, batch)` returns a batch's rows as
  the policy reads them (`rows`, `next_observation`), their legal mask
  (`legal`) and the arrays a rejected step retains (`evidence`).

An update writes its minibatch loss as a closure over the policy forward and
hands it to the learner, because a functional framework differentiates a
function, not a value:

```python
def objective():
    logits, values = learner.agent.forward_distribution(obs)
    return minibatch_objective(backend.rl, logits, values, valid, ...)

loss, policy_loss, value_loss = learner.loss(objective)
norm = learner.step(loss, max_grad_norm, recorder=observer)
```

Torch runs the closure once and keeps the graph for `step`; MLX evaluates it
with its gradients (`mlx.nn.value_and_grad`), so the forward also runs once.

NumPy keeps the host-side bookkeeping and everything computed once per batch
on host rows: which rows train, minibatch indexes, action-type counts, batch
health, `selection_diagnostics` and `value_scale`. Logic that is pure NumPy
lives once in `manabot.policy`: `reference_distribution` (`prepare.py`) and
the returns and selection recursions (`returns.py`), which both backends'
`transition_gae` wrap; torch keeps advantage selection on the minibatch's
device (`selected_rows`, `torch.quantile`), matching `returns.quantile` bit
for bit. The
update modules import no framework and no backend, even for annotations
(`tests/test_package_boundaries.py` enforces it); they name the backend's
arrays, devices and observation batches as the opaque `Array`, `Device` and
`Observations`.

A function has the same name, arguments, shapes and per-row results in every
backend that has it; the torch backend's signatures are the reference, and
[training loops](../../docs/training-loop.md#standard-methods-kept-local)
documents them. A backend never carries its own copy of an update's loss:
`manabot.backends.parity` runs `minibatch_objective` through each backend's
`rl` and `Learner`. What still differs between the two backends:

- MLX has no float64 on the GPU, so its surrogate ratio and loss sum are
  float32; torch's are float64.
- MLX's `rl` holds what the scalar damped update calls. It has no categorical
  critic functions (`categorical_lambda_returns`, `categorical_value_loss`,
  `outcome_probabilities`, `expected_outcome`), and the backend refuses a
  categorical checkpoint; it also lacks `standardized_advantages` and
  `selected_rows`, which only PPO calls.
- `OptimizerHealth` counts steps and rejections on every backend; its sampled
  gradient and parameter detail reads torch modules and is absent on MLX.
- The PPO update calls torch tensor methods (`.float()`, `.flatten()`,
  `.detach()`) and torch's `get_value`; only the damped update runs on MLX.

## The torch backend

| Module | Names |
| --- | --- |
| `manabot.backends.torch.policy` | `ActorCritic` |
| `manabot.backends.torch.rl` | `legal_mask`, `policy_logs`, `action_logs`, `reference_logs`, `damped_policy_loss`, `transition_gae`, `standardized_advantages`, `outcome_probabilities`, `expected_outcome`, `categorical_lambda_returns`, `clipped_surrogate`, `reverse_kl`, `squared_value_loss`, `categorical_value_loss`, `entropy`, `clip_fraction`, `selected_rows`, `selection_mask`, `selected_mean`, `constant`, `require`, `require_finite`, `float64_operand`, `NumericalError` |
| `manabot.backends.torch.learner` | `Learner`, `gradient_step`, `update_ema`, `StepRecorder`, `check_objective`, `forward_precision`, `no_grad`, `apply_matmul_precision` |
| `manabot.backends.torch.rollout` | `AdmittedRollout`, `observation_tensors` |
| `manabot.backends.torch.arrays` | `from_numpy`, `to_numpy`, `float32`, `device_of` |
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

`BACKEND` is torch's `manabot.policy.protocol.Backend`, beside the native
torch API: `load` returns a `TorchPolicy` and a `Learner`, whose parameters
and gradients keep torch's names (`torch_name` in `manabot.policy.checkpoint`
maps canonical names back). Until `ActorCritic` reads `Prepared` indices, its
`prepare` builds the shared `Prepared` and keeps the admitted torch
observation on it (`Retained`), so parity can run torch against its own torch
reference.

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
torch  forward logits                           max |Δ| 9.7e-09 (tol 1e-4)  ok
torch  forward values                           max |Δ| 2.7e-07 (tol 1e-4)  ok
torch  collection logits                        max |Δ| 0.0e+00 (tol 1e-4)  ok
torch  collection values                        max |Δ| 0.0e+00 (tol 1e-4)  ok
torch  learner loss                             rel Δ 0.0000% (tol 0.1%)  ok
torch  gradient ...                             min cosine 1.000000 over 169 tensors (tol 0.999)  ok
torch  one step                                 params max |Δ| 7.5e-09, update min cosine 1.000000 (informative)
mlx    forward logits                           max |Δ| 7.7e-09 (tol 1e-4)  ok
mlx    forward values                           max |Δ| 3.6e-07 (tol 1e-4)  ok
mlx    collection logits                        max |Δ| 7.0e-09 (tol 1e-4)  ok
mlx    collection values                        max |Δ| 3.3e-07 (tol 1e-4)  ok
mlx    learner loss                             rel Δ 0.0000% (tol 0.1%)  ok
mlx    gradient ...                             min cosine 0.999940 over 201 tensors (tol 0.999)  ok
mlx    one step                                 params max |Δ| 9.4e-05, update min cosine 0.944017 (informative)
parity: pass (torch, mlx)
```

The check runs the PyTorch model in float32 on CPU as the reference and every
installed backend, torch included, on the same inputs, each side computing
the damped update's `minibatch_objective` through its own backend: a
capture directory written by `experiments/mlx_pilot.py capture` (real learner
minibatches with their targets and settings, and real collection calls). Each
backend loads the reference's weights through the canonical names and is
compared in its own parameter layout. The check compares logits on legal
actions and values (absolute difference), the update's loss (relative difference)
and each parameter's gradient (cosine similarity). Tolerances are fixed:
logits and values within 1e-4, loss within 0.1%, every gradient cosine at
least 0.999. A tensor whose gradient norm is below 1e-6 of the global norm in
both frameworks counts as zero: the policy-score bias and every attention key
bias have analytically zero gradients (softmax ignores a shift shared by all
its inputs), so both sides hold rounding noise. The check exits 1 naming each
failing backend's worst components, and 2 when the checkpoint or capture
cannot be read, the checkpoint is refused or no backend is installed; a
timing taken from a backend that fails it is not reported.
