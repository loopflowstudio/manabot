# Framework backends

```python
from manabot.backends.torch.checkpoint import load_checkpoint

policy, space = load_checkpoint(".runs/example/self-play-raw.safetensors")
batch = policy.prepare(obs)              # NumPy in: the buffers VectorEnv produces
logits, values = policy.evaluate(batch)  # NumPy out: logits [B,A], values [B]
```

`obs` is the dictionary of NumPy observation buffers for `B` decisions.
`prepare` checks it and does every integer lookup on the host, returning a
`Prepared` batch (`manabot.policy.prepare`); `evaluate` runs the forward and
returns logits, -1e8 off the legal actions, and each row's value. Every
backend reads the same `Prepared` batch and the same checkpoint file, so the
MLX lines differ only in the import and the loader's name
([mlx.md](mlx.md)).

A backend runs the manabot policy forward and the learner step of its
updates in one array framework. Backends share the framework-free contract in
[`manabot.policy`](../policy/README.md): observation admission, host
preparation, the checkpoint container and the `BackendPolicy`,
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
encoders and `live_tokens.py`):

| File | Holds |
| --- | --- |
| `__init__.py` | the import guard naming the optional extra, and the files below as attributes loaded on first use (`backend.rl`) |
| `arrays.py` | conversion to and from NumPy, the float dtype, the device |
| `policy.py` | the policy: `prepare`, and the forward over a `Prepared` batch (logits and values) |
| `rl.py` | the policy terms, losses, returns and selection the updates call, over the framework's arrays |
| `learner.py` | `Learner`: the optimizer, the differentiated loss, the guarded step and precision |
| `rollout.py` | `AdmittedRollout`: a collected batch's observations as the policy reads them |
| `checkpoint.py` | the map from canonical names to the backend's layout, loading (torch also saves), and `BACKEND` |

`BACKEND` in `checkpoint.py` implements `manabot.policy.protocol.Backend`, the
one object `parity` needs. A backend imports its own framework,
`manabot.policy` and the configuration and observation modules; it never
imports another backend (`tests/test_package_boundaries.py` enforces it).

## Who chooses a backend

Library code receives a backend; it does not import one. An update takes the
backend module as its first argument, and simulation and evaluation take a
policy that satisfies `BackendPolicy`. Only the code that starts a program
chooses:

- `experiments/` and `scripts/`;
- recipe entry points and the CLI;
- Academy's opponent loader (`academy/villain.py`);
- the backends' own tests.

They name a backend once, by import or through the resolver:

```python
import manabot.backends

backend = manabot.backends.resolve("torch")  # the module manabot.backends.torch
learner = backend.learner.Learner(policy, "cpu", minibatch_rows=64)
```

`resolve` imports the framework on first use. A backend whose optional extra
is not installed raises `ImportError` naming the extra; an unknown name raises
`ValueError` listing the backends.

`tests/test_package_boundaries.py` refuses an import of `manabot.backends.*`
anywhere else. The modules that still import the torch backend directly
(belief models, verification, the executor, optimizer persistence) are listed
there by name; the list is compared exactly, so it can shrink and never grow.

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

The damped update runs on torch and on MLX. One minibatch at the same
weights gives the same diagnostics on both, within the
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
  `Prepared` batches on the learner's device (`rows`, `next_observation`),
  their legal mask (`legal`) and the arrays a rejected step retains
  (`evidence`: the prepared batch's arrays by field name, in NumPy).

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
algorithm modules (`damped_policy.py` and `ppo.py`) import no framework and no backend, even for annotations
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
  Health reporting and rollout storage still import torch, so running the MLX
  update currently requires torch installed too.
- The PPO update calls torch tensor methods (`.float()`, `.flatten()`,
  `.detach()`) and torch's `get_value`; only the damped update runs on MLX.

## The torch backend

| Module | Names |
| --- | --- |
| `manabot.backends.torch.policy` | `ActorCritic`, `PolicyInference` |
| `manabot.backends.torch.rl` | `legal_mask`, `policy_logs`, `action_logs`, `reference_logs`, `damped_policy_loss`, `transition_gae`, `standardized_advantages`, `outcome_probabilities`, `expected_outcome`, `categorical_lambda_returns`, `clipped_surrogate`, `reverse_kl`, `squared_value_loss`, `categorical_value_loss`, `entropy`, `clip_fraction`, `selected_rows`, `selection_mask`, `selected_mean`, `constant`, `require`, `require_finite`, `float64_operand`, `NumericalError` |
| `manabot.backends.torch.learner` | `Learner`, `gradient_step`, `update_ema`, `StepRecorder`, `check_objective`, `forward_precision`, `no_grad`, `apply_matmul_precision` |
| `manabot.backends.torch.rollout` | `AdmittedRollout` |
| `manabot.backends.torch.arrays` | `from_numpy`, `to_numpy`, `float32`, `device_of`, `on_device` |
| `manabot.backends.torch.checkpoint` | `load_checkpoint`, `save_checkpoint`, `assemble`, `BACKEND` |

`ActorCritic` is the torch `BackendPolicy`: `load_checkpoint` returns it, and
`prepare` and `evaluate` are its methods. `evaluate` takes the NumPy batch,
moves it to the model's device and runs without gradients. Training calls the
module itself, which reads a `Prepared` batch whose arrays are already
tensors on the model's device:

```python
from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.backends.torch.learner import Learner

agent, obs_space = load_checkpoint(".runs/example/self-play-raw.safetensors")
learner = Learner(agent, "cuda", minibatch_rows=64)

batch = on_device(agent.prepare(obs), "cuda")  # one pinned copy per dtype
logits, value = agent(batch)                   # tensors: logits [B,A], value [B]
```

`agent(batch)` and `forward_distribution(batch)` do no raw-observation
validation or index discovery; `prepare` did both. They raise `TypeError` for
anything but a `Prepared` batch of tensors (an observation dictionary, or the NumPy batch
before `on_device`) and `ValueError` for a batch prepared by a policy with
another model or observation configuration. A batch belongs to a
configuration, not to one instance: a restored, target or averaged copy of
the model reads it.

`prepare` trims a batch's variable axes to bucket sizes when the policy was
built with `bucketed_shapes=True`, and packs live tokens with
`live_tokens=True`. Both are execution options passed to `ActorCritic(...)`
or `load_checkpoint(...)`, not checkpoint fields. Every lookup a batch needs
is measured on the host, so the device forward reads nothing back.

`BACKEND` is torch's `manabot.policy.protocol.Backend`: `load` returns an
`ActorCritic` and a `Learner`, whose parameters and gradients keep torch's
names (`torch_name` in `manabot.policy.checkpoint` maps canonical names
back).

The reusable pieces in `rl.py`, `learner.py` and `arrays.py` depend only on
their own backend, the shared host returns and the profiler at runtime;
they load neither the policy network nor the training configuration.

## Policy checkpoints

```python
from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.env.match import Match

save_checkpoint(
    agent, agent.observation_space, "policy.safetensors",
    player_configs=Match().to_rust(),
)
agent, obs_space = load_checkpoint("policy.safetensors")
```

A policy is saved as one file, the shared container, which every backend
loads. Training writes its policy files with the `.safetensors` suffix
(`self-play-raw.safetensors`, `monitor-3.safetensors`). The file is a
[safetensors](https://huggingface.co/docs/safetensors) archive holding:

- the float32 parameters under canonical names. These are the PyTorch
  state-dict names with two framework idioms renamed: a GRU holds
  `input_weight`, `hidden_weight`, `input_bias` and `hidden_bias` (gates
  reset, update, new), and attention's packed in-projection is `qkv_weight` /
  `qkv_bias` (query, key, value);
- every other state entry (catalog programs, record vocabulary, naming joins)
  under `buffer/<name>`, in its own dtype;
- metadata, as JSON strings: the model and observation configuration
  (`hypers`), the `world_binding`, the `architecture` receipt, the `source`
  record of the revision that wrote it, `global_step`, the `bc` record with
  the critic's `value_semantic`, and for a belief policy
  `belief_schema_identity` and `belief_content_manifest_identity`.
  `parameter_names` preserves the writer's parameter order for optimizer warm starts.

The container carries every configuration the torch policy runs: belief
inputs, policy history, FiLM, either critic, hashed or catalog record ids and
both policy inputs. Saving never casts or truncates a tensor; a parameter
that is not float32 fails the save, naming it.

`save_checkpoint` binds the policy to the setups in `player_configs`, or to
the world itself when that is `None`, so the policy may play any deck the
world admits. It writes the container whatever the path's suffix; name policy
files `.safetensors`.

`load_checkpoint` reads a container or a `.pt` file written by an earlier
revision, telling them apart by their first bytes, never by name. Both give
the same policy: world binding, architecture receipt, `value_semantic` and
belief binding are validated the same way and fail with the same errors, and
a refusal names the revision that wrote the file. Both honor the
`bucketed_shapes` and `live_tokens` options. A file that is neither (a
truncated download, an older non-ZIP PyTorch serialization) raises
`NotAContainer`.

Code that needs a checkpoint's configuration but not its weights reads the
header, without importing a framework:

```python
from manabot.policy.checkpoint import read_metadata

metadata = read_metadata("policy.safetensors")
metadata["world_binding"], metadata["hypers"]["agent_hypers"]
```

Optimizer state, update snapshots and incident tensors are PyTorch files and
keep the `.pt` suffix; they are not policy checkpoints and no other backend
reads them.

### Convert an older `.pt` checkpoint

`load_checkpoint` reads old `.pt` policies directly, so the torch backend
needs no conversion. Another backend reads only the container:

```sh
uv run python -m manabot.backends.convert .runs/example/checkpoint.pt
```

```text
read   .runs/example/checkpoint.pt  (180 tensors)
wrote  .runs/example/checkpoint.safetensors
```

The converter reads the `.pt` file with PyTorch and writes the container
beside it, with the original file's name under `converted_from`. Every torch
tensor becomes one parameter or buffer, or conversion fails naming the key.
It converts any configuration and refuses a `.safetensors` input.

A backend refuses what it does not implement when it loads, not when the file
is written: the [MLX backend](mlx.md#what-the-backend-covers) raises
`UnsupportedPolicy` naming every unsupported field, rather than running a
policy partially.

## Parity check

```sh
uv run --extra mlx python -m manabot.backends.parity capture/initial.pt --capture capture
```

```text
torch  clipped Adam weights                     max |Δ| 7.5e-09 (tol 1e-6)  ok
torch  forward logits                           max |Δ| 4.8e-09 (tol 1e-4)  ok
torch  forward values                           max |Δ| 1.8e-07 (tol 1e-4)  ok
torch  collection logits                        max |Δ| 0.0e+00 (tol 1e-4)  ok
torch  collection values                        max |Δ| 0.0e+00 (tol 1e-4)  ok
torch  learner loss                             rel Δ 0.0000% (tol 0.1%)  ok
torch  gradient semantic_cards.token_embedding.weight min cosine 1.000000 over 169 tensors (tol 0.999)  ok
mlx    clipped Adam weights                     max |Δ| 1.2e-07 (tol 1e-6)  ok
mlx    forward logits                           max |Δ| 5.9e-09 (tol 1e-4)  ok
mlx    forward values                           max |Δ| 3.9e-07 (tol 1e-4)  ok
mlx    collection logits                        max |Δ| 4.9e-09 (tol 1e-4)  ok
mlx    collection values                        max |Δ| 2.7e-07 (tol 1e-4)  ok
mlx    learner loss                             rel Δ 0.0004% (tol 0.1%)  ok
mlx    gradient extra_attention.5.mlp.layers.0.weight min cosine 1.000000 over 201 tensors (tol 0.999)  ok
parity: pass (torch, mlx)
```

The check runs the PyTorch model in float32 on CPU as the reference and every
installed backend, torch included, on the same inputs, each side computing
the damped update's `minibatch_objective` through its own backend: a
capture directory written by `experiments/mlx_pilot.py capture` (real learner
minibatches with their targets and settings, and real collection calls). Each
backend loads the reference's weights through the canonical names and is
compared in its own parameter layout. The check compares logits on legal
actions and values (absolute difference), the update's loss (relative difference),
each parameter's gradient (cosine similarity), and every weight after one
Adam step with gradient clipping. Tolerances are fixed:
logits and values within 1e-4, loss within 0.1%, every gradient cosine at
least 0.999, post-step weights within 1e-6 absolute. The weight bar includes
every parameter; only the gradient-cosine check treats negligible tensors as
zero. A tensor whose gradient norm is below 1e-6 of the global norm in
both frameworks counts as zero: the policy-score bias and every attention key
bias have analytically zero gradients (softmax ignores a shift shared by all
its inputs), so both sides hold rounding noise. The check exits 1 naming each
failing backend's worst components, and 2 when the checkpoint or capture
cannot be read, the checkpoint is refused or no backend is installed; a
timing taken from a backend that fails it is not reported.
