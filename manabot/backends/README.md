# Framework backends

A backend runs the manabot policy forward and its PPO learner step in one
array framework. Backends share the framework-free contract in
[`manabot.policy`](../policy/README.md): host preparation of observation
batches, the checkpoint container and the `BackendPolicy`,
`BackendLearner` and `Backend` protocols. Training, evaluation and the CLI
still run the PyTorch model in `manabot.model`, which is also the reference
every backend is checked against.

| Backend | Status | Guide |
| --- | --- | --- |
| `torch` | the reference: training, evaluation and the CLI run it | [training loops](../../docs/training-loop.md) |
| `mlx` | policy and learner step, parity-checked; measurement only | [mlx.md](mlx.md) |

## Layout

Every backend is one package, `manabot/backends/<name>/`, with the same files:

| File | Holds |
| --- | --- |
| `__init__.py` | the import guard naming the optional extra; nothing else |
| `arrays.py` | conversion to and from NumPy and the float dtype |
| `policy.py` | the forward over a `Prepared` batch: logits and values |
| `rl.py` | the array math of the shared updates: policy terms, losses, returns and selection |
| `learner.py` | the optimizer, gradient step, EMA and precision context |
| `checkpoint.py` | the map from canonical names to the backend's layout, loading, and `BACKEND` |

`BACKEND` in `checkpoint.py` implements `manabot.policy.protocol.Backend`, the
one object `parity` needs. A backend imports its own framework,
`manabot.policy` and the configuration and observation modules; it never
imports another backend. Outside `manabot/backends/`, only code that chooses
a backend imports one: the executor and the loops in
`manabot.training.loops` import `manabot.backends.torch`, and an update
imports none (`tests/test_package_boundaries.py` enforces these).

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

Running the update in MLX changes the import to `manabot.backends.mlx`, with
a learner from that backend. Inside an update, array math goes only through
the backend's modules:

- `backend.arrays`: moving host rows in and out (NumPy to the framework's
  arrays and back) and the float dtype.
- `backend.rl`: the policy terms (`legal_mask`, `policy_logs`,
  `damped_policy_loss`), the losses (`clipped_surrogate`, `reverse_kl`,
  `squared_value_loss`, `categorical_value_loss`, `entropy`), returns
  (`transition_gae`, `standardized_advantages`, `categorical_lambda_returns`)
  and selection (`selected_rows`, `selection_mask`, `selected_mean`).
- `backend.learner`: `gradient_step`, `update_ema` and the
  `forward_precision` context.

NumPy keeps the host-side bookkeeping: admitted rollouts, action rows, health
records and diagnostics. Policy logic that is pure NumPy or index logic, such
as `reference_distribution`, lives once in `manabot.policy`.

A function has the same name, arguments, shapes and per-row results in every
backend; the torch backend's signatures are the reference, and
[training loops](../../docs/training-loop.md#standard-methods-kept-local)
documents them. Where a framework cannot match a dtype (MLX's GPU has no
float64, so its surrogate ratio is float32), the backend's guide states the
difference and its bound. A backend never carries its own copy of an update's
loss: the parity check runs the shared update on each backend.

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
`manabot.model.checkpoint.load_checkpoint` and fails with the same errors; a
refusal names the revision that wrote the original torch checkpoint. A file
that is not a container (a `.pt`, or a conversion without the metadata) raises
`NotAContainer`, saying to convert the `.pt`; `convert` refuses a `.safetensors`
input. Backends
cover what the w160l8 regime uses: a catalog world with catalog record ids,
table attention, a masked-mean scalar critic and policy input v9. A checkpoint
that enables anything else (belief inputs, policy history, the pooled token
stream, another critic, FiLM, input v10) is refused with every unsupported
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
shared update over the torch backend, and the same update over every installed
backend on the same inputs: a capture directory written by `experiments/mlx_pilot.py capture` (real learner
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
