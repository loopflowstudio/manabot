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
| `mlx` | policy and learner step, parity-checked; measurement only | [mlx.md](mlx.md) |

## Which backend on Apple Silicon

Recommendation: keep PyTorch in production and retain MLX as an experimental
Apple backend. The [MPS retest](../../experiments/mps_retest.md) found that
float16 MPS narrowed the whole-update gap to MLX to about 10% at w384l8/256
on the busy M4 Max, while every MPS precision lost to CPU at w64l2/256 on both
hosts. Half-precision gradient differences remain unresolved, the wider/batch
matrix is incomplete, and the M2 w384 comparison stopped for sustained paging.
The report owns the measurements and limits; neither it nor the earlier
[MLX pilot](../../experiments/mlx_pilot.md) establishes equivalent learning
trajectories. Defaults and off-CUDA precision refusals remain unchanged.

## Layout

Every backend is one package, `manabot/backends/<name>/`, with the same files:

| File | Holds |
| --- | --- |
| `__init__.py` | the import guard naming the optional extra; nothing else |
| `arrays.py` | conversion to and from NumPy and the float dtype |
| `policy.py` | the forward over a `Prepared` batch: logits and values |
| `rl.py` | the loss terms of `manabot.training.updates` and `manabot.rl` over the framework's arrays |
| `learner.py` | the optimizer, gradient step and precision |
| `checkpoint.py` | the map from canonical names to the backend's layout, loading, and `BACKEND` |

`BACKEND` in `checkpoint.py` implements `manabot.policy.protocol.Backend`, the
one object `parity` needs. A backend imports its own framework,
`manabot.policy` and the configuration and observation modules; it never
imports another backend, and nothing outside `manabot/backends/` imports a
backend (`tests/test_package_boundaries.py` enforces both).

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
