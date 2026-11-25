# Framework backends

```python
from manabot.backends import resolve

backend = resolve("torch")
policy, space = backend.load_policy(".runs/example/self-play-raw.safetensors")
batch = policy.prepare(obs, inference=True)  # NumPy in: the buffers VectorEnv produces
logits, values = policy.evaluate(batch)      # NumPy out: logits [B,A], values [B]
```

`obs` is the dictionary of NumPy observation buffers for `B` decisions.
`prepare` checks it and does every integer lookup on the host, returning a
`Prepared` batch (`manabot.policy.prepare`); `evaluate` runs the forward and
returns logits, -1e8 off the legal actions, and each row's value.
`inference=True` lays the batch out as simulation and managed inference do (on CPU it
drops object and fact columns no row uses; on Torch MPS it keeps used object
columns padded with empty columns to a power of two for attention); the default
is the learning layout, whose outputs differ from it by float summation order. Every
backend reads the same `Prepared` batch and the same checkpoint file, so the
MLX lines differ only in the name passed to `resolve`
([mlx.md](mlx.md)).

A backend runs the manabot policy forward and the learner step of its
updates in one array framework. Backends share the framework-free contract in
[`manabot.policy`](../policy/README.md): observation admission, host
preparation, the checkpoint container and the host inference contract. Tracked training defaults to PyTorch. The local, baseline, Elemental and belief recipes
also accept MLX. PyTorch is the default for evaluation and supplies the
numerical parity reference; local evaluation and baseline verification also run
on MLX in explicit CPU float32.

| Backend | Status | Guide |
| --- | --- | --- |
| `torch` | the reference: training, evaluation, the CLI and Academy | [below](#the-torch-backend) |
| `mlx` | full policy architecture parity, damped and PPO training, live tokens, belief sampling/fitting, portable cross-backend continuation, sampled health, local evaluation, and portable policy and belief containers | [mlx.md](mlx.md) |

## Which backend on Apple Silicon

Keep PyTorch in production and retain MLX as an experimental Apple backend:
MLX half's measured speed advantage still needs learning-quality validation.
See the [MPS retest](../../experiments/reports/mps_retest.md) for results and limits.
Both follow [one precision rule](../../docs/training.md#reduced-precision):
bfloat16 forwards for models of width 256 or more on MPS and MLX, as on CUDA,
with float32 log-probabilities, master weights and optimizer state.

## Layout

Every backend is one package, `manabot/backends/<name>/`, with these files
under the same names:

| File | Holds |
| --- | --- |
| `__init__.py` | the import guard naming the optional extra, and the files below as attributes loaded on first use (`backend.rl`) |
| `arrays.py` | conversion to and from NumPy, the float dtype, the device |
| `runtime.py` | device setup, execution admission and fresh policy construction |
| `policy.py` | the policy: `prepare`, and the forward over a `Prepared` batch (logits and values) |
| `live_tokens.py` | object attention over the valid objects of a packed batch |
| `decision_memory.py` | the decision-memory encoder over prepared records |
| `collection.py` | frozen collection forwards and the collector's private RNG |
| `rl.py` | the policy terms, losses, returns and selection the updates call, over the framework's arrays |
| `learner.py` | `Learner`: the optimizer, the differentiated loss, the guarded step and precision |
| `rollout.py` | `AdmittedRollout`: a collected batch's observations as the policy reads them |
| `state.py` | the backend tensors of saved training state and numerical incidents |
| `checkpoint.py` | the map from canonical names to the backend's layout, portable loading and export, and `BACKEND` |
| `belief.py`, `belief_fit.py` | lazy belief sampling (`SamplerCodec` / `BeliefSampling`) and fitting (`BeliefTrainer`) exposed as `backend.belief` and `backend.belief_fit` |

A backend adds modules only for what it alone needs. Torch's policy encoders
are one module each (`semantic_cards.py`, `facts.py`, `shared_effects.py`,
`records.py`, `recent_events.py`, `name_candidates.py`, `continuation.py`),
with `numerics.py` for device arithmetic boundaries; it also holds the
reference tools `bot.py`, `capture.py`, `decision_memory_probe.py` and
`verify_decision_memory.py`. MLX's forward runs these encoders over packed
rows inside `policy.py`, whose parameter containers mirror the torch modules'
names; `gru.py` holds the packed GRU with its hand-written gradient.

`resolve` returns the configured framework. Simulation loads a policy through
`PolicyLoader`, checkpoint inspection reads its metadata and tensors through
`CheckpointReader`, and updates take the `rl`, `arrays`, `learner` and `rollout` modules. These modules
load on access, so playing a policy does not import training. A backend
imports its own framework,
`manabot.policy` and the configuration and observation modules; it never
imports another backend (`tests/test_package_boundaries.py` enforces it).

## Who chooses a backend

Library code receives a backend; it does not import one. An update takes the
configured backend object (or an imported update module) as its first
argument, and simulation and evaluation take a
policy that satisfies `BackendPolicy`. Checkpoint discovery and registration
receive a reader for metadata and a loader when they need a live policy; they never resolve it themselves.
Only the code that starts a program
chooses:

- `experiments/` and `scripts/`;
- recipe entry points, the CLI and child-process entry points;
- Academy's policy provider (`manabot/sim/policy_provider.py`);
- the backends' own tests.

They name a backend once, by import or through the resolver:

```python
import manabot.backends

backend = manabot.backends.resolve("torch")
learner = backend.learner.Learner(policy, "cpu", minibatch_rows=64)
```

A loader calls `backend.load_policy(path)`, which returns
the policy and its observation space. The
policy supplies host `prepare`/`evaluate`, action sampling and guarded
`inference_scope()` calls. Scalar sampling stays in the backend to preserve its seeded action stream.
Managed evaluation instead uses `eval.agent_loader.load`, the composition root
that chooses Torch and the device. Etude batches its deterministic masked-logit
forward; the Torch-free game-side agent uses NumPy PCG64 sampling. These have
fresh registrations and tournament names. Standalone scalar game children
resolve their backend lazily; code players and replay import no framework.

`resolve(name, *, precision=None)` imports the framework on first use.
MLX carries the selected compute precision on the returned backend; omitted
precision follows each loaded policy's width. Torch accepts omitted precision
or `"float32"`; its learner configures reduced-precision training. A backend
whose optional extra
is not installed raises `ImportError` naming the extra; an unknown name raises
`ValueError` listing the backends.

`tests/test_package_boundaries.py` refuses an import of `manabot.backends.*`
anywhere else. In a module that contains both reusable functions and a CLI,
choose the backend inside `main()` or the child entry point and pass it to the
reusable functions. Having a CLI in the same file does not make its library
functions composition roots.

### Fresh collection and portable export

`fresh_iteration(settings, backend=..., seed=..., out=...)` runs one collection,
learner update and EMA step in a new directory. It returns the live policies
and counts and writes separate raw and EMA policies when averaging is enabled.
Both frameworks use `policy.checkpoint.save_policy_checkpoint`. Either reader
can play or inspect an export whose architecture it supports. A learner initialized
from one starts with fresh Adam state and counts.
See the [fresh baseline example](mlx.md#what-the-backend-covers).

The collector receives a numerical session that owns policy inference,
precision and sampling. Native game streams, seat routing and complete-game
accounting stay in the collector. Rollout records contain NumPy arrays;
packing validates them through the injected numerical capability. Torch
collection uses its private generator seeded at `seed + 1`, shared by learner
actions and current-self opponent actions. Tracked saves record it in
`sampling.safetensors` under the `torch:<device>` domain so same-domain resumes
continue the same stream; cross-domain resumes reseed deterministically from
`(run.seed, counts.updates)`.

### Tracked recipes

The local, baseline, Elemental, `belief_policy` and `belief_sampler` recipes
select the framework through their `backend` setting, which defaults to
`"torch"`. On Apple Silicon, `backend="mlx"` runs both damped and PPO self-play
as well as belief history collection and sampler fitting. The self-play recipes
use the same collection, learning, EMA and portable export as the fresh
iteration. Etude also records metrics, cumulative clocks and training saves.
The loop requires the backend's `Continuation` capability before accessing run
state.

### Continuing a run

Every tracked self-play run writes the same four files into
`out/state/save-NNNNNNNN/` through Etude's `run.resume(export=, restore=, extra=)`
API, whichever backend runs it:

| File | Content |
| --- | --- |
| `state.json` | Etude clocks, seed, commit, file inventory and `Counts` (updates, transitions, optimizer steps, saved model settings, input readings and snapshot-bank state) |
| `continuation.json` | Framework-free `HostContinuation` (schema 2): global-slot stream roots and committed action prefixes, `learned_game_rows`, opponent RNG, minibatch RNG and Python/NumPy RNGs |
| `learner.safetensors` | Canonical float32 `raw/<canonical>`, optional `ema/<canonical>`, `adam/step`, `adam/exp_avg/<canonical>` and `adam/exp_avg_sq/<canonical>` for stepped parameters, plus the `schema="canonical-v1"` manifest (`precision`, `has_ema`, per-parameter `param_steps`, observation config, lookup-buffer digests, `counts_sha256`, `continuation_sha256`, `sampling_sha256`, `sampling_domain`) |
| `sampling.safetensors` | Backend sampling RNG arrays keyed by `sampling_domain` (`mlx`, `torch:cpu`, `torch:cuda`, `torch:mps`) |

Any backend that supports the saved model and precision can continue a save.
For a launched run, supply the new recipe config when continuing from the
parent's saved output. Etude queues the child in the parent's worker pool, so
that pool must provide the selected backend and device.

```sh
uv run etude continue --run RUN --until updates=4 --config '{"backend":"mlx","device":"mps"}'
```

Within the same `sampling_domain`, resume restores the learner, EMA, Adam,
counts, native games and sampling streams. In bounded CPU float32 self-play,
the next update reproduces the host and sampling files byte for byte, every
learner tensor and every work counter. Each attempt measures its own elapsed
training time, which contributes to the learner header's `counts_sha256`.
Across sampling domains (Torch ↔ MLX or Torch across CUDA/MPS/CPU), resume
still verifies `sampling_sha256`, reseeds sampling deterministically from
`(run.seed, counts.updates)`, and logs `run/sampling_reseeded=1` on the first
resumed update.

Because PyTorch `Adam` skips `state[p]["step"]` when a conditional module
receives `grad is None`, `learner.safetensors` records per-parameter
`param_steps` (`0` means unstepped and omits zero moment tensors). Both backends
restore each parameter's moments and update count. MLX uses the difference
between that count and the global step for bias correction on subsequent dense
updates, including a previously unused parameter's first update. On Torch → MLX,
`run/adam_steps_aligned` counts the parameters whose clocks trailed the global
step when training resumed.

Published policy milestones (`out/policies/NNNNNNNN-{raw,ema}.safetensors`)
stay separate from `out/state/save-NNNNNNNN/`: milestones carry `world_binding`,
`checkpoint_source` and forward-compatible catalog rebinding, while training
saves are strict exact-match continuation snapshots that validate every header,
buffer digest, parameter, Adam tensor, host state and sampling digest before
installing weights. The collector then verifies the reconstructed games before
training or publishing a milestone.

Pre-portable saves are refused on both backends before model/reading diffing or
tensor loading, leaving save bytes and live weights untouched: legacy Torch
saves (`learner.pt`) point to manabot 0.2.5 or warm-starting a new run from
the run's published policy, and pre-unification MLX `training.safetensors`
saves are refused explicitly. See the
[tracked MLX training guide](mlx.md#tracked-training-and-recovery) and
[training recovery](../../docs/training.md#recovery).

Numerical incident evidence still uses the producing backend's format (`.pt` on
Torch, `.safetensors` on MLX), and `manabot/training/loops/minimal.py` remains a
standalone Torch tutorial loop.

### Inspection and belief

Frozen-choice inspection accepts a `PolicyLoader`. It keeps the observations
and their offer and object permutations in NumPy and evaluates them through
`policy.prepare(..., inference=True)`. The MLX reader can inspect a portable
policy with Torch absent. A collected corpus stores the arrays in
`observations.npz`, keyed by `<case name>/<buffer name>`. Numerical parity
also takes a fresh learner and the framework's gradient and parameter mappings.

Belief inference uses host observations and the policy's admission and
evaluation methods. Search receives a hand sampler. Both Torch and MLX expose
`backend.belief` (`SamplerCodec` / `BeliefSampling`, owning sampling arithmetic,
the random stream and learned state) and lazy `backend.belief_fit`
(`BeliefTrainer`), and read and write one portable `belief.safetensors`
container (`manabot.belief.checkpoint`); non-safetensors payloads (including
legacy `.pt` archives) are refused on both backends with
`ValueError("unknown sampler checkpoint format")`.
`manabot.reports.belief` reads sampler metadata without raw `torch.load`, while
the host pools calibration, Brier scores and support violations per decision.

Health reporting keeps step and rejection counters on the host behind one
`StepRecorder` protocol (`manabot.training.updates.health.StepRecorder`). Both
Torch and MLX learners supply sampled gradient and parameter measurements on the
first optimizer step of iteration 1 and every 25th iteration. An optimizer
failure after clipping retains the gradient sample. The update, rollout and
search-sampling modules can all be imported with Torch absent.

## Algorithms are written once

An update or a loop is written once and takes the backend as an argument:
the configured backend object returned by `resolve`, or a directly imported
backend module. The maintained updates in `manabot/training/updates/` take
it first:

```python
import manabot.backends.torch as backend

diagnostics = update_damped_iteration(
    backend, learner, batch, move_learning(), iteration=1
)
```

Both maintained updates (`update_damped_iteration` and PPO `update_iteration`)
run on Torch and on MLX. One minibatch at the same weights gives the same
diagnostics on both, within the [parity check](#parity-check)'s bars
(`tests/training/test_update_backends.py` runs them on every installed
backend):

```python
import numpy as np

import manabot.backends.mlx as backend
from manabot.training.learning import Learning
from manabot.training.updates.ppo import update_iteration

learner = backend.learner.Learner(policy, minibatch_rows=64)
damped = update_damped_iteration(
    backend, learner, batch, move_learning(), iteration=1
)
ppo = update_iteration(
    backend, learner, batch, Learning(), progress=0.0, rng=np.random.default_rng(0)
)
```

Inside an update, array math goes only through the backend's modules. The
Protocols in `manabot/training/updates/backend.py` state one uniform surface
(`UpdateBackend` and `RL`, covering both updates and both scalar and
categorical critics), and `tests/training/test_update_backend_protocols.py`
checks every installed backend against it:

- `backend.arrays`: moving host rows in and out (`from_numpy`, `to_numpy`)
  and `float32` for a forward output.
- `backend.rl`: the policy terms (`legal_mask`, `policy_logs`,
  `action_logs`, `reference_logs`), the losses
  (`clipped_surrogate`, `reverse_kl`, `forward_kl`, `squared_value_loss`,
  `categorical_value_loss`, `entropy`, `clip_fraction`), returns
  (`transition_gae`, `standardized_advantages`, `categorical_lambda_returns`,
  `outcome_probabilities`, `expected_outcome`), selection (`selected_magnitudes`) and
  reductions (`selected_mean`, `weighted_sum`), `constant` for a value with no gradient
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
  `minibatches(order, size, device)` is a context manager on `AdmittedRollout`
  in both backends, backed by shared one-ahead host prefetch in
  `AdmittedRows.prefetch_minibatches` (`manabot/training/rollout/admission.py`)
  with strict release-before-submit ordering; exiting joins the worker even on
  failure. Device transfers and all model and optimizer work run on the caller.

The shared `damped_policy_loss(rl, ...)` and `minibatch_objective` in
`manabot.training.updates.objective` compose the policy and critic terms.

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
`transition_gae` and `categorical_lambda_returns` wrap; torch keeps advantage
selection on the minibatch's device (`selected_magnitudes`, `torch.quantile`),
matching `returns.quantile` bit for bit. The
algorithm modules (`damped_policy.py` and `ppo.py`) import no framework and no backend, even for annotations
(`tests/test_package_boundaries.py` enforces it); they name the backend's
arrays, devices and observation batches as the opaque `Array`, `Device` and
`Observations`.

A function has the same name, arguments, shapes and per-row results in every
backend that has it; the torch backend's signatures are the reference, and
[training loops](../../docs/training-loop.md#standard-methods-kept-local)
documents them. A backend never carries its own copy of an update's loss:
`manabot.backends.parity` runs `minibatch_objective` (for both scalar and
`categorical_wdl` policies) through each backend's `rl` and `Learner`. What
still differs between the two backends:

- MLX has no float64 on the GPU, so its surrogate ratio and loss sum are
  float32; torch's are float64.
- In self-play, `live_tokens=True` is admitted per backend in
  `backend.runtime.admit_execution`: Torch requires CUDA, while MLX accepts
  both `mps` and `cpu`. The packed MLX layers share the
  host offsets for all attention heads and layers in a forward.

## The torch backend

`capture(out, settings, run)` records one damped-policy update from `SelfPlay`
settings in an open Etude run. It writes `settings.json`, the initial policy,
collection calls and learner minibatches. Frozen experiment capture commands
require their original revision; retained captures remain readable.

| Module | Names |
| --- | --- |
| `manabot.backends.torch.policy` | `ActorCritic`, `PolicyInference` |
| `manabot.backends.torch.capture` | bounded real-call and learner-input capture for backend comparisons |
| `manabot.backends.torch.rl` | `legal_mask`, `policy_logs`, `action_logs`, `reference_logs`, `transition_gae`, `standardized_advantages`, `outcome_probabilities`, `expected_outcome`, `categorical_lambda_returns`, `clipped_surrogate`, `reverse_kl`, `forward_kl`, `squared_value_loss`, `categorical_value_loss`, `entropy`, `clip_fraction`, `selected_magnitudes`, `selected_rows`, `selection_mask`, `selected_mean`, `weighted_sum`, `constant`, `require`, `require_finite`, `float64_operand`, `NumericalError` |
| `manabot.backends.torch.learner` | `Learner`, `gradient_step`, `update_ema`, `check_objective`, `forward_precision`, `no_grad`, `apply_matmul_precision` |
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

`prepare(obs, inference=True)` prepares inference rows with the complete
catalog program schedule and selects the device's inference layout
([inference column compaction](../../docs/observation-and-model.md#inference-column-compaction)):

- **CPU:** drops unused object columns (`attention="used_columns"`) and, unless
  `bucketed_shapes=True`, unused fact columns (`facts="used_columns"`).
- **MPS:** pads used object columns with empty columns to a power of two
  (`attention="bucketed_columns"`), bounding distinct Metal attention shapes.
- **CUDA:** keeps padded object and fact columns.

When `attention_on` is false, object column compaction is skipped; when
`live_tokens=True` and `attention_on` is true, packed live-token attention takes
precedence over column compaction. Simulation, search and rollout bootstrapping
prepare with `inference=True`. Column compaction changes floating-point
summation order, so learner rows use the default `inference=False`: they retain
padded object columns (unless `live_tokens=True`) and select only referenced
catalog programs. Bucketed CUDA learners use the full schedule to keep the
catalog encoder's shape fixed across batches.
The argument chooses preparation purpose independently of autograd mode;
`no_grad` does not change a default learner batch into an inference batch.
Torch rollout `rows` and `minibatches` always prepare learner rows;
`next_observation` always prepares inference rows for bootstrapping.

Torch MPS uses `facts="bucketed_live_rows"` for inference and learning. It
packs live facts into one `[1,R]` route and adds unlinked rows so `R` is a
power of two of at least 64. Captured 64-board calls had at most 1.3% of
their fact rows live. CPU and CUDA use `[B,F]` fact routes.

Every packed GRU (catalog programs, fact programs, record reducers) goes
through `packed_gru` (`torch/packed_gru.py`). On MPS it projects all packed
rows once and runs a hand-written backward that carries only the state
gradient through the reverse step loop, as MLX's `gru_recurrence` does.
On MPS, `nn.GRU` over a `PackedSequence` dispatches about twenty ops per step.
It reads the module's `nn.GRU` weights and runs in float32 under autocast, as
`nn.GRU` does on MPS. CPU and CUDA call `nn.GRU`.

`Learner` and `forward_precision` accept `"float32"`, `"tf32"` and `"bf16"`,
refusing unknown precision names and `"bf16"` on CPU with `ValueError` rather
than silently falling back to float32. For parity, Torch’s `BACKEND.load`
accepts optional `device`, `precision` and `bucketed_shapes` (defaulting to
CPU, float32 and padded; `"bfloat16"` translates to `"bf16"` for `Learner`)
and returns an `ActorCritic` and a `Learner`, whose parameters and gradients
keep torch's names (`torch_name` in `manabot.policy.checkpoint` maps canonical
names back).

Read card-program rows through `SemanticCardEncoder.lookup`
(`manabot/backends/torch/semantic_cards.py`), never `table[rows]`. Card and fact IDs
repeat heavily in a batch; under CUDA autograd `lookup` uses embedding
backward, which groups repeated rows, where advanced indexing accumulates each
row separately and made PPO updates 2.17× slower on an L4
(`experiments/reports/ppo_cuda.md`). CPU results are identical either way.

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

Tracked training continuation uses the shared four-file safetensors/JSON layout
across both backends ([continuing a run](#continuing-a-run)); Torch uses `.pt`
for numerical incidents, while MLX writes numerical incidents as safetensors.
Portable policy exports (`.safetensors`) and belief samplers
(`belief.safetensors`) load on either backend for evaluation and fresh
learners; legacy `.pt` policies load on Torch (or convert via
`manabot.backends.convert`), while legacy `.pt` training saves and belief
samplers are refused on both backends.

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

Both backends build every `ModelConfig` architecture. The shared checkpoint
reader rejects unsupported input versions through `ObservationConfig.from_checkpoint`.

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

The reference is padded Torch CPU float32. Every candidate loads the same
weights and replays the same learner minibatches and collection calls from
`manabot.backends.torch.capture`. Both learners compute the production
damped objective. After comparing one clipped Adam step, parity reloads the
checkpoint on each side for the remaining comparisons.

Float32 requires legal logits and critic values within 1e-4, relative loss
within 0.1%, per-tensor gradient cosine of at least 0.999 and every post-step
weight within 1e-6. Gradients below 1e-6 of the reference's global norm on
both sides count as zero. A constant shift of all softmax inputs leaves its
output unchanged, so policy-score and attention-key biases can have only
rounding noise in their gradients. Bfloat16 uses a full-gradient cosine and
calibrated forward and loss limits. [Compute precision guide](mlx.md#compute-precision)

To check a bucketed Torch MPS candidate in bfloat16, run

```sh
uv run --extra mlx python -m manabot.backends.parity capture/initial.pt --capture capture \
  --backend torch --device mps --precision bfloat16 --bucketed --threads 4 --json parity.json
```

On the M1 w64l2 baseline capture, this command checked eight minibatches and
eight collection calls. The bucketed Torch MPS forward ran in bfloat16 and
the clipped Adam step was finite. The largest relative learner-loss difference
was 3.63% against the 1% tolerance, so the command exited 1 and wrote
`"failures": ["learner loss"]` to the report.

| Flag | Default | Effect |
| --- | --- | --- |
| `--backend` | all installed | Select `torch` or `mlx`. |
| `--device` | `cpu` (Torch), `mps` (MLX) | Torch accepts `cpu`, `mps`, or `cuda`. MLX accepts `cpu`, `mps`, or its alias `gpu`. |
| `--precision` | `float32` | Use `float32` or `bfloat16` for the forward. The loss takes float32 logits and values. Torch bfloat16 requires MPS or CUDA. |
| `--bucketed` | off (padded) | Trim candidate inputs to bucketed shapes. |
| `--threads` | current Torch thread count | Set Torch threads during the check. MLX manages its own thread pool. |

The JSON records the checkpoint digest, sample indices, actual device,
layout and observed critic dtype before float32 promotion. A target mismatch,
empty learner or collection sample, nonfinite computation or tolerance miss
fails the check (exit 1). Invalid inputs and unavailable devices exit 2.
The M1 benchmark also reports timings for finite bfloat16 candidates that
miss a numerical tolerance and identifies each failed check.
[M1 benchmark results](../../experiments/reports/mini_baseline_backends.md)
