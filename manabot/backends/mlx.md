# MLX backend (Apple Silicon)

`manabot.backends.mlx` runs every `ActorCritic` configuration the PyTorch policy
builds, both maintained updates (`damped_policy` and `ppo`), live-token
attention, and autoregressive belief sampling and fitting in
[MLX](https://ml-explore.github.io/mlx/) on Apple Silicon. It loads
the portable policy files training writes
([policy checkpoints](README.md#policy-checkpoints)), writes and loads portable
`belief.safetensors` samplers, and uses PyTorch as its numerical reference
([parity check](README.md#parity-check)). The ordinary pilot gates timing on
those checks. The [MPS retest](../../experiments/reports/mps_retest.md) explicitly
measures approximate half-precision arms too; it reports failing tolerances
alongside timing and does not treat execution as numerical equivalence. Local
evaluation and baseline verification can also run through MLX in explicit CPU
float32 without importing PyTorch; PyTorch remains the default for evaluation
and the numerical parity reference.

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
vocabulary, legal actions, packed GRU schedules, shared-effect routes, belief
rows, recent events, decision-memory prefixes and live-token offsets) happens
on the host, so the MLX graph holds only gathers, scatters and arithmetic.
`inference=True` drops the object and fact
columns no row uses, exactly as PyTorch CPU inference does. `evaluate` returns
NumPy logits `[B,A]` (-1e8 off the legal support) and values `[B]`, like the
PyTorch policy. Calling `policy(batch)` returns MLX arrays with logits `[B,A]`
and values `[B]` (signed expected outcome `P(win) - P(loss)` for
`categorical_wdl`, scalar `[B]` otherwise); `policy.forward_distribution(batch)`
returns the raw critic output (`[B,1]` or `[B,3]` logits).

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

MLX supports every `ModelConfig` the Torch `ActorCritic` builds: catalog and
hashed record vocabularies (`world="world"` and `world=None`), scalar and
`categorical_wdl` critics, every `value_aggregation` mode (`masked_mean`,
`historical_mean`, `value_token`, `attention_readout`), table attention on or
off (`attention_on`), stream normalization, focus and policy FiLM,
`recent_events`, belief rows (`BeliefPolicy.admit_belief`), and whole-game
`decision_memory` (`gru` and `transformer`, 1 or 4 summaries, including
`pretrained_memory_branch` warm starts). Inputs v9 through v11 use the same
saved column readings as Torch. Collection and learning support both updates
(`damped_policy` and `ppo`), bucketed shapes, bounded execution chunks, and
`live_tokens=True` on both `mps` and `cpu`. All five shipped recipes (`local`,
`baseline`, `elemental`, `belief_policy` and `belief_sampler`) accept
`backend="mlx"`.

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

Run the bounded local recipe through one update, then continue the same run from
the saved update boundary:

```sh
uv run --extra mlx etude train manabot/recipes/local.py --config '{"backend":"mlx","device":"mps"}' --until updates=1
uv run --extra mlx etude train manabot/recipes/local.py --resume --until updates=2
```

Each attempt writes raw and EMA policy exports under `out/policies/`
(`00000001-raw.safetensors`, `00000001-ema.safetensors`, and after resume
`00000002-raw.safetensors`, `00000002-ema.safetensors`) and commits an atomic
update save under `out/state/current.json` (including after update 0).

A schema-2 MLX save pairs two files alongside Etude's `state.json`:

- `continuation.json` (framework-free host state): global-slot stream state
  (0-action roots for `complete-games`, committed action prefixes for
  `exact-quota`, parked fresh roots for snapshot vectors), `learned_game_rows`,
  the opponent RNG when `behavior="random"`, the minibatch RNG, and Python/NumPy
  RNGs;
- `training.safetensors` (backend state): raw and EMA weights, Adam moments and
  step, the collector's private sampling RNG, the MLX RNG, and a SHA-256 digest
  binding the save to `continuation.json` (`schema="2"`).

Before mutating weights or opening native streams, resume validates model
settings, saved column readings, lookup tables, backend, schema, the
`continuation.json` digest, and array names, dtypes and shapes. It restores
weights, EMA and Adam state, reopens streams directly into their committed roots
or replayed action prefixes (verifying every slot's native state digest, chooser
and decision-memory cursors, including across changed worker counts), and
installs host and backend RNGs last. Prefix replay rebuilds native game statistics
and preserves any recorded policy probabilities and values. Policy cutoff counts
and durable game records resume with live play.

On CPU float32, stopping between updates (via `--until` or `ETUDE_STOP_FILE`)
and resuming reproduces the next update's native decisions, counts, weights, EMA
and Adam arrays identically. On Metal, stream keys, slots, action traces, state
digests and restored arrays match identically, while subsequent GPU arithmetic
follows Metal's float32 repeatability. Existing schema-1 MLX saves (without
`continuation.json`) remain readable and start fresh game and sampling streams
at the saved update. A schema-2 save with a missing or tampered
`continuation.json`, digest mismatch, nonfinite Adam state, corrupt action
prefix or incompatible stream geometry fails admission without advancing or
modifying `state/current.json` or milestones. Portable policy exports can start
a fresh learner on either backend.

A numerical failure ends the attempt. Under `numerical-incident/`, the JSON
record names the invariant (`nonfinite_gradients` before Adam,
`nonfinite_parameters_after_step` after Adam), attempted update and last policy
exports. `tensors.safetensors` retains bounded evidence from up to 16 rollout
rows within 32 MiB. `state.safetensors` holds the failure-time model, Adam, EMA
and RNG state when their tensors fit within 512 MiB.

### Local evaluation and baseline verification

Score an exported policy against an anchor or another policy through MLX on CPU
without importing PyTorch:

```sh
uv run --extra mlx python -m manabot.eval.local \
  --backend mlx \
  --checkpoint ~/.etude/track/local/out/policies/00000002-ema.safetensors \
  --opponent heuristic --deal-seed 101 --out .runs/local-mlx-check --seconds 300
```

```text
completed: 1/1 complete deals; 4/4 games; evidence: .runs/local-mlx-check/panel
score: 0.500 (1 deal block; uncertainty unavailable from a single deal); local-check-not-admission
```

In Python, `manabot.eval.evaluation.evaluate_local(..., backend_name="mlx")`
runs the same scalar CPU game children through MLX, and
`manabot.recipes.baseline.evaluate` uses `config.backend`:

```python
from pathlib import Path

from manabot.recipes.baseline import BaselineConfig, evaluate

report = evaluate(
    BaselineConfig(backend="mlx", device="mps", updates=2),
    Path("~/.etude/track/baseline/out").expanduser(),
    seed=0,
)
print(report.learning.verdict, report.learning.gain)
```

Both resolve MLX with explicit `precision="float32"` on CPU so width ≥ 256
checkpoints do not switch to `bfloat16` during CPU verification. `backend_name`
is execution-only and does not enter `EvaluationRequest` or `EvaluationKey`, so
registration digests and existing evaluation directories stay unchanged.

### PPO and belief sampling

Train one Gray Ogre PPO update, then fit the sampler from the registered final
policy. Use the same fresh experiment label on both commands so Etude delivers
the policy checkpoint as the sampler job's `start` input.

```sh
uv run --extra mlx etude train experiments/belief.py --experiment belief-mlx --job policy --config '{"backend":"mlx","device":"mps"}'
uv run --extra mlx etude train experiments/belief.py --experiment belief-mlx --job sampler --config '{"backend":"mlx"}'
```

The sampler job writes `histories.json` with the policy's SHA-256, fits the
autoregressive sampler on the training games, and registers
`belief.safetensors` at step 4. Its output directory is
`~/.etude/track/belief-mlx/sampler@0/out/`.

Both Torch and MLX write `belief.safetensors` in one portable container holding
float32 parameters under canonical names alongside the format, schema, dataset,
policy, world, hyperparameter and metric metadata. `load_belief_sampler` checks
exact bytes and every binding identity, so a sampler trained on MLX loads in
Torch (and vice versa) with matching `log_prob`. MLX refuses legacy `.pt`
belief samplers (`PK\x03\x04`) naming the Torch backend; Torch keeps a legacy
`.pt` read fallback.

### Architecture mirror and execution

`require_supported` checks only that `policy_input_version` is 9–11; no
`ModelConfig` architecture is refused. The modules of `policy.py` mirror the
PyTorch model one to one, so a parameter path in MLX names the same component as
in the torch state dict:

- object embeddings, perspective vector and optional belief rows;
- the catalog GRU or `card_definitions` embedding, and catalog or hashed fact
  and shared-effect encoders;
- optional `recent_events` and `decision_memory` (`gru` or `transformer`)
  encoders;
- the optional attention stack (with `live_tokens=True` packing valid objects to
  `[L, D]` across projections, norms, memory gating and MLP, and compacting
  Q/K/V to `[B, H, max_live, d]` around fused
  `mx.fast.scaled_dot_product_attention` from host `live_offsets` with zero
  device sync);
- action heads, name candidates and the scalar or `categorical_wdl` critic with
  every `value_aggregation` mode.

Two departures from the PyTorch forward are exact up to float rounding. The
catalog GRU always encodes every definition, where PyTorch learning encodes
only the rows a batch references; the others never reach the loss either way.
And the clipped surrogate's likelihood ratio is float32, where PyTorch uses
float64 (the GPU has none). Ordinary values can differ by rounding; a
float32 ratio can also overflow sooner, which the numerical checks refuse.

## Layout mapping

`checkpoint.py` maps the container's canonical parameters to MLX's layout:

- **Attention.** Canonical packed in-projections (`mha.qkv_weight` /
  `mha.qkv_bias` on board attention and `self_attn.qkv_weight` /
  `self_attn.qkv_bias` on the decision-memory transformer, `[3·d, d]` / `[3·d]`)
  split, in that order, into `query_proj`, `key_proj` and `value_proj` of MLX's
  `MultiHeadAttention`.
- **GRU.** `PackedGRU` holds `Wx`, `Wh`, `input_bias` and `hidden_bias`, with
  gates ordered reset, update, new. All four tensors copy unchanged. Keeping
  both biases preserves their independent Adam moments and their separate
  contributions to the global gradient norm used for clipping.
- **Containers.** `nn.Sequential` children `N` become `layers.N`.

Every canonical parameter maps to MLX arrays, or loading fails naming the key.

## Learner

The backend carries no loss of its own. `manabot.backends.mlx.rl` implements the
full `RL` surface (`UpdateBackend`), including scalar and categorical WDL critic
terms (`categorical_lambda_returns`, `categorical_value_loss`,
`outcome_probabilities`, `expected_outcome`) and PPO standardization
(`standardized_advantages`). `manabot.backends.mlx.learner.Learner`
differentiates whatever loss closure an update hands it, clips the global
gradient norm and applies Adam (eps 1e-5, bias correction). Both maintained
updates run on MLX unchanged, using the shared one-ahead host prefetch in
`AdmittedRows`:

```python
import numpy as np

import manabot.backends.mlx as backend
from manabot.recipes.builders import move_learning
from manabot.training.learning import Learning
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.ppo import update_iteration

learner = backend.learner.Learner(policy, minibatch_rows=64)
damped = update_damped_iteration(
    backend, learner, batch, move_learning(), iteration=1
)
ppo = update_iteration(
    backend, learner, batch, Learning(), progress=0.0, rng=np.random.default_rng(0)
)
```

`batch` is a `RolloutBatch` as either backend's collection writes it. Check
diagnostics against the torch backend on the intended batch, weights and
precision using the parity command below. Host return recursions
(`transition_gae`, `categorical_lambda_returns`) and advantage selection live
once in `manabot.policy.returns` in NumPy, and the surrogate ratio is float32
([what differs](README.md#algorithms-are-written-once)).

`Learner.step(..., recorder=observer)` accepts the shared `StepRecorder`
protocol (`manabot.training.updates.health.StepRecorder`). On sampled steps (the
first step of iteration 1 and every 25th iteration under `OptimizerHealth`),
MLX retains zero-copy pre-step parameter references, computes per-leaf squared
sums on device in float32 alongside the finite check, and reduces them in host
float64 to record pre/post-clip gradient norms, critic readout (`value_head.`,
`value_readout.`) and optional `value_token` norms, missing/zero/nonzero/frozen
tensor counts, and absolute and relative L2 parameter displacement. MLX returns
zero gradients for unused trainable leaves, which count as zero tensors. A
missing tensor means the gradient tree has no entry for a trainable parameter.
The recorder keeps the gradient sample if Adam produces a nonfinite parameter.
Unsampled steps skip the measurement reductions and weight snapshots.
Disabling sampling preserves the collected actions and learned weights.

The step runs eagerly: packed GRU schedules and legal-action counts change
with every batch, so a compiled graph would retrace per call; only the GRU cell
is compiled. Collection samples once per group across execution chunks.
Portable exports contain the policy’s float32 parameters and lookup buffers.

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
