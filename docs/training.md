# Training manabot

[How a manabot learns](learning.md) explains the learning loop itself. This is
the operations guide to training a manabot: how a run is declared and executed,
how remote jobs are built, what the monitoring and numerical-health records
mean, and how interrupted or failed work is recovered. Comparisons are declared
in [experiments.md](experiments.md); metric definitions live in
[experiment-metrics.md](experiment-metrics.md); evaluation protocols live in
[evaluation.md](evaluation.md).

- [Run training](#run-training)
- [Regime functions](#regime-functions)
- [Regimes and stages](#regimes-and-stages)
- [Experiments](#experiments)
- [Remote training](#remote-training)
- [Monitoring](#monitoring)
- [Numerical health](#numerical-health)
- [Recovery](#recovery)
- [Game failures](#game-failures)
- [Ataraxos](#ataraxos)
- [Policy inputs](#policy-inputs)
- [Distributed training](#distributed-training)
- [History](#history)

## Run training

```bash
uv run manabot train                                   # --recipe local: bounded smoke
uv run manabot train --recipe baseline --seed 197      # baseline-v3 with evaluation
uv run manabot train --function experiments.two_models:two_models \
    --arguments arguments.json --seed 197 --out .runs/two-models/a1
uv run manabot train --function experiments.two_models:two_models \
    --arguments arguments.json --smoke --out .runs/two-models/smoke   # one tiny real update
uv run manabot train --regime-name direct_self_play --seed 197 --out .runs/direct-pilot
uv run manabot train --regime regime.json --seed 197 --out .runs/example
```

`--recipe local` (`manabot.recipes.local`) is a two-update production-path
smoke with no evaluation. `--recipe baseline` (`manabot.recipes.baseline`,
version `baseline-v3`) runs the ordinary baseline through the experiment runner
with initial and endpoint evaluation. `--steps`, `--seconds` and
`--learning-seats` and `--device cpu|mps|cuda` apply only to these recipes.
Without `--device`, both select CUDA first, then Apple MPS. Baseline requires
an accelerator and rejects CPU; use `--recipe local` on CPU-only machines.
Local defaults to CPU when no accelerator is available and accepts an explicit
CPU override.
`--preset` and `--set` are
retired and fail.

Apple Silicon MPS is admitted for float32 ordinary self-play through
`execute_regime` and both CLI recipes, and requires Torch 2.15 or newer: on an
Apple M2, Torch 2.10–2.13 compute wrong MPS linear-layer weight gradients,
which gradient clipping then applies (see `manabot/training/mps.py`). The
repository lock installs 2.15. With an older Torch, automatic selection skips
MPS, so local trains on CPU; baseline, an explicit `--device mps` and any MPS
stage stop before training with a message naming both versions. An unavailable
device fails explicitly. Float64 loss and diagnostic arithmetic runs on the
host with gradients preserved; no automatic operator fallback is enabled.
Reduced precision remains CUDA-only. MPS process recovery, belief
inputs and frozen opponents are unsupported. Evaluation and search stay on CPU
for this workflow. The baseline coordinator reserves a CPU evaluator thread
alongside accelerator training. Monitoring exports do not require process
recovery; accelerator runs cannot resume a stopped learner. This support makes
no speed or strength claim.

Each experiment is one module, `experiments/<experiment>.py`, next to its
write-up. New training is a [regime function](#regime-functions) in that
module, selected with `--function <module>:<function>` and given its keyword
arguments as a JSON object in `--arguments`. Older modules declare a saved
recipe instead: the zero-argument function named like the module returning a
`TrainingRegime`, with other such functions as variants.
`--regime-name <experiment>` or `<experiment>:<variant>` selects one, and
`manabot.cli.regime_names()` lists them. Reusable composition helpers are in
`manabot.recipes.builders` (`move_learning`, `move_checkpoints`,
`ataraxos_baseline`, `with_capacity`, `with_value_output`,
`with_recent_events`, ...). Write a regime file from Python:

```bash
uv run python -c "from experiments.batched_baseline import batched_baseline; print(batched_baseline().model_dump_json(indent=2))" > regime.json
```

Function and regime runs require `--out`. The CLI
opens the canonical `training.sqlite` in the **parent** of `--out`; choose a
new output directory for every attempt. `--function`, `--regime` and
`--regime-name` are exclusive. `--resume-from`, `--source-repair-reason`
([recovery](#recovery)), `--checkpoint-updates` and the legacy
`--checkpoint-seconds` ([monitoring](#monitoring)) apply only to saved regime
runs; a function run takes `--resume-from` and `--checkpoint-updates`.
`--smoke` runs either kind in its [tiny form](#smoke-runs) and takes no resume
or cadence controls.

## Regime functions

Manabot's maintained training loops are regime functions: ordinary Python
functions whose first argument is the run. We keep a few of them, each our
best current recipe for its question, built from composable stage functions;
the run records every one the same way. A loop of your own is the same thing:
take the stage functions (or the learning pieces under them) and call them
from a function of yours, or launch it through Etude as we do.

A regime function creates models, calls stage functions in order and passes
results from one stage to the next as ordinary values:

```python
# experiments/two_models.py (abridged)
from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.recipes.builders import move_learning
from manabot.training.decks import UniformDecks
from manabot.training.regime import DampedPolicyLearning, Learning
from manabot.training.run import Run
from manabot.training.stages import self_play


def two_models(
    run: Run,
    *,
    decks: MatchConfig | UniformDecks,
    updates: int,
    learning: Learning | DampedPolicyLearning = move_learning(),
    reference: ModelConfig = ModelConfig(hidden_dim=32),
    learner: ModelConfig = ModelConfig(hidden_dim=64),
) -> None:
    """Train `learner` against an untrained `reference` held frozen."""
    observation = ObservationConfig()
    fixed = run.model("reference", reference, observation)
    trained = run.model("learner", learner, observation)
    self_play(run, trained, opponent=fixed, decks=decks, updates=updates,
              learning=learning)
    run.export(trained)
    run.export(fixed)
```

A regime with its arguments bound is a `functools.partial`. `execute` and
`smoke` (`manabot.training.run`) take any such callable:

```python
from functools import partial
from pathlib import Path

from experiments.two_models import two_models
from manabot.infra.config import MatchConfig
from manabot.training.run import execute, smoke

regime = partial(two_models, decks=MatchConfig(), updates=200)
smoke(regime, out=Path(".runs/two-models/smoke"))  # one tiny real update per stage
result = execute(regime, out=Path(".runs/two-models/a1"), seed=197)
result.status                                       # "completed"
result.checkpoints["learner"].raw                   # self-play-raw.pt
result.checkpoints["reference"].raw                 # export-reference-raw.pt, never trained
```

There is no trainer class, decorator or stage schema: anything Python can
express between stages (a loop over curriculum phases, a branch on a recorded
measurement, a third model) is part of the regime.

### The run handle

`Run` (`manabot.training.run`) owns everything about an attempt besides the
training math: its `TrainingRun` record and `run.json` export, the writer
lease, named seed streams, wall/stop/pause supervision, monitoring exports and
failure settlement. A regime function receives it and never builds one.

| `Run` provides | Meaning |
| --- | --- |
| `run.id`, `run.seed`, `run.world` | The run's identity, its root seed and the installed world it is bound to. |
| `run.model(name, agent, observation, seed=None)` | A new randomly initialized `Model`, seeded from `run.seed` and `name` unless `seed` is given, without moving the run's own RNG streams. Names are lowercase and unique. |
| `run.export(model, weights="raw", decks=None)` | Publish the model's current weights through ordinary checkpoint admission as an `export-<model>` stage and return its `ArtifactReference`. Weights a stage already published are returned as they are. |
| `run.stage(name)` | A context that records one stage of your own: its cost, status and artifacts. |
| `run.metrics(values, update=...)` | Record named values in the current stage's diagnostics. |

A `Model` is a name bound to a `ModelConfig`, an `ObservationConfig` and its
current weights. Stage functions read and change models in place; a model is
never copied implicitly. Every model shares the run's world; models may differ
in architecture and observation capacity. An export binds the setups of
`decks` when given a `MatchConfig`; otherwise it binds the world, which needs a
model reading the complete world input.

### Stage functions

Stage functions live in `manabot.training.stages`. Each records one stage in
`VerifyStore` while it runs, with its operation and its resolved arguments, so
a stage's record, cost and artifacts look the same whether a regime function
or a [saved regime](#saved-regimes) produced them.

| Function | Does | Returns |
| --- | --- | --- |
| `self_play(run, model, *, decks, updates, learning, opponent=None, ...)` | Self-play training of `model`, or training against `opponent` held frozen. | `SelfPlayResult`: the stage's raw/EMA checkpoints and counters. |
| `import_policy(run, name, *, source_run, source_stage, checkpoint, decks, weights)` | Admit a published policy (as the saved `import_policy` stage does) as a new model. | The `Model`. |
| `collect_belief(run, model, *, decks, games, weights)` | Collect hidden-hand labels with `model` frozen. | The `SamplerDataset`. |
| `train_belief(run, corpus, ...)` | Fit a belief sampler to a corpus. | The sampler's `ArtifactReference`. |

`self_play` takes the settings a saved `TrainSelfPlay` stage declares, as
keyword arguments with the same names and meanings: `streams`, `transitions`,
`games_per_stream`, `learning_seats`, `batch_closure`, `behavior`,
`matchup_curriculum`, `adaptive`, `trainable`, `execution` and `learning`.
`decks` is a `MatchConfig` or a `UniformDecks` source. Its schedules run on the
`iteration_fraction` clock. Training the same model in the next self-play
stage continues its live learner, Adam state, EMA and collector, as `initial`
does in a saved regime, and must keep the settings that continuation fixes. A
stage's name defaults to the function's, numbered in call order (`self-play`,
`self-play-2`); `stage=` chooses one. Regime functions do not yet tolerate
game failures, and a regime function's run cannot yet be the producer of an
`import_policy` or learning-state continuation.

### Identity

The run records a regime function as `regime_function` in `run.json` (in place
of `regime`): the callable's `module:qualname`, every keyword argument after
the run (bound by the partial or a default) as JSON with its
`arguments_sha256`, the source revision (commit, plus the digest of
uncommitted tracked changes) and the run's wall budget and recovery policy.
Arguments must serialize to JSON: numbers, strings, the typed records
(`Learning`, `Execution`, `ModelConfig`, `UniformDecks`, ...) and collections
of them. Each is validated against the function's annotation before the run
opens, so a wrong type, a missing or unknown argument, a lambda or an open
file fails at once. `load_regime(function, arguments)` binds a recorded or
remote regime again. Two runs with equal `function`, `arguments_sha256` and
revision ran the same recipe; to compare runs, compare their `regime_function`
blocks and their stages' `arguments`.

### Smoke runs

`smoke(regime, out=...)` (or `--smoke`) runs the regime in its tiny form for
real, into `out`: models are built, games are played, and every stage
completes **one** update of one transition per stream (at most one game per
stream) on CPU, then exports and reloads its checkpoints through admission.
`out` then holds real, loadable checkpoints that an evaluation's smoke games
can use. A bad deck, an observation capacity the decks overflow, a world
mismatch or an error in the function's own code fails here in seconds. A smoke
run says nothing about speed or learning; pass `device=` to smoke on an
accelerator.

### Pause, stop and resume

`execute(..., pause_path=...)` (with a `recovery=RecoveryPolicy(...)`), the
run's wall budget or `stop_path` end the current stage at its next update
boundary. A pause commits a recovery snapshot and raises `TrainingPaused`; the
attempt ends `interrupted`. `execute(regime, out=..., seed=...,
resume_from=<run id>)` resumes it and refuses a different function, argument,
revision, supervision or seed. Resuming re-enters the function from the top:
stages already completed return their recorded results and their models load
the recorded weights, and the interrupted stage continues from its last
recovery snapshot at an update boundary ([recovery](#recovery)). Code between
stages runs again on resume, so keep it deterministic and let it act only
through `run` and stage functions.

### Saved regimes

`run_saved_regime(run, regime)` (`manabot.training.stages`) runs a saved
`TrainingRegime` through the same stage bodies in its stage order;
`execute_regime` and `--regime` call it. A saved regime keeps its serialized
identity: its run record embeds the same regime bytes, and receipts that hash
those bytes still match. `partial(run_saved_regime, regime=regime)` is a
regime like any other, which is how `--smoke` and `TrainingJob.smoke(out)` run
a saved regime's tiny form.

## Regimes and stages

A `TrainingRegime` (`manabot.training.regime`) is a saved, serialized recipe;
a `TrainingRun` (`manabot.training.records`) is one bounded execution, of a
regime function or a saved regime, with resolved settings, named seed streams,
artifacts, measured costs and status. `execute_regime`
(`manabot.training.execution`) runs a saved regime through
[`run_saved_regime`](#saved-regimes). New recipes are
[regime functions](#regime-functions); the stage settings below apply to both. `VerifyStore` is the
canonical SQLite owner of runs and stages; `run.json` is an export of committed
state. Files are atomically published before their SHA-256 identities enter
stage records, and every checkpoint uses the ordinary writer and world/setup-
bound reload path.

The regime binds `ModelConfig`, `ObservationConfig`, `MatchConfig` (including
sideboards) and the world once. Unsupported operations, forward references and
a world that differs from the installed native runtime fail validation.
New regimes use schema 2; saved schema-1 recipes keep their one-seat
interpretation and serialized identity.

### Stage operations

| Operation | Purpose |
| --- | --- |
| `train_self_play` | Ordinary self-play learner (PPO-style `Learning` or `DampedPolicyLearning`). |
| `import_policy` | Admit a frozen TrainingRun export's raw or EMA checkpoint without training. |
| `collect_selection` | Frozen-policy complete-game selection diagnostics; no weights change. |
| `collect_belief` / `train_belief` | Collect hidden-hand labels from a policy stage and fit a sampler; see [beside the loop](../README.md#beside-the-loop). |

Every stage declares device, float32 precision, worker count, Torch thread
count, wall and memory limits (`Execution`). `execution.workers` accepts 1
(default), 2 or 4 CPU observation producers for self-play only: persistent
threads own disjoint native stream ranges, a barrier completes all writes
before the single policy owner reads, and stream identities and seeds are
unchanged across partitions. Sampled rosters (a deck source or
`elemental-uniform`) draw each game's decks from that game's native seed, so
draws, exposure and batches agree for 1, 2 and 4 producers.
`execution.threads` controls Torch threads independently. The field default
stays 1, so saved regimes keep their meaning; new sampled-roster recipes built by
`manabot.recipes.builders.sampled_self_play` default to four producers, a
damping horizon fitted to the run's expected updates and `recover_capacity`.
The matched GPU comparison for a fixed matchup is in
[worker-scaling run019](../experiments/worker-scaling.md#persistent-production-producers-run019).
Direct `SeatRoutedCollector` users must call `close()`.

Self-play `initial` continues the live collector, learner and optimizer of an
earlier stage in the same execution. External checkpoints never act as
implicit resume. The policy reads only the acting viewer's tensors; private
labels are never model inputs. Learner and behavior weights are raw; EMA is a
separately named evaluation output with its own clock. The selection rule is
`last-complete-raw`, never the best seed or best evaluation.

### Schedules and clocks

`schedule_clock="run_elapsed_budget"` (the serialized default) divides elapsed
run wall time by the regime's `wall_seconds`; continuation stages do not reset
it. `iteration_fraction` uses completed iterations and is required for step
cadence, recovery and portable continuation; `RunControl()` selects it for new
declarations. Damped-policy (Ataraxos) schedules use absolute iteration
formulas regardless of this field. Evaluation is outside training and does not
consume the training clock.

### Learning from both seats

`train_self_play.learning_seats` defaults to `"both"` for current-self and
EMA-self training; `"one"` is the ablation control. Random and frozen
opponents default to one seat and are never trained on. The default is a
choice, not a measured strength or cost gain.

Both-seat collection records each acting seat's behavior outputs; each seat
gets its own trajectory and terminal win/loss/draw reward (draws zero), and
GAE and lambda returns stop at each seat's game boundary. `transitions` is a
**minimum** decision count per native stream counting both seats; the stream
then finishes its current game, so batches have variable row counts. Complete
seat trajectories are packed into `[rows, 1]` with terminal separators and no
padding. `games_per_stream` overrides the floor with complete games per stream
per update ([batch boundaries](learning.md#can-a-batch-contain-unfinished-games)); set it on both arms for an equal-game comparison and report
learner transitions, environment decisions, games, exposures and cost
separately. For two live models, use one `SeatRoutedCollector` with
`opponent_mode="learning"`, `learning_seats="both"` and `opponent_agent`;
`collect_pair` returns each model's rows.

### Batch closure

`train_self_play.batch_closure` says when a batch ends, independently of who
learns. `"complete-games"`, the default, treats `transitions` as a per-stream
floor and finishes the game in progress, so every return ends at a real result;
finished streams idle until the slowest game ends, recorded as
`idle_stream_steps`. `"exact-quota"` pauses each stream at exactly `transitions`
learner rows and bootstraps the mid-game tail from `V(next_obs)`. Exact quota
needs one learning seat, because at a pause only the acting seat has a next
observation, and rejects `games_per_stream`. It accepts `game_errors` only with
`recover_capacity`: a failed game discards the rows no update has learned from, its
stream refills to the quota, and the rows earlier updates learned from stay, counted
as `learned_samples`.

A saved stage never declared its closure, so `TrainingRegime` resolves omission
the way those runs behaved: one seat with neither `games_per_stream` nor
`game_errors` is exact quota, anything else complete games. Stages serialize the
field only when it differs from that reading, so saved digests are unchanged and
a new one-seat complete-game stage writes it out. `AttackPlan` schema 1 retains
its exact-quota attackers; schema 2 uses the template's closure.

### Costs and counters

Collection, optimization and export time are separate from evaluation.
Completed stages freeze `cumulative_seconds` from run start at artifact
admission; use it for checkpoint cutoffs. Records distinguish complete games,
environment decisions, `learner_transitions`, `optimizer_updates` (completed
`optimizer.step()` calls) and `optimizer_exposures` (sample uses across
minibatches); `updates` is absolute completed iterations and `segment_updates`
those in this stage. An empty-filter iteration is a completed step with zero
optimizer calls. A failed or interrupted stage retains its error, cost and
published artifacts. Reproducibility means resolved settings and artifact
provenance, not byte-identical stochastic training. A fresh execution owns
fresh costs; no pre-existing corpus is free.

### Compound decisions (removed)

The learned compound-decision policy (`compound_decisions` models and
`train_compound` stages) has been removed. Saved configurations, checkpoints
and regimes that select it fail validation rather than loading as flat
policies; reproduce them from source commit
`ba3cd58de25bcc5d0b32e0fa3c546ee3d27be52b`. Retained receipts and reports
that record compound fields remain readable.

### Selection diagnostics

`collect_selection` freezes a prior self-play stage's raw or EMA checkpoint
(optionally from another run in the same store via `source_run`) and collects
a declared population with unique seeds and whole-game development/held-out
splits. Scalar and categorical values are supported; belief-input sources
are not. The protocol is in
[ataraxos-omitted-controls](../experiments/ataraxos-omitted-controls.md#frozen-complete-game-selection-diagnostic);
results are frozen-policy associations, not causal filter benefits.

### Reuse a published policy

`ImportPolicy` admits a frozen export and one checkpoint into the stage graph:

```python
from manabot.training.regime import CollectBelief, ImportPolicy

policy = ImportPolicy(
    id="policy", operation="import_policy",
    source_run={"path": "/artifacts/producer/run.json",
                "sha256": run_sha256, "bytes": run_bytes},
    source_stage="producer",
    checkpoint={"path": "/artifacts/producer/producer-ema.pt",
                "sha256": checkpoint_sha256, "bytes": checkpoint_bytes},
    weights="ema",
)
histories = CollectBelief(
    id="histories", operation="collect_belief", policy="policy", weights="ema", games=256,
)
```

Paths may relocate; hash and size must identify the original bytes. The
regime's model, observation contract and world must match the producer, the
producer stage must be completed, and checkpoint metadata must agree with the
export. Admission copies the bytes into the destination run. An EMA import
never masquerades as raw. Producers that themselves import policies, and
legacy checkpoints without TrainingRun metadata, are rejected.

`TrainSelfPlay.pretrained = PretrainedPolicy(...)` (same reference shape)
starts self-play from a published policy instead of random weights. The world,
`ModelConfig` and observation contract must equal the producer's; the learning
rule, schedules, streams, transitions, updates, behavior, opponent, device and
seed are free. Only the policy is inherited: Adam, EMA, streams and the
schedule iteration start fresh, so choose a schedule meant for a trained
policy. Remote jobs reject `pretrained`. To continue one recipe with its Adam
and EMA, use [portable learner continuation](#portable-learner-continuation).

`StageRecord.producer_cost` carries the producer's cumulative cost through
checkpoint admission. Imported-stage `seconds` measure only copying, hashing
and loading. Charge a shared producer once per comparison, not once per
consumer.

## Experiments

Declaring, executing and calibrating a comparison is in
[experiments.md](experiments.md#declaring-a-comparison).

## Remote training

Manabot trains on rented machines through
[Etude](https://github.com/loopflowstudio/etude), which rents the machine,
enforces hours and dollars, ships metrics and checkpoints during the run and
deletes the machine at its deadline.

```python
# experiments/<name>.py
from decimal import Decimal
from pathlib import Path

from experiments.batched_baseline import batched_baseline
from manabot.experiment.job import TrainingJob


def runs(revision: str):
    job = TrainingJob(
        regime=batched_baseline(), experiment=Path(__file__).stem, milestones=(32, 64, 96),
    ).job(repo=Path("."), revision=revision, max_hours=6, max_usd=Decimal(5))
    return job.runs(1, 2)
```

```sh
uv run etude launch experiments/<name>.py --revision COMMIT --cohort <name>-1 --rent l4 --max-usd 10
uv run etude report experiments/<name>.py --source r2://manabot/jobs --open
```

`TrainingJob.smoke(out)` runs the job's regime in its [tiny form](#smoke-runs)
into `out`, leaving a loadable checkpoint; `etude smoke` and `etude launch`
run each training job's tiny form before spending. Etude jobs run one saved
self-play regime; a regime function is not yet a `TrainingJob`, so run it
with `execute` or `manabot train --function` on the machine you rent.

Building a job is pure; nothing is submitted until `etude launch`. Runs track
under `experiment`, the declaring file's stem, so that file's `eval(state)` and
`report()` find them ([experiment files](../experiments/README.md#etude-experiment-files)).
An `Experiment` with `jobs` (`PlannedRun`) gives the same via
`experiment.runs(experiment=..., repo=..., revision=...)`. On the machine,
`manabot.experiment.job` trains to the absolute update target and records a
checkpoint at every milestone so Etude evaluation can follow. Setup installs
Rust 1.98.1 with rustup and runs the locked `uv sync`, which builds the native
engine. `validate_numerics=True` runs the CUDA numerics gate
(`manabot.experiment.numerics`) first and refuses to train if it fails.
The committed dependency requirements and `uv.lock` select the same Etude SDK
release for both the launcher and this setup.
Refresh it only through the [explicit dependency update workflow](development.md#environment-and-commands)
before freezing a new run; never upgrade dependencies inside a training attempt.

`TrainingJob.continuation` (`LearnerContinuation`) continues another run's
learner from its published `run.json` and raw/EMA/Adam exports, bound as a
`LearningStateImport` (see [portable continuation](#portable-learner-continuation));
the seed must be the parent's. A parent interrupted before its final exports
continues only by declaration (`from_monitoring`): from the raw weights of one
of its monitoring exports, with the EMA reset to them and a fresh Adam. A job
stops at an update boundary before its watchdog (`stop_reserve_seconds`),
exports its learner state and ends `paused`; Etude records that attempt as
succeeded at the update it reached. `etude continue` resumes from a policy
checkpoint only, which carries no Adam or collector state, so the worker
refuses it. Pick the machine at launch (`--pool`, `--rent`); read progress with
`etude status`, `etude logs` and `etude usage`. See Etude's
[launcher guide](https://github.com/loopflowstudio/etude/blob/main/docs/launcher.md).

## Monitoring

TrainingRun keeps the scientific receipt: regime, identities, counters,
artifacts, monitoring checkpoints and status. Per-update diagnostics go once
to the stage's Etude run; checkpoint evaluation goes to a separate Etude
monitoring run. No account is needed to train.

### Update metrics

A local run writes its Etude run under `OUT/<stage>/etude/<run id>/`, named in
`TrainingRun.metrics_run`; under `etude launch` it is the launcher's run in
`ETUDE_TRACK_DIR`. A recovery attempt continues its parent's Etude run, and
later rows replace earlier ones from the resumed update. Draw it with:

```bash
uv run etude report experiments/elemental.py --source .runs/example --open
```

Rows use flat names: `loss`, `numerical/...`, `coordinates/learner_transitions`,
phase seconds under `time/` (`time/collection`,
`time/learning/minibatches/forward_and_loss`), CUDA high-water marks
(`memory/peak_allocated_cuda_bytes`, `memory/peak_reserved_cuda_bytes`),
`selection_groups/<action type>/<terminal distance>/rows`. Completed self-play
outcomes are rows with `deck`, `opponent_deck`, `position` (`play` or `draw`),
`wins`, `losses` and `draws` in an immutable JSON table under the metric run's
`artifacts/self-play-outcomes/`. The update's `self_play_outcomes/table` metric
names its SHA-256 digest. Etude carries these files alongside metrics, including
for launched jobs; keep the artifacts when copying a run for offline analysis.
Deck pairs no longer create metric columns, which made background index refreshes
repeatedly load large matchup histories.

`manabot.training.update_metrics.read_updates` verifies the selected tables and
projects their counts as `self_play_outcomes/<deck>/<opponent>/<play|draw>/wins`
(and `losses`/`draws`) for existing reports. It selects resumed and duplicate
updates before reading their tables; unreferenced tables supply no results.
Missing or corrupt referenced tables raise an error. Historical scalar outcome
metrics remain readable unchanged. Selection and belief
summaries stay in their stage's `diagnostics`.

The Perf panel's "Update time" series shows where an update goes:
`environment_step_and_encoding`, which compares a run to retained ones, divides by
the engine's own clock into `native_stepping` and `observation_encoding`, beside
preparation, inference, `stream_bookkeeping`, assembly and learning.
`TrainingRun.host` names the machine the attempt got. With one producer
(`execution.workers` 1) training occupies one CPU thread, so host single-thread
speed sets its pace, not vCPU count or GPU size; see
[training update time](../experiments/training-update-time.md) before choosing a
machine. `scripts/bench_training_update.py` prints the same split for one machine.

Reading the curves: RL policy loss is the clipped objective, **not log loss**.
PPO value loss is half the squared lambda-target residual; categorical move
learning records its own. Loss, entropy and KL describe the last optimized
minibatch. RSS, CPU time and host load are resource observations, not
learning quality. Throughput is cumulative work over observed elapsed cost.

### Checkpoint cadence

`--checkpoint-updates N` (recorded as `TrainingRun.monitoring_checkpoint_updates`)
exports a raw checkpoint at every absolute multiple of N completed self-play
iterations; the final raw export owns the endpoint. Exports preserve Torch
RNG, Adam and the collector. A failed export leaves an explicit missing
milestone. `ExperimentSchedule` and `PlannedRun` default to 1,000 updates, an
authoring default rather than an hourly estimate. Portable continuation keeps
the absolute grid (resuming at 26,000 with interval 1,000 first exports at
27,000). Step cadence requires one step-target self-play stage; multi-stage
recipes keep explicit legacy cadence. `--checkpoint-seconds`
preserves historical elapsed-time cadence for frozen recipes.

### Checkpoint evaluation

Run evaluation in a separate process and account for its compute:

```bash
uv run python -m manabot.experiment.monitor_worker --run .runs/example/run.json \
  --out .runs/example-monitor --watch-seconds 30 --concurrent-activity 'training running concurrently'
uv run python -m manabot.experiment.monitor_worker --run .runs/example/run.json --checkpoint 0 --out .runs/checkpoint-0-monitor
uv run python -m manabot.experiment.monitor_worker --run .runs/example/run.json --stage policy-0 --out .runs/stage-0-monitor
```

The follower evaluates each newly admitted export once, sequentially, and
never replaces stopped attempts. The default `MonitorProtocol`
(`manabot.eval.monitor`) is 100 games: 25 reserved deals
(`1910101000`–`1910101024`), four seat/deck legs each, against source-pinned
scripted greedy. Exclude these deals from scientific cohorts. `--protocol`
accepts a frozen custom protocol. Arena owns game execution, timeouts, legal
Commands and exact replay. Any failed or missing leg makes aggregate rates
unavailable; complete cohorts report 95% bootstrap intervals over whole
four-leg deals, which are conditional on the checkpoint and say nothing about
training-seed variation. `monitor.json` records evaluation wall, CPU, RSS and
declared concurrent activity.

Saved results can be published without new games:

```bash
uv run python -m manabot.reports.training --evaluations a/monitor.json b/monitor.json --out .runs/monitor-dashboard
uv run python -m manabot.experiment.monitor_worker --import-manifest manifest.json --import-rows rows.json --out .runs/imported-monitor
```

For `manabot.reports.training`, `--destination` (directory or
`s3://bucket/prefix`) and `--experiment` select delivery; settled rows are never rewritten and a changed prefix is rejected.

### Stored bytes

Retained bytes live in `s3://etudefantasia/manabot/` and W&B metrics under
`loopflow-studio/etude` (bucket in `us-west-2`, public access blocked, AES256,
versioned). Use the standard AWS credential
chain; never put keys in recipes. Training exports are retained as Etude
checkpoints; experiment results are published with
`uv run --extra artifacts python -m manabot.data publish <path>`. Older
TrainingRun manifests remain readable as `StoredArtifact`
(`manabot.infra.artifacts`), which `load_checkpoint` accepts directly, caching
under `MANABOT_ARTIFACT_CACHE` (default `.runs/artifact-cache`) and rehashing
on every hit. Storage never relaxes world, architecture or setup admission;
load only trusted producers' Torch files.

## Numerical health

Ordinary `train_self_play` retains masked normalized policy logs through
collection and learning, identified as `policy_numerics=masked-log-softmax-v1`
in the run's identities. Per-update health is in the Etude metrics
(`numerical/...`); TrainingRun owns counters and terminal incidents, and the
experiment report has a Numerical health section.

### Sampling contract

For each legal offer, `log_softmax` gives normalized log probabilities after
masking padding to negative infinity; legal logs must be finite. Collection
samples with `torch.multinomial` on their exponentials and stores the weights,
full logs and chosen-action log likelihood. Learners validate masks, nonempty
support, zero illegal mass, normalization, agreement between weights and logs,
legal sampled actions and the saved chosen log. Float32 `exp(-104)` can round
to zero: a finite legal log with zero weight is accepted only when its own
exponential rounds to zero. There is no epsilon, floor, mixture,
renormalization repair or skipped corrupt batch; nonfinite values fail with a
named invariant. A rounded-zero action cannot be sampled; underflow counts
expose that limit. Reverse KL uses saved logs with padding removed, and ratios
exponentiate in float64. The PPO-style learner uses the same contract (no
`1e-12` KL clamp).

### Metrics

| Metric | Population and meaning |
| --- | --- |
| `legal_logit_gap_max`, `legal_probability_min` | All collected learner rows before filtering; minimum includes rounded zeros. |
| `legal_underflow_count` | Legal entries with zero sampling weight and finite logs. A count, not proof of harmless exploration loss. |
| `nonfinite_count` | Zero in admitted batches; failures name their invariant. |
| `actor_rows`, `critic_rows` | Unique selected rows before epoch reuse. Actor-only filtering can leave zero actor rows. |
| `forced_rows`, `zero_advantage_rows`, `empty_actor_batch` | Forced decisions have zero policy gradient but critic/shared updates can still move parameters. |
| `gradient_norm_before`, `gradient_norm_after` | Global L2 norm around clipping on the sampled step. |
| `gradient_missing_tensors`, `gradient_zero_tensors`, `gradient_nonzero_tensors`, `frozen_parameter_tensors` | Missing, all-zero and nonzero `.grad`, and frozen tensors. All-missing gradients or a detached objective reject the step. |
| `parameter_delta_l2`, `parameter_relative_delta` | Actual parameter movement on the sampled step. Adam momentum can move parameters with zero gradient. |
| `optimizer_steps`, `skipped_steps` | Completed optimizer calls and skipped minibatches per update. |
| `rejected_steps`, `rejected_updates` | Terminal failures, separate from completed diagnostics. |
| `clip_fraction` | Ratios outside the clip interval in the last minibatch. |
| entropy, collection/reference KL, loss | Last optimized minibatch; padding excluded. |

Batch summaries run every update. Gradient counts and parameter movement
sample the **first optimized minibatch at iteration 1 and every 25th
iteration**; finiteness checks run on every optimizer step. Missing samples
are absent, not interpolated. Forced choices and filtering can explain zeros;
missing gradients or zero parameter movement need investigation. None of
these measure playing strength.

Overhead probe (optimizer only, 188,418 parameters, 100 steps):

```bash
uv run python scripts/benchmark_numerical_health.py --device cpu
```

Measured 1.74× (about 0.27 ms per step) on CPU on 2026-10-07 and 1.33× on an
NVIDIA L4 on 2026-10-08; neither is full training throughput.

### Terminal incidents

`StageRecord.numerical_failure` records the invariant, attempted iteration and
update health. Each terminal numerical error writes a private
`<stage>-numerical-incident/` directory, registered in
`StageRecord.rejected_artifacts`: `incident.json` (identities, shapes,
extrema, counts), `tensors.pt` (up to 16 offending rows and 32 MiB, plus the
triggering minibatch for optimizer failures) and `state.pt` (learner, Adam,
EMA, gradients and RNGs, up to 512 MiB). These may contain invalid state; they
never become admitted policies or recovery snapshots and cannot replay the
whole iteration. The last valid checkpoint is unchanged. Collection and
learning timers settle in `finally`.

### Validation

```bash
uv run --extra dev pytest tests/training/test_numerical_health.py -q
MANABOT_NUMERICS_DEVICE=cuda uv run --extra dev pytest tests/training/test_numerical_health.py -q
uv run python scripts/benchmark_numerical_health.py --device cuda
```

An explicit CUDA request fails without CUDA; it never counts a CPU fallback.
New source or numerical changes need their own CUDA admission on the target
host (or `validate_numerics=True` on a remote job). The 2026-10-08 L4 receipt
is `experiments/data/etu126/cuda-admission-20261008.json` in the artifact store.

## Recovery

There are three recovery contracts and one failure-tolerance policy. Select
one recovery contract per regime.

| Contract | Use | Keeps | Restarts |
| --- | --- | --- | --- |
| Portable continuation (`learning_state`) | New segment on any CPU/CUDA host | Weights, Adam, EMA, absolute updates and inherited counts | Game streams and RNGs |
| Exact CPU recovery (`recovery_max_microsteps`) | Same host, CPU, legacy journal | Everything, including native collector and RNGs | Uncommitted work |
| Current-game recovery (`recovery`) | Sustained CPU runs, active-runtime schedules | Everything at the last snapshot; unfinished games rebuilt from their prefix | Work after the last commit |

### Portable learner continuation

`TrainSelfPlay.learning_state = LearningStateImport(source_run=...,
source_stage=..., raw=..., ema=..., optimizer=...)`, each an
`ArtifactReference` (`path`, `sha256`, `bytes`). Admission copies the bytes
into the child run, validates world/setup/architecture and producer metadata,
and rejects missing or corrupt Adam or inconsistent EMA. Completed or
allocation-paused producers with complete exports are supported; an
incomplete upload never silently becomes a weights-only restart.

An interrupted producer has no final exports, so it continues only by
declaration: `LearningStateImport(..., raw=..., monitoring_update=N)` with no
`ema` or `optimizer`. `raw` must be the bytes of the producer's monitoring
export at absolute update `N` (its record's receipt), the origin takes that
receipt's coordinates, the EMA starts as a copy of the raw weights and Adam
starts with no moments. The origin's `reset` lists `ema` and `optimizer`, so
the discontinuity stays visible in every successor record.

`execute_regime(..., stop_reserve_seconds=R)` avoids leaving such a producer:
a single step-target self-play stage stops at the first update boundary where
the slowest update so far plus `R` no longer fits in its watchdog, exports
raw/EMA/Adam and ends `paused`. Etude training jobs always run this way.

`updates` is an **absolute endpoint**. Use `schedule_clock="iteration_fraction"`,
one self-play stage, no `initial`, no recovery policy, and the parent's seed,
learning, model, observation, match, stream count and transition batch;
placement and allowances may differ. A parent at 26,000 can produce a segment
ending at 63,000 and then 100,000. Ataraxos rates keep their absolute formulas
(update 26,001 uses LR `6.957944302072213e-6`, tau `0.0023684978179155515`
under the default schedule). Fresh streams are seeded from the run seed and
imported boundary. `StageRecord.learning_state_origin` holds inherited
coordinates and cost; `TrainingRun.updates_through()` is absolute. Charge each
segment's allocation separately.

### Exact CPU process recovery

Opt in with `schedule_clock="iteration_fraction"` and an explicit
`recovery_max_microsteps` (up to 20 million active native microsteps over the
collector's lifetime). All stages must be `train_self_play` on CPU.

```bash
uv run manabot train --regime recipe.json --seed 197 --out .runs/first
uv run manabot train --regime recipe.json --seed 197 --out .runs/continued --resume-from RUN_ID
```

Both directories must share a parent so they use the same `training.sqlite`.
Only the latest committed boundary resumes, with one child per attempt;
continue the child after a further failure. An interrupted attempt with all
stages complete finalizes without re-exporting. Snapshots hold parameters,
Adam, EMA, all RNGs, native collector state (reconstructed from seed and a
bounded action journal, then every observation buffer checked) and progress.
They are written at stage entry, every `recovery_every_updates` updates
(default 1) and after export. The journal grows with collector lifetime; size
disk from measured growth. Recovery requires the same recipe, seed, source,
runtime and host; a POSIX lease rejects simultaneous writers (never delete
lease files). Snapshots contain hidden state and pickles: never load untrusted
files or serve them as model exports.

### Current-game recovery

`TrainingRegime.recovery = RecoveryPolicy(max_game_microsteps=40000,
checkpoint_updates=128)` requires current-self collection and iteration-based
schedules. Native reset seeds and each unfinished game's legal prefix rebuild
collection; every observation buffer must match before learning resumes.
Snapshots are private, source-bound, compressed and all retained. The
allocation clock is awake elapsed time (monotonic, excluding suspend); after an
abrupt death the gap is charged as uncertain awake time, and after a reboot the
calendar gap is charged conservatively. This can exhaust the allowance; it
never grants a fresh one.

A `pause.request` in the experiment output forces a recovery commit and returns
`TrainingPaused`; the supervisor drains its running evaluation and records
`paused=true`. `run_experiment(..., resume=True, recover=(N,))` continues
attempt N, subtracting consumed allowance. A failed seed is never silently
retried. [current-baseline](../experiments/current-baseline.md) has a runner
with CLI controls.

### Cost semantics

`seconds` measures this attempt, `prior_seconds` charges ancestors once, and
`recovery_seconds` is a subset of this attempt. Sum attempt costs, never
inherited stage receipts. Uncommitted work may repeat and its cost stays
charged. Abrupt death leaves a running record; reacquiring its lease on the
same host settles it as interrupted and charges the unobserved interval,
including downtime. Orphan files never become authoritative. Foreign-host
recovery, network-filesystem leases and byte-identical replay are unsupported.

## Game failures

### Tolerating game failures in training

A single-stage `TrainSelfPlay` may opt into `game_errors`
(`GameErrorPolicy`, `manabot.training.state.game_errors`):

```json
"game_errors": {"threshold": 0.1, "window": 100, "warmup": 100, "max_game_actions": 40000}
```

An engine rejection or native panic quarantines only that stream's current
game. Set `recover_capacity: true` in `game_errors` to also recover typed native
observation-capacity failures during active steps, recorded as `resource_limit`
separately from engine errors. This option defaults to false and is omitted
from serialization when false, preserving frozen configurations.

Recovery discards only the affected game's uncommitted samples from both seats;
earlier completed games, committed updates and other streams survive. Other
active streams advance once. Replacement uses the native next seed and declared
matchup distribution, without blacklisting decks or inventing a terminal reward
or outcome. One-seat collection also finishes games before updating when opted
in, which changes its batch freshness and must be declared.

The rolling window counts completed games, engine failures, resource limits,
truncations and action limits once per settled native game. Run interruptions
are recorded as `deadline` or `abandoned` attempts outside that denominator.
After warmup, `rate > threshold` stops the run; the stop survives restart and
source repair. The example above tolerates 10 failures in the latest 100 settled
games and stops at 11, after a 100-game warmup. The prospective
[TLA v10 pilot](../experiments/protocols/tla-trophy-213-v10.md) selects capacity
recovery with threshold `0.25`, window `100` and warmup `100`: 25/100 continues,
26/100 stops. These are pilot settings, not changed global defaults.

Setup/reset, malformed buffers, invalid provenance, numerical and infrastructure
failures remain fatal. An unencodable replacement root stops the run; process
aborts, OOM and hangs belong to the external supervisor. Evaluation retains
failed/incomplete games in their original scheduled slots, never replacing them
with successful games. Training replacement biases learning toward games that
fit the declared limits.

`game-attempts.jsonl` is a private, append-only, fsynced ledger: setup, seed,
stream, seat, full action prefix including the rejected action, native
pre-command frame, error and identities. Resource-limit receipts also retain
the observation configuration so replay crosses the same encoding boundary.
Never publish the ledger as a replay. Replay a receipt by its one-based line:

```sh
uv run python -m manabot.training.state.game_errors .runs/example/game-attempts.jsonl --attempt 1 --seconds 30
```

For process recovery, combine `game_errors` with the current-game `recovery`
policy; legacy journals are rejected. After a compatible fix, resume with the
same recipe, seed, cadence and allocation and a stated reason:

```sh
uv run manabot train --regime recipe.json --seed 197 --out .runs/repaired \
  --resume-from ORIGINAL_RUN_ID \
  --source-repair-reason "Payment implementation repaired; world and input contract unchanged"
```

The reason permits changed Python/native source only; world, content,
action/observation ABI and architecture must match, and the original
allocation deadline carries forward.

### Automatic capture and replay

Capture is independent of `game_errors`. Strict native calls retain private
evidence (`manabot.env.failures`) before propagating the original exception,
under the TrainingRun's `failures/` directory or `.runs/game-failures/`
elsewhere. Files are create-only, mode 0600 and fsynced; a storage failure is
attached to the original exception. Coverage includes strict self-play,
`sim.rollout` and vector environments, search players (flat-MC and rollout
pool roots), selection journals, and arena evaluation, which journals
each pending Command under `failures/` before execution (pass
`keep_game_logs=True` to keep journals of successful games).

```sh
uv run python -m manabot.training.state.game_errors .runs/example/failures/ID.jsonl --attempt 1 --seconds 30
uv run python -m manabot.sim.replay .runs/arena/traces/CELL.commands.jsonl.gz --game 1 --seconds 30
uv run python -m manabot.sim.replay .runs/arena/failures/CELL.SEED.LEG.jsonl --game 1 --seconds 30
```

Replay (`manabot.env.replay`) runs in a bounded subprocess. New records require
matching source and native bytes unless `--allow-changed-runtime` is given; a
world mismatch is always refused. It reports:

| Field | Meaning |
| --- | --- |
| `matches_binding` | The retained pre-command frame (and for arena, prefix, offer, Command and state) agrees; `null` if unavailable or not reached. |
| `at_retained_failure` | Replay reached execution at the retained boundary. |
| `matches_error` | Observed exception class and message agree, comparing `AgentError` / `NativePanic` with PyO3's `PyAgentError` / `PanicException` and ignoring `; native_failure=` suffixes (`matches_retained_error`). |
| `original_outcome` | The retained outcome; interrupted journals stay `execution_outcome_unknown`. |

Only agreement on binding, boundary and error reproduces a known failure.
Replay never rewrites evidence, restores allowance or runs an optimizer.
Engine failures in evaluation get no loss, draw or strength credit.

## Ataraxos

The objective and the fidelity table are in
[how a manabot learns](learning.md#fidelity-to-the-sources).

## Policy inputs

The continuation input, capacities and the optional recent-event input are in
[observations and the policy network](observation-and-model.md#policy-inputs).

## Distributed training

No distributed learner exists. ETU-108 research (2026-10-05/06) recommends one
optimizer and checkpoint owner with explicit actor, inference and frozen
evaluation boundaries, actor-local batched inference as the first candidate,
and zero-lag collection as the compatibility baseline. Policy lag is a
learning-rule decision: PPO clipping is not V-trace, and damped-policy learning
additionally needs full behavior distributions, central filtering and one
schedule step per learner iteration. Matched single-run CPU probes measured
45.32 s (laptop) versus 65.00 s (mini) for equal work, without controlled host
load. Proposed lease, envelope, admission and idempotent-retry contracts are in
the [original note](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/distributed-rl.md).

## History

Earlier documents, with progress narrative, evidence tables and superseded
designs, at their last revision:
[training-regimes](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/training-regimes.md),
[training-experiments](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/training-experiments.md),
[training-monitoring](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/training-monitoring.md),
[training-recovery](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/training-recovery.md),
[training-calibration](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/training-calibration.md),
[remote-training](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/remote-training.md),
[distributed-rl](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/distributed-rl.md),
[numerical-health](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/numerical-health.md),
[ataraxos](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/ataraxos.md),
[recent-events](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/recent-events.md),
[experiment-execution](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/experiment-execution.md),
[game-crash-recovery](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/game-crash-recovery.md).
