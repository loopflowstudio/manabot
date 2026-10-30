# Training manabot

This is the current-system guide to training a manabot: how a run is declared
and executed, how comparisons and remote jobs are built, what the monitoring and
numerical-health records mean, how interrupted or failed work is recovered, and
which learning rule the default recipes implement. Metric definitions for the
experiment report live in [experiment-metrics.md](experiment-metrics.md);
evaluation protocols live in [evaluation.md](evaluation.md).

- [Run training](#run-training)
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
uv run manabot train --regime-name direct_self_play --seed 197 --out .runs/direct-pilot
uv run manabot train --regime regime.json --seed 197 --out .runs/example
```

`--recipe local` (`manabot.recipes.local`) is a two-update production-path
smoke with no evaluation. `--recipe baseline` (`manabot.recipes.baseline`,
version `baseline-v3`) runs the ordinary baseline through the experiment runner
with initial and endpoint evaluation. `--steps`, `--seconds` and
`--learning-seats` apply only to these recipes. `--preset` and `--set` are
retired and fail.

Each experiment is one module, `experiments/<experiment>.py`, next to its
write-up. Its canonical recipe is the zero-argument function named like the
module returning a `TrainingRegime`; other such functions are variants.
`--regime-name <experiment>` or `<experiment>:<variant>` selects one, and
`manabot.cli.regime_names()` lists them. Reusable composition helpers are in
`manabot.recipes.builders` (`move_learning`, `move_checkpoints`,
`ataraxos_baseline`, `with_capacity`, `with_value_output`,
`with_recent_events`, ...). Write a regime file from Python:

```bash
uv run python -c "from experiments.batched_baseline import batched_baseline; print(batched_baseline().model_dump_json(indent=2))" > regime.json
```

Regime runs require `--out`. The CLI opens the canonical `training.sqlite` in
the **parent** of `--out`; choose a new output directory for every attempt.
`--regime` and `--regime-name` are exclusive. `--resume-from`,
`--source-repair-reason` ([recovery](#recovery)), `--checkpoint-updates` and
the legacy `--checkpoint-seconds` ([monitoring](#monitoring)) apply only to
regime runs.

## Regimes and stages

A `TrainingRegime` (`manabot.training.regime`) is an ordered recipe; a
`TrainingRun` (`manabot.training.records`) is one bounded execution with
resolved settings, named seed streams, artifacts, measured costs and status.
`execute_regime` (`manabot.training.execution`) runs it. `VerifyStore` is the
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
| `train_compound` | Ragged compound-decision policy; see [compound decisions](#compound-decisions). |
| `import_policy` | Admit a frozen TrainingRun export's raw or EMA checkpoint without training. |
| `collect_selection` | Frozen-policy complete-game selection diagnostics; no weights change. |
| `collect_belief` / `train_belief` | Collect hidden-hand labels from a policy stage and fit a sampler; see [search-and-beliefs.md](search-and-beliefs.md). |

Every stage declares device, float32 precision, worker count, Torch thread
count, wall and memory limits (`Execution`). `execution.workers` accepts 1
(default), 2 or 4 CPU observation producers for self-play only: persistent
threads own disjoint native stream ranges, a barrier completes all writes
before the single policy owner reads, and stream identities and seeds are
unchanged across partitions. `execution.threads` controls Torch threads
independently. More workers are not a recommended default; the matched GPU
comparison is in
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
opponents default to one seat and are never trained on. The default was
selected by Jack Heart on 2026-10-20 and does not by itself establish a
strength or cost gain.

Both-seat collection records each acting seat's behavior outputs; each seat
gets its own trajectory and terminal win/loss/draw reward (draws zero), and
GAE and lambda returns stop at each seat's game boundary. `transitions` is a
**minimum** decision count per native stream counting both seats; the stream
then finishes its current game, so batches have variable row counts. Complete
seat trajectories are packed into `[rows, 1]` with terminal separators and no
padding. `games_per_stream` overrides the quota with complete games per stream
per update; set it on both arms for an equal-game comparison and report
learner transitions, environment decisions, games, exposures and cost
separately. For two live models, use one `SeatRoutedCollector` with
`opponent_mode="learning"`, `learning_seats="both"` and `opponent_agent`;
`collect_pair` returns each model's rows.

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

### Compound decisions

`ModelConfig.compound_decisions=true` selects the learned ragged policy;
`train_compound` collects complete self-play games before each update. The
recipes are `experiments/compound_decisions.py` variants
`{sequential,grouped}_{bootstrap,outcome}`.

managym's `compound_offers` supplies the legal surface: complete attacker
declarations and simple single-target casts; complete remaining blocker
declarations when each blocker's legal targets are independent (menace stays
sequential); and waterbend payments with fixed tap/mana economics. Kicker,
ward, mana abilities and triggers remain sequential decisions. A native
`compound_commands_json` call lowers a sampled submission on an exact fork into
ordinary revision-bound Commands; the live root is untouched, and interruption
rejects a stale suffix. No Python code reconstructs these rules.

A small GRU conditions include/exclude logits on the selected prefix over the
shared viewer encoder. Candidates are visited once in native order with
cardinality masks, so each subset has one encoding; joint log probability is
the sum of normalized conditional log probabilities, and forced tokens have
log probability zero. Conditional entropy and KL terms are sampled-prefix
regularizers, not exact joint quantities. `grouping=grouped` uses one joint
ratio and root value per submission; `sequential` uses per-factor ratios and
prefix values. `estimator=outcome` uses terminal Monte Carlo returns;
`bootstrapped` uses transition-end GAE. Recipes fix gamma=1. Raw weights only;
EMA is unsupported. Native `skip_trivial` stays on; report complete-game wall
cost, since fewer exposed prompts alone are not a speedup.

```bash
OMP_NUM_THREADS=1 uv run --extra notebook python -m experiments.compound_decisions --profile smoke --out .runs/compound-smoke
uv run --extra notebook python -m experiments.compound_decisions --report-only .runs/compound-smoke
```

The smoke is one seed; a scientific comparison needs independent training
seeds, fresh deals, equal wall budgets and measured inference cost.

### Selection diagnostics

`collect_selection` freezes a prior self-play stage's raw or EMA checkpoint
(optionally from another run in the same store via `source_run`) and collects
a declared population with unique seeds and whole-game development/held-out
splits. Scalar and categorical values are supported; compound and belief-input
sources are not. The protocol is in
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

`manabot.experiment.config.Experiment` declares a typed comparison. `resolve()`
returns complete validated `TrainingRegime` values plus a provenance receipt;
it never builds a model, opens a store or launches work.

```python
from experiments.batched_baseline import batched_baseline
from manabot.experiment.config import Axis, Baseline, Case, Experiment, ModelOverride
from manabot.infra.config import ModelConfig

experiment = Experiment(
    name="value-output",
    baseline=Baseline.capture("control", batched_baseline()),
    matrix=(Axis("output", (
        Case("scalar", (ModelOverride(ModelConfig(value_kind="scalar")),), "Scalar"),
        Case("wdl", (ModelOverride(ModelConfig(value_kind="categorical_wdl")),), "WDL"),
    )),),
)
resolved = experiment.resolve()
regime = resolved.regimes["value-output-wdl"]
receipt = resolved.receipt()
```

`cases` vary together; `matrix` axes cross, last axis fastest. IDs join
experiment, case and choice names with hyphens; names are lowercase
alphanumeric. Labels are presentation only.

| Component | Owns |
| --- | --- |
| `ModelOverride(ModelConfig(...))` | Explicitly supplied `agent.*` fields. |
| `Environment(world=, match=, observation=)` | World, matchup, input capacity; a supplied deck map replaces the whole map. |
| `LearningRule(Learning(...) \| DampedPolicyLearning(...))` | The complete learning rule on every self-play stage. |
| `Resources(execution=, wall_seconds=)` | Explicit execution fields on every stage and the run deadline. |
| `Pipeline(stages=(...))` | The complete ordered stage list. |
| `RunControl(...)` | `schedule_clock` (default `iteration_fraction`), `recovery_max_microsteps`, `selection`, `recovery_every_updates`. |

Every setting has exactly one owner. Writes from common overrides, cases and
axes may replace a baseline value but never each other, even with equal
values; overlapping parent/child writes fail (so `Pipeline` cannot accompany
`LearningRule`). There is no last-write-wins. Pydantic's explicitly-set fields
define a patch, so an explicit default still counts as an override.
Resolution runs `validate_regime`; artifact existence and runtime binding stay
execution checks.

`Baseline.capture(name, regime)` freezes all current serialized defaults into
immutable bytes; its identity is the full snapshot. `ResolvedCase.digest` is
the canonical regime digest; `ResolvedCase.identity` also hashes label,
baseline and provenance. Tests use the small categorical-WDL baseline in
`tests/training/wdl_baseline.py`, built with `ataraxos_baseline`. A
declaration is not a study type: the arena studies in
`manabot.experiment.study` admit only their own arms, seeds and budgets.

### Execute locally

`Experiment.schedule = ExperimentSchedule(...)` declares seeds, hardware,
`wall_seconds`, `process_seconds`, `MonitoringBudget` and `checkpoint_updates`.
`manabot.experiment.runner.run_experiment(experiment, hardware, out)` executes
it and records actual attempts in `ExperimentRun`.

```python
hardware = HardwareInventory.model_validate_json(Path("hardware.json").read_text())
result = run_experiment(experiment, hardware, Path(".runs/my-comparison"))
result = run_experiment(experiment, hardware, out, resume=True)              # pending work only
result = run_experiment(experiment, hardware, out, resume=True, recover=(0,))  # recover attempt 0
```

`HardwareInventory` lists resource names, exact hostnames, `backend:
"local-cpu"`, `cpu_threads` and optional `dollars_per_hour`. Only that host and
CPU are admitted; there is no discovery, provisioning or device fallback. One
learner and at most one evaluator run concurrently; reserve a thread for
evaluation. `wall_seconds` bounds the whole execution; `process_seconds`
bounds additive learner plus evaluator time. The full monitoring allocation
is reserved before each learner; jobs that do not fit stay pending rather than
shrink. Failed learners keep their exported checkpoints; stopped evaluations
are not retried; resume grants no extra budget. `active_runtime=True` requires
the [current-game recovery policy](#current-game-recovery) on every learner.
A `pause.request` file in the output directory requests a pause at the next
learning boundary.

The report is one editable `comparison.ipynb` per execution directory, created
only when absent so edits survive; it reads retained files only and renders
`comparison.html`. `demo_declaration()` in
`tests/training/test_experiment_runner.py` is a complete small declaration;
the retained [demo HTML](../tests/fixtures/experiment-demo/comparison.html)
and [evidence note](../tests/fixtures/experiment-demo/README.md) show the
output. Interpretation belongs to the experiment author, never the runner.

### Freeze step targets

`manabot.experiment.execution.StepCalibration.from_export(path)` reads a
completed single-stage fresh self-play `run.json` and measures completed
iterations per collection-plus-learning second; `updates_for(hours)` floors to
whole iterations. Use a representative, separately authorized pilot per
capacity; apply each target to `TrainSelfPlay.updates` and one shared interval
to every `checkpoint_updates`; commit the frozen declaration before launch.
Execution never consults pilot rates. Equal steps do not mean equal
transitions, exposures, active time or dollars.

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
uv run etude launch experiments/<name>.py --revision COMMIT --rent l4 --max-usd 10
uv run etude report experiments/<name>.py --source r2://manabot/jobs --open
```

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

`TrainingJob.continuation` (`LearnerContinuation`) continues another run's
learner from its published `run.json` and raw/EMA/Adam exports, bound as a
`LearningStateImport` (see [portable continuation](#portable-learner-continuation));
the seed must be the parent's. `etude continue` resumes from a policy
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
`time/learning/minibatches/forward_and_loss`), `selection_groups/<action
type>/<terminal distance>/rows` and `self_play_outcomes/<deck>/<opponent
deck>/<play|draw>/wins`. Offline readers use
`manabot.training.update_metrics.read_updates`. Compound, selection and belief
summaries stay in their stage's `diagnostics`.

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
and compound recipes keep explicit legacy cadence. `--checkpoint-seconds`
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

`--destination` (directory or `s3://bucket/prefix`) and `--experiment` select
delivery; settled rows are never rewritten and a changed prefix is rejected.

### Stored bytes

Jack Heart selected `s3://etudefantasia/manabot/` for retained bytes and
`loopflow-studio/etude` for W&B metrics on 2026-10-06 (bucket in `us-west-2`,
public access blocked, AES256, versioned). Use the standard AWS credential
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

### Tolerating engine failures in training

A single-stage `TrainSelfPlay` may opt into `game_errors`
(`GameErrorPolicy`, `manabot.training.state.game_errors`):

```json
"game_errors": {"threshold": 0.1, "window": 100, "warmup": 100, "max_game_actions": 40000}
```

The denominator is settled native game attempts. After 100 settle, the latest
100 are assessed: 10 failures are tolerated, 11 stop the run, and the stop
survives restart and source repair. An engine rejection or native panic
quarantines only that stream's current game, removing both seats' rows from
it; other streams advance once. Replacement uses the native next seed and the
declared matchup distribution; nothing is blacklisted. One-seat collection
also finishes games before updating when opted in, which changes its batch
freshness and must be declared. Truncations and the action limit count as
failures; run interruptions are recorded as `deadline` or `abandoned`
attempts, not errors. Setup, reset, buffer, numerical and infrastructure
errors remain fatal; process aborts, OOM and hangs belong to the external
supervisor.

`game-attempts.jsonl` is a private, append-only, fsynced ledger: setup, seed,
stream, seat, full action prefix including the rejected action, native
pre-command frame, error and identities. Never publish it as a replay. Replay
a receipt by its one-based line:

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
pool roots), compound/selection journals, and arena evaluation, which journals
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

The default move-learning rule adapts the Ataraxos Stratego method to MTG. It
is not an exact reproduction or a strength result. Sources (checked
2026-10-04 and 2026-10-05): the
[final Nature paper](https://www.nature.com/articles/s41586-026-11036-y) and
[supplement S3.2–S3.7, Tables S7/S10](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf)
(SHA-256 `74f217e942cd824140dd5221c02d8c36821a715664f47aee6f489f0ca6932a08`).
The damped move update **does use clipped importance ratios**.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run manabot train --regime-name ataraxos_move --seed 601 --out .runs/move
uv run manabot train --regime-name ataraxos_move:ataraxos_move_scalar --seed 601 --out .runs/move-scalar
```

These `experiments/ataraxos_move.py` recipes are bounded two-stage, one-thread,
180-second proofs. Ablation recipes are in `experiments/ataraxos_ablation.py`.

### Objective

`manabot/training/learning/damped_policy.py` owns the objective; recipes select
`DampedPolicyLearning(gradient="damped_policy")`. Historical `ataraxos_move`
inputs decode at the compatibility boundary with their original tag and
digest. Per retained observation the policy loss is the negative clipped-ratio
advantage surrogate plus `collection_kl * KL(current || behavior)` and
`tau * KL(current || magnet)`, summed over all legal offers. Advantages use
signed expected outcomes with their own lambda. Outcome targets are three-way
lambda mixtures stored in **loss, draw, win** order with one-hot terminals; the
critic gets cross-entropy, equally weighted with the policy loss. The scalar
ablation uses squared-error lambda returns instead. Only current logits receive
gradients: behavior, references, estimates and targets are detached; the
shared encoder receives both losses.

Collection freezes learner weights within each batch; both-seat and
`games_per_stream` collection finish games before updating, while one-seat
quota collection may span updates. One epoch still changes parameters between
minibatches, so per-action ratios remain necessary. No replay buffer,
trajectory ratio product or state-occupancy correction is implemented.
Filtering changes the training-state measure, as in the source, and
advantages are not renormalized after filtering. Truncation fails rather than
becoming a draw; opponent actions are traversed, not trained on.

Optional controls: `filter_scope="actor"` keeps all critic rows;
`behavior="ema-self"` collects and bootstraps with averaged weights
([protocol](../experiments/ataraxos-omitted-controls.md)). EMA evaluation
outputs are selected explicitly at arena registration.

### Fidelity table

| Technique / source | Implementation | Boundary |
| --- | --- | --- |
| S3.4 eq. (6), clipped surrogate | `damped_policy_loss`; both clipping branches tested | Same loss |
| Both reverse KLs, S3.4 | Exact legal-support sums against saved behavior and magnet | Finite legal log support required; rounded zeros counted, no smoothing |
| Magnet, S3.4 | `reference_distribution`, action-type uniform by default; uniform-offer control | MTG action types replace piece-then-move groups |
| Scalar advantage, S3.4 | Transition-end GAE, gamma=1, lambda=.5; detached collection values | Learner decisions replace same-player positions |
| Outcome estimator, eq. (5) | Vector lambda=.8, terminal one-hot, paused-tail bootstrap | Loss/draw/win storage order is a permutation |
| Value representation, S3.4 | Categorical head with CE | Scalar MSE is an explicit ablation |
| Filtering, S3.4/Table S7 | Inclusive .75 quantile and .01 minimum; ties included | Torch linear quantile; source interpolation unspecified |
| Optimizer damping, Table S7 | Adam, clip=.2, collection KL=.1, max norm=.267, value weight=1 | Adam betas .9/.999, epsilon 1e-5 (source unspecified) |
| Schedules, Table S7 | LR clip(.5 / iteration^1.1, 5e-6, 1e-4); tau=.05 / iteration^.3 | One-based iterations; no elapsed-budget substitution |
| Batch reuse, S3.4 | One epoch, increasing timestep batches, no shuffle | CPU streams replace 202-step GPU batches of 1536 |
| EMA, S3.4 | .999 per iteration including empty filter; parameters averaged, buffers copied | Initialized from learner; evaluation only |
| S3.3 setup learning | Not implemented | MTG has no learned setup |
| S3.5 belief, S3.7 search | Outside this trainer | See [search-and-beliefs.md](search-and-beliefs.md) |
| S3.6 network scale | Semantic Agent, small width, CPU float32 | The paper uses an 8-layer width-384 network on GPUs |

**Unresolved source details:** Adam epsilon/betas, iteration origin, EMA
initialization, quantile interpolation and exact bootstrap pseudocode at a
collection boundary. The choices above are explicit implementations, not
recovered details. Supplement S5–S7 recipes are not blended in; DeepNash's
R-NaD is not implemented. Departures also include post-normalization
attention (the paper specifies pre-normalization), the value token as a local
choice, and missing ordered past-move inputs.

Before any scientific comparison of PPO, categorical and scalar move learning,
freeze a protocol, independent seeds, budgets, world and code identities,
held-out paired deals with both seat/deck assignments and the raw/EMA choice;
keep all failures and compare overlapping cost intervals. The technique
inventory is [ataraxos-technique-screen](../experiments/ataraxos-technique-screen.md);
ablations are in [ataraxos-mtg-ablations](../experiments/ataraxos-mtg-ablations.md).

## Policy inputs

### Current continuation input

Fresh `ObservationConfig` uses policy input v5. A native `relations[64,33]`
float32 buffer carries ordered stack entries, announced targets and clauses,
chosen modes and abilities, kicker, reserved sacrifices, suspended payment
obligations and waiting triggers with their context; v5 also widens player
rows to 60 columns and card rows to 55. Native buffers bind the version by
relation width (32 for v2/v3, 33 for v4/v5) and, at 33, player width (40 for
v4, 60 for v5); validity is column 31. Checkpoints omitting the version load as
v1 through `ObservationConfig.from_checkpoint`; loading never upgrades weights.
`agent/continuation.rs` projects the facts once per observation and
`manabot/env/observation.py` is the matching reference encoder. Only the
acting viewer receives private continuation; overflow fails rather than
truncating. Transport IDs are joins to the card catalog and object table,
never numeric features. At `Decision::ManaChoice` the activation source is
unavailable because rules state has discarded it.

Encode-only benchmark: `cargo run --release --features benchmark --manifest-path managym/Cargo.toml --example bench_observation`.
Matched full-vector benchmarks use `scripts/bench_breakdown.py --matched-corpus`
and `scripts/bench_observation_pair.py` on a release extension built with the
`benchmark` feature. On 2026-10-26 the paired comparison on a 32-vCPU EPYC
host showed full-step ratios of 0.978–1.051, which met Jack Heart's revised
2026-10-26 shipping acceptance permitting small measured overheads.

### Optional recent public events

`ModelConfig(recent_events=True)` (or `with_recent_events` / `ModelOverride`)
adds shared policy/value context from a bounded suffix of public events; the
regime derives `observation.policy_history_version=1`. `Game::emit` maintains
the projection at the same append as the rules ledger, keeping at most 32
rows; private arrivals produce no row, and owner (not controller) is recorded.
History-on uses an `[E,12]` tensor (`1 <= E <= 32`) with a validity mask:
kind (public arrival, damage, life change, spell cast, resolved, countered),
signed amount, and source/target references `(definition+1, owner_role,
zone+1, context_index, availability)`. Valid rows are mean-pooled and added to
shared object embeddings; an empty window gives exactly zero context.
History-off keeps the historical `[E,7]` tensor and parameters. This is not
complete decision history, and no training benefit has been measured.

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
