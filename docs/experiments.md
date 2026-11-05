# Experiments

An experiment here is a question with a numeric prediction, committed to git
before anything runs, and a report that stays on the record whichever way it
goes. This guide covers how a question becomes a report, what makes a result
count, and the tools that declare and execute a comparison. The
[research program](../manabot/RESEARCH.md) says which questions are open; the
[experiments index](../experiments/README.md) lists every report.

- [What an experiment is](#what-an-experiment-is)
- [How a claim gets made](#how-a-claim-gets-made)
- [A worked example](#a-worked-example)
- [Failures worth reading](#failures-worth-reading)
- [Declaring a comparison](#declaring-a-comparison)
- [Reading a report](#reading-a-report)

## What an experiment is

One experiment is one module beside its report:
`experiments/<name>.py` and `experiments/<name>.md`. The module holds the
recipes, built from [`manabot.recipes.builders`](../manabot/recipes/builders.py),
and anything only this experiment needs to run and report; `manabot/` keeps the
reusable instruments. A file defining `report()` or `eval(state)` is an
[Etude](https://github.com/loopflowstudio/etude) experiment, and its training
runs track under its stem:

```sh
uv run etude launch experiments/seat_comparison.py --revision COMMIT --cohort seats-1 --rent l4 --max-usd 60
uv run etude eval experiments/elemental.py --source r2://manabot/jobs     # play what is new
uv run etude report experiments/ --source r2://manabot/jobs --serve       # every experiment
```

`runs()` names the training runs, `eval(state)` returns the matchups still
wanted so repeated passes play only new ones, and `report()` draws the figures.
Raw results live in the artifact store, not in git; `experiments/data.manifest.json`
records each file's digest ([data](../experiments/README.md#data)).

## How a claim gets made

The full rules are in [how we measure](../manabot/RESEARCH.md#how-we-measure)
and the [experiment discipline](../experiments/README.md#discipline). These
matter most on day one:

- **Three kinds of progress are easy to confuse.** Optimization progress
  (loss, KL, entropy), relative progress (wins against the latest opponent)
  and durable progress (strength across held-out opponents). The first two can
  improve while the third gets worse.
- **Predictions come first.** The question, numeric prediction, kill criteria
  and cost cap are committed before a run. Without that, a result is
  exploratory.
- **Seeds are the unit of evidence for a method.** Many games from one
  training run don't replace independent training runs. Three seeds are three
  data points.
- **Games are paired.** Candidates play the same deals, with both decks and
  both seats, which removes a large source of variance
  ([evaluation](evaluation.md#one-deal)).
- **Results belong to a world.** Numbers don't carry across
  [world versions](../WORLDS.md).
- **Refutations stay on the record,** dated and never silently edited.

## A worked example

[exp-04](../experiments/exp-04-potential-shaping.md) is from an earlier world,
so read it for its shape rather than its numbers. The project believed that
shaped rewards (pay per land, creature and damage) sped up learning, and an
earlier run had shown those rewards training agents into cast-everything play.
The question was whether a policy-invariant form of shaping would learn without
that fingerprint. The prediction was committed first:

> E2c reaches ≥90% of E2b's win-vs-random while holding instants
> significantly longer. […]

The prediction held, and an addendum overturned the premise anyway: a
terminal-only control matched the potential-shaped arm and beat pay-per-event
shaping by about 14 points, so the shaping the project had been paying for was
buying nothing. Reward in every recipe since has been win, loss or draw and
nothing else. The report records the question, the prediction, the method, the
results, a verdict on the prediction and the caveats, in that order, which is
the shape reports here aim for.

## Failures worth reading

Refuted ideas teach faster than wins, and these stay on the record. Most come
from an earlier world, so read them for the reasoning, not the numbers.

| What we believed | What happened | Lesson |
| --- | --- | --- |
| Shaped rewards speed up learning | Terminal reward matched potential shaping and beat pay-per-event shaping by about 14 points ([exp-04](../experiments/exp-04-potential-shaping.md)) | Test the obvious helper |
| Expert iteration compounds | Round 1 lost 74–26 to round 0 ([exp-07](../experiments/exp-07-expert-iteration.md)) | Distillation doesn't improve a weak teacher |
| Win rate measures competence | A strong-looking search pilot couldn't play a control deck ([exp-09](../experiments/exp-09-control-competency.md)) | Test specific skills, not just outcomes |
| Matchup tables measure the decks | A four-card swap moved a matchup by 55 points ([exp-08b](../experiments/exp-08b-ancestral-dose.md)) | The pilot and the cards are confounded |
| An N-game evaluation played N deals | A seed bug made every game replay one deal ([F1](../paper/understanding.md)) | Check the instrument before the result. Seat-advantage claims built on it were retracted. |
| A borrowed technique helps by default | Actor-only filtering: +3.33 points, interval [−4.33, +11.33], at 1.45× wall time ([technique screen](../experiments/ataraxos-technique-screen.md)) | Keep it as a switch; promote only on evidence |

## Declaring a comparison

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
the [current-game recovery policy](training.md#current-game-recovery) on every
learner. A `pause.request` file in the output directory requests a pause at
the next learning boundary.

The report is one editable `comparison.ipynb` per execution directory, created
only when absent so edits survive; it reads retained files only and renders
`comparison.html`. `demo_declaration()` in
`tests/training/test_experiment_runner.py` is a complete small declaration;
the retained [demo HTML](../tests/fixtures/experiment-demo/comparison.html)
and [evidence note](../tests/fixtures/experiment-demo/README.md) show the
output. Interpretation belongs to the experiment author, never the runner.

Remote runs go through Etude ([remote training](training.md#remote-training)).

### Freeze step targets

`manabot.experiment.execution.StepCalibration.from_export(path)` reads a
completed single-stage fresh self-play `run.json` and measures completed
iterations per collection-plus-learning second; `updates_for(hours)` floors to
whole iterations. Use a representative, separately authorized pilot per
capacity; apply each target to `TrainSelfPlay.updates` and one shared interval
to every `checkpoint_updates`; commit the frozen declaration before launch.
Execution never consults pilot rates. Equal steps do not mean equal
transitions, exposures, active time or dollars.

## Reading a report

Reports select over Etude jobs and average across seeds, and their metric
definitions are in [experiment-metrics.md](experiment-metrics.md). When reading
a training curve, keep the three kinds of progress apart: `rl/loss`,
`rl/collection_kl` and `rl/entropy` in the report (`loss`, `collection_kl` and
`entropy` in the raw Etude rows) describe optimization; monitoring scores
against the latest opponent describe relative progress; only held-out
evaluation on paired deals describes durable progress
([reading a run](learning.md#reading-a-run)).
