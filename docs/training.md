# Training manabot

[How a manabot learns](learning.md) explains the learning loop itself, and
[training loops](training-loop.md) lists the updates and pieces a loop is made
of. This guide is about running training: recipes, what a run leaves behind,
running on rented machines, monitoring, numerical health and what happens when
a run is interrupted. Comparisons are declared in
[experiments.md](experiments.md); metric definitions live in
[experiment-metrics.md](experiment-metrics.md); evaluation protocols live in
[evaluation.md](evaluation.md).

An update learns from one batch (`manabot/training/updates/`), a loop keeps
playing and updating until it is done (`manabot/training/loops/`), and a recipe
is our loops with our settings (`manabot/recipes/`). Write your own recipe or
use one of ours, and use [Etude](https://github.com/loopflowstudio/etude) to
launch, track and evaluate it, or bring your own.

- [Run training](#run-training)
- [Write a recipe](#write-a-recipe)
- [Recipes in stages](#recipes-in-stages)
- [Self-play settings](#self-play-settings)
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

A recipe is a file with a `train(config)` function, and
[Etude](https://github.com/loopflowstudio/etude) runs it. Everything a run
records is ordinary files in a directory you own.

```bash
uv run etude run manabot/recipes/local.py                                # a two-update smoke on CPU
uv run etude run manabot/recipes/elemental.py --seed 197 --name elemental-197
```

Each command starts a new run and prints its id. The second trains the current
Elemental recipe (damped-policy self-play on the ten Elemental decks, both
seats learning) as the run `elemental-197` of the experiment `elemental`, the
recipe file's name. Local runs are written under `~/.etude/track` (moved by
`ETUDE_HOME`):

```text
~/.etude/track/elemental/elemental-197/
  run.json              the job, seed, config and clocks
  ...                   logged rows and tables
  out/
    state/              what the next attempt loads; replaced whole at each save
      state.json        where each clock stood, the seed, the commit, cumulative counts and the file names
      learner.pt        raw weights (a `state_dict` written with `torch.save`)
      optimizer.pt      Adam
      ema.pt            EMA weights
    policies/           world-bound raw and EMA policies, named by update and never rewritten
    game-attempts/      append-only ledgers when game-error handling is configured
    failures/           private replay evidence, created when a game fails
```

The policy a run trained is its last checkpoint: `out/policies/00003200-ema.pt`
for a recipe that reached update 3200. `state/` reads with `json` and
`torch.load`, without Etude or manabot.

| Option | Sets |
| --- | --- |
| `--config JSON` | Fields of the recipe's config; the rest keep their defaults. `--config '{"device": "cuda", "streams": 64}'` |
| `--seed N` | The run's seed (0 when omitted). The seed is the run's, never a config field. |
| `--name NAME` | The run's id. It must be new within its experiment: a taken name is refused, never resumed. |
| `--until updates=N` | Where this run stops, in place of the recipe's own update count. |
| `--seconds N` | A deadline. A run that reaches it saves its state and stops. |
| `--resume RUN` | Another attempt of an existing run instead of a new one ([recovery](#recovery)). |

The command exits 0 when the run finished, 75 when it saved and stopped on
purpose (a deadline or a stop request), and with another status when it
failed. A run that ended without finishing, whether it stopped, failed or was
interrupted with Ctrl-C, is carried on with `--resume`:

```bash
uv run etude run manabot/recipes/elemental.py --resume elemental-197
```

That is a new attempt of the same run on the current code, from its last saved
state, with the config and seed the run recorded. Work since that save is
repeated. `--resume` refuses a finished run, and refuses a config or seed that
differs from the run's, naming the fields. To train a finished run further,
continue it as a new run that starts from the first one's state:

```bash
uv run etude continue --run elemental-197 --until updates=6400
```

`--until updates=N` sets only the stop target. It leaves the recipe's own
update count, which fixes the baseline rates and the checkpoint grid,
unchanged: `manabot/recipes/baseline.py --until updates=32` runs the first
quarter of its 128-update schedule
([schedules and clocks](#schedules-and-clocks)).

The recipes we maintain are the files in `manabot/recipes/` that define
`train`: `local.py` (a two-update smoke with no evaluation, for checking an
installation), `baseline.py`, `elemental.py`, and `belief_policy.py` with
`belief_sampler.py` ([recipes in stages](#recipes-in-stages)). An experiment
file that defines `train` runs the same way
(`uv run etude run experiments/batched_baseline.py`). A recipe's device is a
config field with the recipe's own default (`cpu` for local and Elemental,
`cuda` for baseline); nothing selects a device automatically. Frozen
zero-argument regime builders are data readers, not executable recipes.

`manabot train` is removed. It exits naming the `etude run` command above. Its
`--out` directories, which kept the learner in one `state.pt`, are not read
by this loop and there is no loading path for them; the last revision that
resumes one is `f577362135562e569033fe33e48c7ef0411a829d`. Their policy
checkpoints load as before.

CPU is the default and recommended device on Apple Silicon. On a quiet Apple
M2, MPS was slower than CPU for collection at every measured batch size, only
tied 8-thread CPU for learning, and its driver memory grew past 20 GB within a
few learner steps; `torch.compile` and fixed minibatch sizes did not change
that. MPS is admitted on explicit request (`--config '{"device": "mps"}'`) for
float32 self-play and requires Torch 2.15 or newer: on an Apple M2, Torch
2.10–2.13 compute wrong MPS linear-layer weight gradients, which gradient
clipping then applies (see `manabot/training/mps.py`). The repository lock
installs 2.15; with an older Torch, an MPS run stops before training with a
message naming both versions. An unavailable device fails explicitly. Reduced
precision remains CUDA-only. Belief inputs and frozen opponents are unsupported
on MPS. Evaluation and search stay on CPU. A separate
[MLX backend](../manabot/backends/mlx.md) of the policy and the damped update
reads the same checkpoints, converted to the shared container, for measurement
on Apple Silicon; no recipe runs on it. This support makes no speed or strength claim.

### Saved regimes

Before recipes were functions, a run was a saved `TrainingRegime` executed by
`execute_regime`. Saved regime files, their `run.json` records and their
checkpoints stay readable: `manabot.training.records` reads every retained
record and `load_checkpoint` loads every world-bound checkpoint as before,
with serialized keys unchanged. They no longer execute. Their last executor
revision is `f4e1822faf46bbc9cdc43909293f8cbc8de4a300`. For an exact
reproduction, use the experiment's recorded revision and lock; the revision
above retains the executor.

## Write a recipe

A recipe is a training function: `train(config)`, an ordinary typed function
whose body shows what runs. It opens its run with Etude's `track` and calls a
loop. Nothing wraps it; `etude run` and a launched job both simply call it.
The config holds only what varies between runs of the recipe, and never the
seed. Everything else is code in the function. This is the Elemental recipe
(abridged: its optional CUDA numerics gate, `validate_numerics`, is left out):

```python
# manabot/recipes/elemental.py
from typing import Literal

from etude.train.run import track
from pydantic import BaseModel, Field

from manabot.recipes.builders import elemental_self_play
from manabot.training.loops.self_play import CLOCKS, self_play


class ElementalConfig(BaseModel):
    updates: int = Field(default=3200, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    streams: int = Field(default=256, ge=2)


def train(config: ElementalConfig) -> None:
    settings = elemental_self_play().model_copy(
        update={"device": config.device, "streams": config.streams}
    )
    with track(clocks=CLOCKS, config=config) as run:
        self_play(settings, run, updates=config.updates)
```

What Etude asks of a recipe:

- The config is a Pydantic model. There is no Etude base class and no required
  field. It survives a trip through JSON, because that is how it reaches a
  rented machine; Etude checks this at launch and refuses there.
- `train` takes that one parameter, annotated with the config's type, and sits
  at the top level of an importable file.
- A job has no seed; a run is a job with a seed. The loop reads `run.seed`.

Calling `train(config)` is one attempt at the whole job. It returns (the run
finished), raises `Stopped` (it saved and left on purpose; it is called
again), or raises anything else (the run failed).

`CLOCKS` are the two x-axes every self-play run declares:

```python
CLOCKS = [
    Step("updates", "collection/update iterations"),
    Time("training_seconds", "active collection and learning seconds"),
]
```

The `updates` clock counts collection/update iterations: one batch collected
and learned from. It is not a count of `optimizer.step()` calls, which a run
logs separately ([costs and counters](#costs-and-counters)).

`self_play` (`manabot.training.loops.self_play`) is the loop every current
recipe uses. Read it as the template for your own; its body is the loop you
would write:

```python
def self_play(settings: SelfPlay, run: Run, *, updates: int) -> Path:
    """Train to the run's update target and return the final policy's path."""
    with device_settings(settings):
        learner, ema = create(settings, seed=run.seed)
        counts = run.resume(
            learner=learner.agent, optimizer=learner.optimizer, ema=ema, extra=Counts()
        )
        with self_play_games(settings, run) as collector:
            if counts.updates == 0:
                publish(run, learner, ema, counts)  # the untrained policy
            for update in run.steps("updates", to=updates):
                counts, row = step(settings, learner, ema, collector, counts)
                run.log(row, training_seconds=counts.training_seconds)
                run.log_table(
                    OUTCOME_TABLE,
                    collector.stats.self_play.drain(),
                    training_seconds=counts.training_seconds,
                )
                run.save()
                if is_checkpoint(settings, update):
                    publish(run, learner, ema, counts)
        return publish(run, learner, ema, counts)
```

It returns the policy it trained: the EMA checkpoint of its last update when
the learning record keeps an EMA, the raw one otherwise. There is no resume
branch, stop check or state file in it. Those are the run's:

| From Etude | What the loop gets |
| --- | --- |
| `track(clocks=, config=)` | The run. Leaving the block finishes the attempt: as stopped when the block raised `Stopped`, as failed on any other exception. |
| `run.seed` | This run's seed. |
| `run.resume(**objects, extra=)` | Registers what the run keeps, anything with `state_dict()` and `load_state_dict()`, and loads the last save into it. `extra` is a Pydantic model for the loop's own numbers, here `Counts`; it comes back as saved, or as given on a fresh run. |
| `run.steps("updates", to=)` | Counts from the resumed update to the target: the launch's (`--until`, a continuation) or else `to`. When a stop is asked for or the deadline is near, the current pass finishes and the next raises `Stopped`, so the lines after the loop run only when the target was reached. |
| `run.save()` | Called every pass. Etude writes the registered objects when a save is due and always on the last pass before a stop; a write replaces the previous state whole or not at all. |
| `run.log`, `run.log_table` | Metric rows and row tables on the declared clocks. The stepped clock is filled in; the loop passes `training_seconds`. |
| `run.checkpoint(path)` | Publishes an immutable policy with where every clock stood. Evaluations, reports and other runs read these. |
| `run.out` | The run's directory, the same on every attempt. Under a launch Etude carries it to storage as it is written and restores it before an attempt on another machine. |
| `run.input(name)` | A file this run was given at launch, checked by digest ([recipes in stages](#recipes-in-stages)). |

`track` names no experiment. Under `etude launch` the run belongs to the
experiment file that launched it, and `etude run` records it under the recipe
file's name.

The pieces that are manabot's:

| Piece | What it does |
| --- | --- |
| `SelfPlay` | The loop's settings: match and decks, model and observation configs, the update and its learning record, streams, transitions, learning seats, batch closure, game-error policy, workers, device, precision and `checkpoint_every`. A Pydantic model holding data only; everything the loop does with it is a function. A recipe builds one in its body (`elemental_self_play()`). |
| `create(settings, seed=)` | A `Learner` (model and Adam) with fresh weights from `seed`, and the EMA as a copy, or `None` when the learning record keeps none. `run.resume` then loads a saved run over them. |
| `Counts` | Cumulative work across every attempt: completed updates, `training_seconds`, games, decisions, learner transitions and optimizer calls. A Pydantic model, saved in `state.json`. |
| `self_play_games(settings, run)` | A context manager around the `Collector` (`open_collector` in `manabot.training.rollout.collection`), the game-error ledger and failure capture under `run.out`. Streams, sampling and minibatch order are seeded from `run.seed` and the update the attempt starts at, so a resumed loop plays fresh, reproducible games. |
| `step(settings, learner, ema, collector, counts)` | One update: collect a batch with the behavior policy, `learn` from it, and return the next `Counts` with the metric row. The learner, Adam and EMA are updated in place. |
| `learn` | The update the learning record selects: `update_damped_iteration` for `DampedPolicyLearning`, `update_iteration` for `Learning` ([training loops](training-loop.md#the-updates-we-maintain)), then the EMA. Rates come from the completed update count alone and never from the target ([schedules and clocks](#schedules-and-clocks)). |
| `publish(run, learner, ema, counts)` | Writes world-bound raw and EMA policies at this update into `run.out / "policies"` (`NNNNNNNN-raw.pt`, `-ema.pt`), each carrying `checkpoint_source`, and records the one the loop returns with `run.checkpoint(path)`. A file is written once; publishing an update again keeps its file and its one record. |

The loop takes Etude's `Run`; there is no untracked path through it. The
updates and pieces it calls need no run
([training loops](training-loop.md)), and
[`loops/minimal.py`](training-loop.md#a-complete-self-play-loop) is a whole
resumable loop in about ninety lines to copy from.

## Recipes in stages

A recipe whose later stage reads an earlier stage's result is two jobs, and
the second names the first as its start. Split this way when the stages want
different machines or budgets, when several runs start from one, or when the
earlier result is evaluated on its own. The belief recipe trains a policy,
then collects hidden-hand histories with it and fits a sampler:

```python
# manabot/recipes/belief_sampler.py
from decimal import Decimal

from etude.jobs import training
from etude.launch.records import Job
from etude.train.records import Step
from etude.train.run import track
from pydantic import BaseModel, Field

from manabot.belief.collect import collect_belief
from manabot.belief.train import SamplerConfig, train_belief
from manabot.recipes import belief_policy
from manabot.recipes.belief_policy import PolicyConfig


class BeliefConfig(BaseModel):
    games: int = Field(default=3, ge=3)
    max_steps: int = Field(default=1000, ge=1)
    sampler: SamplerConfig = SamplerConfig(
        steps=4, batch_size=8, hidden_size=16, evaluation_samples=4
    )


def train(config: BeliefConfig) -> None:
    with track(clocks=[Step("steps", "sampler gradient steps")], config=config) as run:
        policy = run.input("start")  # the policy run's final checkpoint
        histories = collect_belief(
            policy,
            run.out / "histories.json",
            match=belief_policy.match(),
            games=config.games,
            seed=run.seed,
            max_steps=config.max_steps,
        )
        sampler = train_belief(histories, run.out / "belief.pt", config.sampler, seed=run.seed)
        run.checkpoint(sampler, steps=config.sampler.steps)


def runs() -> list[Job]:
    policy = training(
        belief_policy.train, config=PolicyConfig(), seeds=(197,), max_hours=1, max_usd=Decimal(1)
    )
    sampler = training(
        train, config=BeliefConfig(), start=policy, seeds=(197,), max_hours=1, max_usd=Decimal(1)
    )
    return [policy, sampler]
```

`manabot/recipes/belief_policy.py` is an ordinary one-stage recipe: PPO
self-play on the Gray Ogre match, whose rates follow the horizon the recipe
states (`gray_ogre_self_play(horizon=config.updates)`). `start=policy` gives
each sampler run the final checkpoint of the policy run with its seed, checked
by digest, and records that edge between the two runs. The sampler run starts
only when that policy run has finished.

```bash
uv run etude smoke manabot/recipes/belief_sampler.py
uv run etude launch manabot/recipes/belief_sampler.py
```

Its defaults are a pilot's: one update, three games and a small sampler, enough
to exercise both stages in seconds. Each stage's files are in its own run:

```text
belief_sampler/policy-seed-197/out/policies/00000001-raw.pt
belief_sampler/sampler-seed-197/out/histories.json
belief_sampler/sampler-seed-197/out/belief.pt
```

A policy trained further is another run (`etude continue`), and a sampler for
it is another sampler run started from it; nothing made from the earlier
policy is overwritten or mistaken for the later one's.

`collect_belief` (`manabot.belief.collect`) plays `games` complete games with a
frozen policy and writes each viewer's history with its private hand label;
`train_belief` (`manabot.belief.train`) fits the sampler on them. Neither
changes the policy, and neither keeps state: an interrupted sampler run
collects and fits again when resumed. `load_checkpoint` checks that the start
policy belongs to this world and these model and observation configs.

Phases that share one run and each keep state of their own pass `name=` to
`run.resume` and `run.save`. No current recipe needs it.

## Self-play settings

`SelfPlay` binds the `ModelConfig`, `ObservationConfig`, `MatchConfig`
(including sideboards) and the world once; a world that differs from the
installed native runtime fails validation. `elemental_self_play()`,
`sampled_self_play()` and the other builders in `manabot.recipes.builders`
return the settings our recipes use. A variant is a training function that
changes a copy's fields (`settings.model_copy(update={...})`) in its body and
adds a config field only for what its runs vary.

`workers` accepts 1 (default), 2 or 4 CPU observation producers: persistent
threads own disjoint native stream ranges, a barrier completes all writes
before the single policy owner reads, and stream identities and seeds are
unchanged across partitions. Sampled rosters (a deck source or
`elemental-uniform`) draw each game's decks from that game's native seed, so
draws, exposure and batches agree for 1, 2 and 4 producers. `threads` controls
Torch threads independently. `sampled_self_play()` defaults to four producers,
a damping horizon fitted to the expected updates and `recover_capacity`. The
matched GPU comparison for a fixed matchup is in
[worker-scaling run019](../experiments/worker-scaling.md#persistent-production-producers-run019).
A collector opened directly with `open_collector` must be closed with `close_collector`. `bucketed_shapes`
selects [bucketed padding](observation-and-model.md#bucketed-padding)
and `live_tokens` selects
[live-token attention](observation-and-model.md#live-token-attention)
for the collection and learner policies; both are off by default.

The policy reads only the acting viewer's tensors; private labels are never
model inputs. Learner and behavior weights are raw; EMA is a separately named
evaluation output with its own clock. The selection rule is
`last-complete-raw`, never the best seed or best evaluation.

### Schedules and clocks

The target (`updates`, `--until updates=N`, `etude continue --until`) says where
training stops and nothing else. Rates are set by the learning record, so the
updates a run has already made are the same whatever target it is later given:

```python
# PPO: progress is update / 64.
Learning(horizon=64)
# Damped policy: the absolute iteration, here rescaled onto 3200 updates.
DampedPolicyLearning(gradient="damped_policy", iteration_schedule=IterationSchedule(steps=3200))
```

| Learning record | What its rates read | Past the horizon |
| --- | --- | --- |
| `DampedPolicyLearning` | The absolute iteration, or that iteration mapped onto `iteration_schedule.steps` when a schedule is declared ([step sizes](learning.md#4-update)) | The mapped iteration stays at its last value |
| `Learning` (PPO) | `update / horizon`, the fraction of the declared horizon completed | The rates stay at their final values |

`Learning.horizon` is required for self-play: `SelfPlay` refuses a PPO record
without one, naming the field, because there is no target-independent default.
A recipe sets it from its own update count
(`gray_ogre_self_play(horizon=config.updates)`), so the horizon follows a
config field and resuming under another is refused like any other changed
config; to train under another horizon, start a new run. Extending a run past
its horizon is allowed and trains on at the final rates. An unset `horizon` is omitted from serialization, so saved regimes that
hold a `Learning` record keep their bytes; their stages took progress from the
stage's own length and run only from the revision
[saved regimes](#saved-regimes) names.

Evaluation is outside training and does not consume the training clock. Saved
regimes with `schedule_clock="run_elapsed_budget"` divided elapsed wall time by
a budget; they run only from the revision [saved regimes](#saved-regimes) names.

### Learning from both seats

`SelfPlay.learning_seats` defaults to `"both"` for current-self and
EMA-self training; `"one"` is the ablation control. Random behavior requires an explicit one-seat setting. Frozen-opponent
regimes run from their recorded revision; the current loop has no frozen opponent. The default is a
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
separately. For two live models, open one collector with
`opponent_mode="learning"`, `learning_seats="both"` and `opponent_agent`;
`collect_pair(collector, agent, n)` returns each model's rows.

### Batch closure

`SelfPlay.batch_closure` says when a batch ends, independently of who
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

Saved regimes' stages never declared their closure, and reading one resolves
omission the way those runs behaved: one seat with neither `games_per_stream` nor
`game_errors` is exact quota, anything else complete games. Their digests are
unchanged. `SelfPlay` always states its closure. `AttackPlan` schema 1 retains
its exact-quota attackers; schema 2 uses the template's closure.

### Costs and counters

Each update logs collection and optimization time separately from
evaluation, with complete games, environment decisions, `learner_transitions`,
`optimizer_updates` (completed `optimizer.step()` calls) and
`optimizer_exposures` (sample uses across minibatches); `updates` is the
absolute completed update count. An empty-filter iteration is a completed
update with zero optimizer calls. Hours and dollars are Etude's: they span every
attempt of a run, including work repeated after an interruption. Reproducibility
relies on the config, seed, the revision that wrote each checkpoint (and its
Etude run under a launch), and Etude's attempt records; it does not promise
byte-identical stochastic training. A fresh run owns fresh costs; no pre-existing corpus is free.

### Compound decisions (removed)

The learned compound-decision policy (`compound_decisions` models and
`train_compound` stages) has been removed. Saved configurations, checkpoints
and regimes that select it fail validation rather than loading as flat
policies; reproduce them from source commit
`ba3cd58de25bcc5d0b32e0fa3c546ee3d27be52b`. Retained receipts and reports
that record compound fields remain readable.

### Selection diagnostics

`collect_selection(policy, out, population, match=...)` (`manabot.training.rollout.selection_data`)
freezes a checkpoint's raw or EMA policy and collects a declared population
with unique seeds and whole-game development/held-out splits; no weights
change. It returns at once when `out` binds the same policy, population, match,
reference and step cap; a different request refuses. Scalar and
categorical values are supported; belief-input policies are not. The protocol
is in
[ataraxos-omitted-controls](../experiments/ataraxos-omitted-controls.md#frozen-complete-game-selection-diagnostic);
results are frozen-policy associations, not causal filter benefits.

### Reuse a published policy

A stage that reads a policy takes a checkpoint path, so a published policy is
used by passing its file where a recipe would pass the one its run was started
from:

```python
histories = collect_belief(Path("inputs/producer-ema.pt"), run.out / "histories.json",
                           match=match, games=256, seed=run.seed)
```

`load_checkpoint` refuses a checkpoint from another world or with other model
or observation configs, and an EMA policy never loads as raw.

`SelfPlay.pretrained` (a checkpoint path) starts self-play from a published
policy instead of random weights. Only the policy is inherited: Adam, EMA,
streams and the schedule start fresh, so choose a schedule meant for a trained
policy. To continue a run with its Adam and EMA, continue the run
([recovery](#recovery)).

## Experiments

Declaring, executing and calibrating a comparison is in
[experiments.md](experiments.md#declaring-a-comparison).

## Remote training

Manabot trains on rented machines through
[Etude](https://github.com/loopflowstudio/etude), which rents the machine,
enforces hours and dollars, ships metrics and checkpoints during the run,
retries a run that loses its machine and deletes the machine at its deadline.
An experiment file keeps its explicit functions: `train(config)` is what a run
does, `runs()` declares what to launch, `eval(state)` says which matches its
checkpoints should play and `report()` draws them
([experiment files](../experiments/README.md#etude-experiment-files)). `runs()`
returns the jobs to launch. A job is the training function and its config;
each of its seeds is one run:

```python
# experiments/<name>.py
from decimal import Decimal

from etude.jobs import training
from etude.launch.records import Job

from manabot.experiment.setup import machine_setup
from manabot.recipes.elemental import ElementalConfig, train


def runs(revision: str) -> Job:
    return training(
        train,
        config=ElementalConfig(updates=3200, device="cuda", validate_numerics=True),
        smoke=ElementalConfig(updates=1, device="cpu", streams=2),
        seeds=(197, 198),
        revision=revision,
        setup=machine_setup(" --extra artifacts"),
        max_hours=12,
        max_usd=Decimal(9),
    )
```

One job with two seeds is two runs, named for the job and the seed. A name
already taken in the experiment is refused before anything is rented.

An experiment whose training differs from a maintained recipe defines its own
`train(config)` at the top of the file, with its own small config, and passes
that to `training`.

```sh
uv run etude smoke experiments/elemental.py --step train --revision COMMIT --updates 3200 --hours 11.9 --usd-per-run 9
uv run etude launch experiments/elemental.py --revision COMMIT --updates 3200 --hours 11.9 --usd-per-run 9 --cohort elemental-1 --rent l4
uv run etude status
uv run etude report experiments/elemental.py --source r2://manabot/jobs --open
```

Building a job is pure; nothing is submitted until `etude launch`. On the
machine Etude imports `train` from the committed source and calls it with the
config, validated back into its type. `track` inside it takes the experiment,
run, seed and attempt from the launch, and `run.out` is the run's directory,
which Etude carries to storage as the function writes it and gives back, whole,
to the next attempt. An attempt that raised `Stopped`, or lost its machine, is
called again while the run has hours, dollars and attempts left; any other
exception ends the run as failed. A run that ended unfinished is brought back
with `uv run etude launch experiments/elemental.py --resume RUN`, as another
attempt of the same run on the current code. Runs track under the experiment file's stem,
so that file's `eval(state)` and `report()` find them; every milestone
checkpoint is recorded with `run.checkpoint`, so Etude evaluation can follow
during training.

Setup installs Rust 1.98.1 with rustup and runs the locked `uv sync`, which
builds the native engine. An `ElementalConfig` with `validate_numerics=True` runs
the CUDA numerics gate (`manabot.experiment.numerics`) before its first update
and refuses to train if it fails. The committed dependency requirements and
`uv.lock` select the same Etude SDK release for both the launcher and this
setup. Refresh it only through the
[explicit dependency update workflow](development.md#environment-and-commands)
before freezing a new run; never upgrade dependencies inside a training attempt.

Pick the machine at launch (`--pool`, `--rent`); read progress with
`etude status`, `etude logs` and `etude usage`. To train an existing run
further, `etude continue --run RUN --until updates=6400` makes a child run that
starts from a copy of the parent's `run.out` with the same config;
`run.steps` counts to the new target, so the config does not change.
See Etude's
[launcher guide](https://github.com/loopflowstudio/etude/blob/main/docs/launcher.md).

## Monitoring

A run's Etude record keeps its config, seed, source revision, host, attempts,
costs, logged rows and recorded checkpoints; `state/` and the policies
under `run.out` are the learner. Checkpoint evaluation is a separate Etude
evaluation that reads those checkpoints. No account is needed to train.

The recipes declare `updates` and `training_seconds` clocks. Every update row,
outcome table and checkpoint carries both. `training_seconds` is the sum of each
completed update's `time/collection` and `time/learning`, and the saved `Counts`
keep it across attempts; startup, checkpoint writes, reporting and failed partial
updates are excluded. The belief sampler's run declares its own `steps` clock. The training report's default active-time clock
therefore works for loop runs. No transitions clock is declared; cumulative
learner transitions remain a metric.

### Update metrics

A run's rows are in its folder: under `~/.etude/track` for a local run and in
the launch's storage for a launched one. A resumed attempt continues the same
run, and later rows replace earlier ones from the resumed update. Draw it
with:

```bash
uv run etude report experiments/elemental.py --open                           # local runs
uv run etude report experiments/elemental.py --source r2://manabot/jobs --open
```

Rows use flat names: `loss`, `numerical/...`, `coordinates/learner_transitions`,
phase seconds under `time/` (`time/collection`,
`time/learning/minibatches/forward_and_loss`), CUDA high-water marks
(`memory/peak_allocated_cuda_bytes`, `memory/peak_reserved_cuda_bytes`),
MPS current tensor/driver allocations (`memory/allocated_mps_bytes`,
`memory/driver_allocated_mps_bytes`), process RSS (`coordinates/rss_bytes`),
one-minute host load (`coordinates/host_load_1m`) and
`selection_groups/<action type>/<terminal distance>/rows`. Device metrics are
absent on other devices. CUDA peaks cover the process lifetime; MPS allocations
are current observations, not peaks. Completed self-play
outcomes are a row table, logged once per update with Etude's
`run.log_table("self_play_outcomes", rows, updates=..., training_seconds=...)`. Each row has `deck`
and `opponent_deck` (deck identities), `position` (`play` or `draw`), `wins`,
`losses` and `draws`, so deck pairs add rows rather than metric columns. Etude
stores the table as an immutable JSON file under the metric run's `artifacts/` and carries it with the
metrics, including for launched jobs; keep `artifacts/` when copying a run for
offline analysis. Read the rows directly through Etude:

```python
from etude.train.read import read

# The runs directory holding the metric run, its experiment and the run's id.
history = read(runs_dir, experiment, run_id)
for table in history.read_table("self_play_outcomes", "updates"):
    print(table.x, table.rows)
```

```text
1.0 ({'deck': '7d663f0e…', 'opponent_deck': 'fc948195…', 'position': 'draw', 'wins': 1, 'losses': 1, 'draws': 0}, ...)
```

`manabot.training.update_metrics.read_updates` reads the same tables and
projects their counts as `self_play_outcomes/<deck>/<opponent>/<play|draw>/wins`
(and `losses`/`draws`) for existing reports. Etude selects resumed and duplicate
updates before fetching tables; unreferenced tables supply no results, and a
missing or damaged selected table raises, naming the file to restore. Historical
scalar outcome metrics remain readable unchanged. Runs that recorded
`self_play_outcomes/table` references (written from `1659b0e29`) are refused;
read them from a revision that still reads that format, such as `bf4e280f7`.
Selection and belief summaries are written beside the step's output.

The Perf panel's "Update time" series shows where an update goes:
`environment_step_and_encoding`, which compares a run to retained ones, divides by
the engine's own clock into `native_stepping` and `observation_encoding`, beside
preparation, inference, `stream_bookkeeping`, assembly and learning.
Etude's attempt record names the machine each attempt got. With one producer
(`workers` 1) training occupies one CPU thread, so host single-thread
speed sets its pace, not vCPU count or GPU size; see
[training update time](../experiments/training-update-time.md) before choosing a
machine. `scripts/bench_training_update.py` prints the same split for one machine.

Reading the curves: RL policy loss is the clipped objective, **not log loss**.
PPO value loss is half the squared lambda-target residual; categorical move
learning records its own. Loss, entropy and KL describe the last optimized
minibatch. RSS, device allocation and host load are resource observations, not
learning quality. Throughput is cumulative work over observed elapsed cost.

### Checkpoint cadence

`SelfPlay.checkpoint_every` (`n`) makes the loop publish a world-bound
policy at absolute updates n, 2n, … with `run.checkpoint`; the loop always
publishes the untrained policy at update 0 and its last update. A run's final
checkpoint is the last one of a finished run, including after extension.
Elemental uses a 50-update grid, chosen by the recipe rather than estimated
from wall time. The grid is absolute: a run continued from 26,000 with
`checkpoint_every=1000` publishes 27,000 next. A published policy carries raw
or EMA weights only; `state/` carries what resuming needs. A failed write
fails the update rather than leaving a silent gap. `run.checkpoints()` lists
what a run has published, oldest first, the same records evaluation reads.

### Checkpoint evaluation

Checkpoints are evaluated by Etude, in their own jobs whose compute is
counted apart from training. An experiment file's `eval(state)` returns the
benchmarks it wants for the checkpoints its runs have recorded, played by
manabot's `play` function (`manabot.eval.monitor.play`), and each pass plays
only the matches that are new:

```bash
uv run etude eval experiments/elemental.py --source r2://manabot/jobs
```

The default benchmark, `MonitorProtocol`
(`manabot.eval.monitor`), is 100 games: 25 reserved deals
(`1910101000`–`1910101024`), four seat/deck legs each, against source-pinned
scripted greedy. Exclude these deals from scientific cohorts. An experiment's `eval(state)`
may return a frozen custom protocol instead. Arena owns game execution, timeouts, legal
Commands and exact replay. Any failed or missing leg makes aggregate rates
unavailable; complete cohorts report 95% bootstrap intervals over whole
four-leg deals, which are conditional on the checkpoint and say nothing about
training-seed variation. Etude records each evaluation job's wall time, CPU,
RSS and cost.

Saved results can be published without new games:

```bash
uv run python -m manabot.reports.training --evaluations a/monitor.json b/monitor.json --out .runs/monitor-dashboard
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

Self-play retains masked normalized policy logs through
collection and learning, identified as `policy_numerics=masked-log-softmax-v1`
in the run's identities. Per-update health is in the Etude metrics
(`numerical/...`); the run's `run.out` holds terminal incidents, and the
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

### Reduced precision

A CUDA loop can trade exact float32 arithmetic for tensor-core speed in its
settings:

```python
settings = settings.model_copy(update={"device": "cuda", "precision": "tf32", "bucketed_shapes": True})
```

Retained regime builders resolve `Execution.precision="auto"` alongside
`bucketed_shapes` and `live_tokens`. It selects float32 everywhere after the
BF16 learning gate failed. Explicit choices and saved precision are preserved;
a saved stage predating the field reads as float32. Current `SelfPlay` settings
use explicit precision, defaulting to float32.

| `precision` | What changes |
| --- | --- |
| `float32` (default) | Nothing; the process's own TF32 switches are left as they were. |
| `tf32` | For the loop, CUDA matmul and cuDNN may use TF32 tensor cores (`torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`). Every tensor stays float32. |
| `bf16` | `torch.autocast(cuda, bfloat16)` around the policy forwards only: behavior sampling and stored values in collection, bootstrap targets and the loss forward in the learner. Forward outputs are cast back to float32, so behavior log-probabilities, targets, losses and their reductions are float32; weights and Adam state stay float32 and no gradient scaler is used. |

Checkpoint evaluation and search run on CPU in float32. A reduced precision off CUDA fails
validation (`precision tf32 requires CUDA`). The field is serialized in every
`SelfPlay`, so a run's recorded config names the precision it trained under,
and a reduced-precision run is a declared recipe variant, never an edit of a
float32 recipe. Precision changes no `ModelConfig` field, checkpoint or world
binding: a checkpoint trained in any precision loads and runs in any other.

The measured speed crossover is width 256 (`BF16_MIN_HIDDEN_DIM` in
`manabot.training.regime`). On an L4 with bucketed shapes, BF16 is slower than
float32 at w160l8, 1.66× faster per learner step at w256l8, and 2–2.7× faster
at w384l8 ([measurement](../experiments/cuda_precision.md)). But in the
[learning A/B](../experiments/precision_ladder.md), BF16 seed 32401 stopped
after 140 completed updates with `behavior_log_probability_mismatch`.
That terminal numerical failure rejects BF16 under the pre-registered rule;
the speed crossover therefore remains inactive. Playing strength was not
scored, and the failure does not establish a strength difference. The A/B
used bucketed shapes without live-token attention; their combination with
BF16 has no learning admission from this run. TF32 remains explicit too.

Before training a reduced precision on another GPU, run the
[CUDA numerics validation](#validation), which checks float32, and the bf16
dtype-join admission on that host:

```bash
MANABOT_REQUIRE_CUDA=1 uv run --extra dev pytest tests/model/test_semantic_card_precision.py -q
```

### Terminal incidents

A terminal numerical error ends the attempt with the invariant, attempted
update and update health in its error, and writes a private
`numerical-incident/<failure>/` directory under `run.out`: `incident.json` (identities, shapes,
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

There is one recovery contract: a run carries on from the last state it saved,
and Etude does the saving, the carrying and the calling again. The loop
registers its learner, Adam, EMA and counts with `run.resume` and calls
`run.save()` every update; it has no state file or resume branch of its own.

| The run | What carries it on |
| --- | --- |
| lost its machine, reached its deadline or was asked to stop, under a launch | Nothing to do: the launcher calls `train` again as another attempt, with the same `run.out`, until the run finishes or uses up its hours, dollars or attempts. |
| ended unfinished: it failed and the bug is fixed, ran out of hours, was cancelled, or was interrupted locally | `uv run etude run RECIPE --resume RUN` here, `uv run etude launch FILE --resume RUN` on machines: another attempt of the same run on the current code. |
| finished, and should go further | `uv run etude continue --run RUN --until updates=N`: a new run descended from it ([portable learner continuation](#portable-learner-continuation)). |

A new run and a resumed one are different commands; nothing is inferred from
a directory that happens to exist. `--resume` refuses a finished run, and
refuses a config or seed that differs from the run's, naming the fields.
Attempts, retries, deadlines, the stop reserve, the carriage of `run.out` and
the run record are Etude's. A retried attempt's first logged row sets where
its rows replace the earlier attempt's, so the record reads as one run.

| Kept across a resume | Started again |
| --- | --- |
| Raw weights, EMA, Adam moments, the completed update count and the other cumulative counts, the schedules that follow the update count, and every published checkpoint | Native game streams (seeded from the run seed and the update they start at), RNG state, and every update after the last save |

Work after the last save repeats, and Etude charges its hours and dollars to
the run. Etude writes state on a wall-time interval and always on the last
update before a planned stop, so a deadline stop repeats nothing; an update is
the unit, so nothing mid-collection is saved.

Etude compares the config and the seed between attempts and nothing else.
Keeping the code compatible is the recipe author's: an attempt may run a later
commit, which it records, but settings built in the recipe's body, the model's
shape and the world are not compared, and changing them within a run is
undefined. A changed model shape fails when the saved weights load; published
policies stay world-bound, so `load_checkpoint` refuses one from another world.

### What resume does not promise

- **Exact resume mid-collection.** Games in progress at an interruption are
  abandoned, not rebuilt from their action prefix. The executor's current-game
  recovery (`RecoveryPolicy`) and exact CPU process recovery
  (`recovery_max_microsteps`, `--resume-from`) are not available; the frozen
  regimes that declare them (`elemental-long-183-v1`, `elemental-two-seat-181-v1`,
  `elemental-uniform-174-v1`, `elemental-ur-gw-186-v1`,
  `tla-frontier-c5fba4-w64`) reproduce with that behavior only from the
  revision [saved regimes](#saved-regimes) names.
- **Proof that a resumed run equals an uninterrupted one.** A resumed run
  plays different games after the resume point than an uninterrupted run with
  the same seed would have. Each checkpoint names the revision that wrote it
  and, under `etude launch`, its Etude run. Etude's run record names the attempts and their resume
  coordinates. These records preserve provenance, not identical bytes.
- **A check that a fix is compatible.** After a bug fix, `--resume` runs the
  current code as another attempt of the same run; `--source-repair-reason`
  is gone. Each attempt records its commit, existing checkpoints stay as
  written and new ones name the revision that wrote them. Only the config and
  seed are compared, so whether the fixed code still fits the saved state is
  the author's judgment.

### Portable learner continuation

A run continued with `etude continue --run RUN --until updates=N` starts from a
copy of its parent's `run.out`, so from its saved state with Adam and EMA: it is
the same learner trained further. The child runs the parent's config and
seed; `run.steps` counts to the new target, and only that target and the
machine may change. Rates do not move
with the target ([schedules and clocks](#schedules-and-clocks)). A parent at 26,000
updates can produce a child ending at 63,000 and then one ending at 100,000;
damped-policy rates keep their absolute formulas (update 26,001 uses LR
`6.957944302072213e-6`, tau `0.0023684978179155515` under the default
schedule).

## Game failures

### Tolerating game failures in training

`SelfPlay.game_errors` (`GameErrorPolicy`, `manabot.training.state.game_errors`)
opts a loop into tolerating game failures:

```python
settings = settings.model_copy(update={"game_errors": GameErrorPolicy(
    threshold=0.1, window=100, warmup=100, max_game_actions=40000)})
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
are Etude attempts, outside that denominator. Each attempt appends to its own
ledger under `game-attempts/` and starts by counting every earlier one, so a
resumed run keeps counting from where it stopped, including games after the
last saved state. After
warmup, `rate > threshold` stops the run; the stop survives a resume. The example above tolerates 10 failures in the latest 100 settled
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

Each `game-attempts/NNNN.jsonl` is a private, append-only, fsynced ledger: setup, seed,
stream, seat, full action prefix including the rejected action, native
pre-command frame, error and identities. Resource-limit receipts also retain
the observation configuration so replay crosses the same encoding boundary.
Never publish the ledger as a replay. Replay a receipt by its one-based line:

```sh
uv run python -m manabot.training.state.game_errors RUN_DIR/out/game-attempts/0000.jsonl --attempt 1 --seconds 30
```

After a fix, resume the run or start a new one
([recovery](#recovery)). A resumed run keeps its ledgers, so resuming does not
clear a game-error threshold stop.

### Automatic capture and replay

Capture is independent of `game_errors`. Strict native calls retain private
evidence (`manabot.env.failures`) before propagating the original exception,
under the run's `run.out / "failures"` or `.runs/game-failures/`
elsewhere. Files are create-only, mode 0600 and fsynced; a storage failure is
attached to the original exception. Coverage includes strict self-play,
`sim.rollout` and vector environments, search players (flat-MC and rollout
pool roots), selection journals, and arena evaluation, which journals
each pending Command under `failures/` before execution (pass
`keep_game_logs=True` to keep journals of successful games).

```sh
uv run python -m manabot.training.state.game_errors RUN_DIR/out/failures/ID.jsonl --attempt 1 --seconds 30
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
