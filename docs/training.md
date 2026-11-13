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
- [Recipes with several steps](#recipes-with-several-steps)
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

This cutover requires the Etude run API in
[Etude PR #103](https://github.com/loopflowstudio/etude/pull/103). SDK 0.2.16
does not contain it; the release and dependency pin are pending before publication.

```bash
uv run manabot train                                         # local: a two-update smoke
uv run manabot train --recipe elemental --seed 197 --out .runs/elemental
```

The second command trains the current Elemental recipe (damped-policy
self-play on the ten Elemental decks, both seats learning) and leaves its run in
the output directory:

```text
.runs/elemental/
  state.pt              raw weights, EMA, Adam and the update count, rewritten atomically
  checkpoints/          world-bound raw and EMA policies, named by update and never rewritten
  game-attempts/        one append-only game-error ledger per attempt
  failures/             private evidence of any game failure, for replay
  etude/                the Etude run record and its logged rows
```

The policy a run trained is its last checkpoint: `checkpoints/00003200-ema.pt`
for a recipe that reached update 3200. Run the same command again to resume:
the training function reads `state.pt` and carries on from the update it
records. One that already reached its target does nothing and ends finished.
`--steps` is a deprecated, hidden alias for `--updates` and will be removed in
Manabot 0.3.0.

`--updates N` sets only the stop target. Unlike the old `--steps`, it leaves
the recipe's own update count, which fixes local and baseline rates and the
checkpoint grid, unchanged: `--recipe baseline --updates 32` runs the first
quarter of its 128-update schedule. Run a finished recipe again with a larger
`N` and it trains on from where it ended and writes a later checkpoint beside
the earlier ones, which stay as they were. Choose a new `--out` for a
new run; reusing one continues it, and a directory whose `state.pt` was
written for other settings, another seed or another world is refused before
training, naming what differs. Without `--out` a recipe uses
`.runs/<recipe>-<seed>`, so repeating the command resumes that run. The `local`
smoke is the exception: it writes a fresh timestamped directory under `.runs/`
each time, because it checks an installation and keeps nothing worth resuming.
A command stopped by a stop request or at its deadline saves its state, prints
the update it reached and exits 75, the status Etude gives a planned stop; run
it again to carry on.

`--recipe local` is the default: a two-update smoke on CPU with no evaluation,
for checking an installation. `--device cpu|mps|cuda` overrides the recipe's
device; without it a recipe uses CUDA when available and CPU otherwise.
`uv run manabot train --list` names every recipe the CLI can run: ours in
`manabot.recipes`, and each experiment module's recipe functions as
`<experiment>` for modules exposing `train(config)`. An explicitly named
`<experiment>:<function>` also accepts a function with one typed Pydantic config.
Frozen zero-argument regime builders are data readers, not executable recipes.

CPU is the default and recommended device on Apple Silicon. On a quiet Apple
M2, MPS was slower than CPU for collection at every measured batch size, only
tied 8-thread CPU for learning, and its driver memory grew past 20 GB within a
few learner steps; `torch.compile` and fixed minibatch sizes did not change
that. MPS is admitted on explicit request (`--device mps`) for float32
self-play and requires Torch 2.15 or newer: on an Apple M2, Torch 2.10–2.13
compute wrong MPS linear-layer weight gradients, which gradient clipping then
applies (see `manabot/training/mps.py`). The repository lock installs 2.15;
with an older Torch, `--device mps` stops before training with a message
naming both versions. An unavailable device fails explicitly. Reduced precision
remains CUDA-only. Belief inputs and frozen opponents are unsupported on MPS.
Evaluation and search stay on CPU. A separate [MLX backend](../manabot/backends/mlx.md)
of the policy and the damped update reads the same checkpoints, converted to
the shared container, for measurement on Apple Silicon; no recipe runs on it. This support makes no speed or strength claim.

### Saved regimes

Before recipes were functions, a run was a saved `TrainingRegime` executed by
`execute_regime`. Saved regime files, their `run.json` records and their
checkpoints stay readable: `manabot.training.records` reads every retained
record and `load_checkpoint` loads every world-bound checkpoint as before,
with serialized keys unchanged. They no longer execute. Their last executor revision is
`f09f5c2919b8b1a5cf6d72c4f4f4721dac68f653`. `manabot train
--regime FILE` and `--regime-name` refuse, naming the last revision that runs
them; check that revision out to reproduce a frozen experiment exactly.

## Write a recipe

A recipe is a training function: `train(config)`, an ordinary typed function
whose body shows what runs. It declares its run with Etude's `track`, reads
its run directory and target from the run, and calls a loop. Nothing wraps it;
the command line and a launched job both simply call it. The config holds only
what varies between runs of the recipe. Everything else is code in the
function. This is the Elemental recipe (abridged: its optional CUDA numerics
gate, `validate_numerics`, is left out):

```python
# manabot/recipes/elemental.py
from typing import Literal

from etude.train.records import Step, Time
from etude.train.run import track
from pydantic import BaseModel, Field

from manabot.recipes.builders import elemental_self_play
from manabot.training.loops.self_play import self_play


class ElementalConfig(BaseModel):
    seed: int = 197
    updates: int = Field(default=3200, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    streams: int = Field(default=256, ge=2)


def train(config: ElementalConfig) -> None:
    settings = elemental_self_play().model_copy(
        update={"device": config.device, "streams": config.streams}
    )
    with track(
        clocks=[
            Step("updates", "collection/update iterations"),
            Time("training_seconds", "active collection and learning seconds"),
        ],
        seed=config.seed,
        config=config.model_dump(mode="json"),
    ) as run:
        updates = int(run.until("updates", default=config.updates))
        self_play(settings, run.out, seed=config.seed, updates=updates, run=run)
```

The `updates` clock counts collection/update iterations: one batch collected
and learned from. It is not a count of `optimizer.step()` calls, which a run
logs separately ([costs and counters](#costs-and-counters)).

Etude supplies the parts that are true of any training run, and the function
asks for them where it uses them:

| From Etude | What the function gets |
| --- | --- |
| `track(...)` | The run: its record, seed, config and clocks. Leaving the block finishes the attempt, as `stopped` when the block raised `Stopped` and as failed on any other exception. |
| `run.out` | The run's directory, the same on every attempt. Under a launch Etude carries its files to storage as they are written and restores them before an attempt on another machine. |
| `run.until("updates", default=...)` | The absolute update to train to: the launch's target when it names one (a retry or `etude continue`), otherwise the recipe's own. |
| `run.stopping()` | True once a stop was requested or the attempt's deadline is nearer than the slowest update so far. |
| `run.log`, `run.log_table`, `run.checkpoint` | Metric rows, row tables and checkpoints in the run's record. |
| `Stopped` | Raised by a loop that saved its state short of its target. Under a launch the function is called again while the run has hours, dollars and attempts left. |

`track` names no experiment. Under `etude launch` the run belongs to the
experiment file that launched it, and `uv run manabot train --recipe elemental`
records it under the recipe's name. Naming one in the recipe would stop every
other experiment file from launching it: Etude refuses a run whose `track`
names a different experiment from its launch.

The command line calls the function the way a launch does, through Etude's
`local_launch`, which names the run directory and experiment for every `track`
in its block:

```python
from etude.train.run import local_launch

from manabot.recipes.elemental import ElementalConfig, train

with local_launch(out=".runs/elemental", experiment="elemental"):
    train(ElementalConfig(seed=197))
```

`self_play` (`manabot.training.loops.self_play`) is the loop every current
recipe uses. Read it as the template for your own; its body is the loop you
would write (shown without its game-error ledger, failure capture, initial checkpoint and outcome
table):

```python
def self_play(
    settings: SelfPlay, out: Path, *, seed: int, updates: int, run: Tracker | None = None
) -> Path:
    state = load_state(out / "state.pt", settings, seed) or new_state(settings, seed)
    if state.update >= updates:
        return record_checkpoint(out, state, run)  # covers a crash before recording
    base = state.counts  # earlier attempts' work; this collector's totals add to it
    rng = np.random.default_rng(seed + MINIBATCH_SEED + state.update)
    collector = self_play_collector(settings, seed=seed, start=state.update)
    seed_random(seed + state.update)
    try:
        while state.update < updates and not (run and run.stopping()):
            state, row = step(state, collector, rng, base=base)
            if run:
                run.log(row, updates=state.update)
            checkpoint_due = state.update < updates and is_checkpoint(settings, state.update)
            if checkpoint_due or state.update % settings.save_every == 0:
                save_state(out / "state.pt", state)
            if checkpoint_due:
                record_checkpoint(out, state, run)
    finally:
        close_collector(collector)
    save_state(out / "state.pt", state)
    if state.update < updates:
        raise Stopped
    return record_checkpoint(out, state, run)
```

It returns the policy it trained: the EMA checkpoint of its last update when
the learning record keeps an EMA, the raw one otherwise.

The pieces it uses:

| Piece | What it does |
| --- | --- |
| `SelfPlay` | The loop's settings: match and decks, model and observation configs, the update and its learning record, streams, transitions, learning seats, batch closure, game-error policy, workers, device, precision, `save_every` and `checkpoint_every`. A Pydantic model holding data only; everything the loop does with it is a function. A recipe builds one in its body (`elemental_self_play()`); its digest is saved in `state.pt`. |
| `new_state(settings, seed)` | Fresh weights from `seed`, EMA as a copy, a fresh Adam, update 0. |
| `load_state(path, settings, seed)` | The saved learner, or `None` when `path` does not exist. Refuses a state written for other settings, another seed or another world, naming what differs. `device`, `workers` and `threads` may change between attempts. The target `updates` is not part of the state, so a later target carries on; a schedule's horizon is a learning setting, so a changed horizon is refused. |
| `save_state(path, state)` | Writes raw weights, EMA, Adam, the update count and the source revision to a temporary file, then renames it over `path`. A crash leaves the previous state whole. |
| `save_checkpoint(directory, state)` | World-bound raw and EMA policies at this update (`NNNNNNNN-raw.pt`, `-ema.pt`), each carrying `checkpoint_source`; returns the EMA path when the learning record keeps an EMA and the raw path otherwise. A file is written once: when it already exists for this update the path is returned and nothing is rewritten, because Etude records a checkpoint inside `run.out` where it is and readers verify its digest. |
| `record_checkpoint(out, state, run)` | `save_checkpoint` into `out / "checkpoints"`, then `run.checkpoint(path, updates=)`. The loop adds no `final` tag: Etude's `TrainingRun.final` is the last checkpoint of a finished run. |
| `self_play_collector(settings, seed=, start=)` | A `Collector` (`open_collector` in `manabot.training.rollout.collection`) whose streams are seeded from the run seed and the update it starts at, so a resumed loop plays fresh, reproducible games. Sampling (`seed_random`) and minibatch order are reseeded the same way. With `start=0` these use the retired executor's seed offsets. A two-update CPU comparison against the retired executor after the backend merge produced bitwise-equal raw and EMA weights for PPO and damped policy (seed 197, two streams); this does not promise uninterrupted equivalence after resume. |
| `step(state, collector, rng, base=)` | One update: collect a batch with the behavior policy, `learn` from it, and return the next `LoopState` (update and counts replaced; the learner, Adam and EMA are updated in place) with its metric row. |
| `learn(state, batch, rng)` | The update the learning record selects: `update_damped_iteration` for `DampedPolicyLearning`, `update_iteration` for `Learning` ([training loops](training-loop.md#the-updates-we-maintain)), then the EMA. Rates come from the state alone: damped-policy schedules read the absolute iteration, PPO schedules read `state.update / horizon`, and neither reads the target ([schedules and clocks](#schedules-and-clocks)). |
| `Tracker` | What a loop asks of a tracker: `log(values, **clocks)`, `log_table(name, rows, **clocks)`, `checkpoint(path, **clocks)` and `stopping()`. Etude's `Run` is one; pass `None` to train with no tracking. |
| `Stopped` | Etude's planned-stop exception (`etude.stopping.Stopped`); the loop raises it after saving, short of its target. |

`run.stopping()` is asked once per update. The loop then saves its state and
raises `Stopped`, and the next call carries on from that update. There is no
separate resume path: the first attempt, a retry after a lost machine and a
run continued to a later target are all the same call.

Without Etude, call the loop yourself and keep what it returns:

```python
from pathlib import Path

from manabot.recipes.builders import elemental_self_play
from manabot.training.loops.self_play import self_play

policy = self_play(elemental_self_play(), Path(".runs/mine"), seed=3, updates=64)
```

## Recipes with several steps

A recipe with several steps passes file paths from one step to the next. Each
step writes its output atomically and returns at once when that output already
exists, so a recipe resumed after any failure skips the steps it finished. A
step's output is named for the input it was made from, so a policy trained
further is a new file and the steps after it run again rather than reusing
what an earlier policy produced. The belief recipe trains a policy, collects
hidden-hand histories with it and fits a sampler:

```python
# manabot/recipes/belief_sampler.py
class BeliefConfig(BaseModel):
    seed: int = 197
    updates: int = Field(default=1, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    games: int = Field(default=3, ge=3)
    max_steps: int = Field(default=1000, ge=1)
    sampler: SamplerConfig = SamplerConfig(
        steps=4, batch_size=8, hidden_size=16, evaluation_samples=4
    )


def train(config: BeliefConfig) -> None:
    settings = gray_ogre_self_play(horizon=config.updates).model_copy(
        update={"device": config.device}
    )
    with track(
        clocks=[
            Step("updates", "collection/update iterations"),
            Time("training_seconds", "active collection and learning seconds"),
        ],
        seed=config.seed,
        config=config.model_dump(mode="json"),
    ) as run:
        updates = int(run.until("updates", default=config.updates))
        policy = self_play(
            settings, run.out / "policy", seed=config.seed, updates=updates, run=run
        )  # policy/checkpoints/00000001-raw.pt
        histories = collect_belief(
            policy,
            run.out / f"histories-{policy.stem}.json",
            match=root_match(settings),
            games=config.games,
            seed=config.seed,
            max_steps=config.max_steps,
        )
        sampler = train_belief(
            histories, run.out / f"belief-{policy.stem}.pt", config.sampler, seed=config.seed
        )
        run.checkpoint(sampler, tags=("belief",), updates=updates)
```

Its defaults are a pilot's: one update, three games and a small sampler, enough
to exercise every step in seconds.

```bash
uv run manabot train --recipe belief --out .runs/belief
uv run manabot train --recipe belief --out .runs/belief --updates 2
```

```text
.runs/belief/
  policy/checkpoints/00000001-raw.pt   policy/checkpoints/00000002-raw.pt
  histories-00000001-raw.json          histories-00000002-raw.json
  belief-00000001-raw.pt               belief-00000002-raw.pt
```

The second command trains the same learner one update further and runs the
later steps again for the new policy; nothing made from the earlier policy is
overwritten or mistaken for the later one's. This recipe's policy learns with
PPO, whose rates follow a horizon the recipe states (`horizon=config.updates`),
so the first update is the same whether the run stops there or is extended.
The second update lies past that one-update horizon and uses the final rates
([schedules and clocks](#schedules-and-clocks)).

`collect_belief` (`manabot.belief.collect`) plays `games` complete games with a
frozen policy and writes each viewer's history with its private hand label;
`train_belief` (`manabot.belief.train`) fits the sampler on them. Neither
changes the policy. To start from a published policy rather than training one,
pass its checkpoint path in place of `policy`; `load_checkpoint` checks that it
belongs to this world and these model and observation configs.

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

The target (`updates`, `--updates`, `etude continue --until`) says where
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
(`gray_ogre_self_play(horizon=config.updates)`). The horizon is part of the
learner's settings in `state.pt`, so resuming with a different one is refused
like any other changed setting; to train under another horizon, start a new
run. Extending a run past its horizon is allowed and trains on at the final
rates. An unset `horizon` is omitted from serialization, so saved regimes that
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
means the config, seed, source revision and checkpoint provenance, not
byte-identical stochastic training. A fresh run owns fresh costs; no
pre-existing corpus is free.

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

A step that reads a policy takes a checkpoint path, so a published policy is
used by passing its file where a recipe would pass the one it trained:

```python
histories = collect_belief(Path("inputs/producer-ema.pt"), run.out / "histories.json",
                           match=match, games=256, seed=config.seed)
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
returns Etude training jobs that call the training function with a config:

```python
# experiments/<name>.py
from decimal import Decimal

from etude.jobs import training
from etude.launch.records import Job

from manabot.recipes.elemental import ElementalConfig, train


def runs() -> list[Job]:
    return [
        training(train, config=ElementalConfig(seed=seed, updates=3200),
                 smoke=ElementalConfig(seed=seed, updates=1, streams=2),
                 max_hours=12, max_usd=Decimal(9))
        for seed in (197, 198)
    ]
```

An experiment whose training differs from a maintained recipe defines its own
`train(config)` at the top of the file, with its own small config, and passes
that to `training`.

```sh
uv run etude smoke experiments/<name>.py     # the smoke config here, spending nothing
uv run etude launch experiments/<name>.py --revision COMMIT --cohort <name>-1 --rent l4
uv run etude status
uv run etude report experiments/<name>.py --source r2://manabot/jobs --open
```

Building a job is pure; nothing is submitted until `etude launch`. On the
machine Etude imports `train` from the committed source and calls it with the
config, validated back into its type. `track` inside it takes the experiment,
run, seed and attempt from the launch, and `run.out` is the run's directory,
which Etude carries to storage as the function writes it and gives back, whole,
to the next attempt. An attempt that raised `Stopped`, or lost its machine, is
called again while the run has hours, dollars and attempts left; any other
exception ends the run as failed. Runs track under the experiment file's stem,
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
`run.until` returns the new target, so the config does not change.
See Etude's
[launcher guide](https://github.com/loopflowstudio/etude/blob/main/docs/launcher.md).

## Monitoring

A run's Etude record keeps its config, seed, source revision, host, attempts,
costs, logged rows and recorded checkpoints; `state.pt` and the checkpoints
under `run.out` are the learner. Checkpoint evaluation is a separate Etude
evaluation that reads those checkpoints. No account is needed to train.

The recipes declare `updates` and `training_seconds` clocks. Every update row,
outcome table and checkpoint carries both. `training_seconds` is the sum of each
completed update's `time/collection` and `time/learning`, and `state.pt` keeps it
across attempts; startup, checkpoint writes, reporting and failed partial updates
are excluded. In the belief recipe it measures policy self-play, not history
collection or sampler fitting. The training report's default active-time clock
therefore works for loop runs. No transitions clock is declared; cumulative
learner transitions remain a metric.

### Update metrics

A local run keeps its Etude record inside its output directory, under
`etude/`; under `etude launch` it is the launcher's run in
`ETUDE_TRACK_DIR`. A resumed attempt continues the same Etude run, and later
rows replace earlier ones from the resumed update. Draw it with:

```bash
uv run etude report experiments/elemental.py --source .runs/elemental --open
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

`SelfPlay.checkpoint_every` (`n`) makes the loop write a world-bound
checkpoint at absolute updates 0, n, 2n, … and record it with
`run.checkpoint`; the loop always writes its last update. Etude derives
`TrainingRun.final` as the last checkpoint of a finished run, including after
extension. 1000 is the usual grid,
an authoring default rather than an hourly estimate. The grid is absolute, so a
run continued from 26,000 with `checkpoint_every=1000` first writes at 27,000. A milestone checkpoint carries raw and EMA weights only; `state.pt`
carries what resuming needs. A failed write fails the update rather than
leaving a silent gap.

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

Reduced precision changes learning numerics, not just speed, and finite
metrics do not show that it learns the same. CUDA training stays float32: the
L4 measurement for w160l8 and w384l8
([`experiments/cuda_precision.md`](../experiments/cuda_precision.md)) left TF32
unresolved and BF16 rejected under its pre-registered rule, though both stayed
within 0.6% of float32's loss at fixed weights and its 200-step replay could
not resolve learning differences that small. TF32 is 1.1× faster at w160l8 and
1.3–1.5× at w384l8; BF16 is slower at w160l8 and 2–2.7× faster at w384l8. A
reduced precision becomes a recommendation only after a real learning
comparison: same seeds, bucketed shapes, strength at matched updates. Before
training in one on another GPU or model size, run the
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

There is one recovery contract: a run resumes from the last `state.pt` its loop
saved. Running the same command again does this locally; under Etude, a run
that loses its machine, reaches its deadline or is asked to stop is called
again on another attempt with the same `run.out`, until it finishes or uses up
its hours, dollars or attempts. Attempts, retries, deadlines, the stop
reserve, the carriage of `run.out` and the run record are Etude's
([out, a stop signal and running again](https://github.com/loopflowstudio/etude/blob/main/docs/tracking.md#out-a-stop-signal-and-running-again));
manabot keeps only `state.pt` and the loop that reads it. A retried attempt's
first logged row sets where its rows replace the earlier attempt's, so the
record reads as one run.

| Kept across a resume | Started again |
| --- | --- |
| Raw weights, EMA, Adam moments, the absolute update count, the schedules that follow it, every finished step's output and every recorded checkpoint | Native game streams (seeded from the run seed and the update they start at), RNG state, and every update after the last save |

Work after the last save repeats, and Etude charges its hours and dollars to
the run. `save_every` (default every update) bounds that work; an update is
the unit, so nothing mid-collection is saved. A deadline stop saves at an
update boundary first, so a planned stop repeats nothing.

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
  the same seed would have. Its updates and checkpoints are as valid, and
  checkpoint provenance names every attempt, but they are not the same bytes.
- **Repair of a run's source during the run.** A run executes one source
  revision for all its attempts; `--source-repair-reason` is gone. Fix the
  code and start a new run, or continue the old one as a child.

### Portable learner continuation

A run continued with `etude continue --run RUN --until updates=N` starts from a
copy of its parent's `run.out`, so from its `state.pt` with Adam and EMA: it is
the same learner trained further. The child runs the parent's config and
seed; its target comes from `run.until`, and only that target and the machine
may change. A state written for other settings is refused. Rates do not move
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
last `state.pt`. After
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
uv run python -m manabot.training.state.game_errors .runs/example/game-attempts/0000.jsonl --attempt 1 --seconds 30
```

A run executes one source revision for all its attempts. After a fix, start a
new run from the fixed revision, or continue the stopped run as a child
(`etude continue`), which records the new revision beside its parent's.

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
