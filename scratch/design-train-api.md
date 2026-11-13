# The Etude train API and Manabot's loops: design for EXP-56 and MTG-329

2026-11-13. Written from the day's exploration (`scratch/positioning.md` holds
every quoted remark in full; `scratch/explore.html` and `scratch/sketch.html`
are the pages). Jack Heart asked for it: "Be ambitious. Help us land in the
new world where we're set up for the most success in terms of adoption by Jace
and evolution into the Etude Cloud world. Don't worry about requiring a lot of
churn on existing code of Manabot or sticking with the design as currently
written or proposed. None of that matters." and "Really lean into all the
things I emphasize today and write a design for MTG-329 and EXP-56 that helps
us get there."

This document replaces the client-facing shape of EXP-56 (PR #103 at
`105abdd`) and the loop and recipe shape of MTG-329. The launcher machinery
PR #103 built (out carriage, planned-stop retries, attempt cutoffs,
continuation, digest-checked inputs) is kept and built on. Nothing here has
been implemented or run.

## Where this is going

Three layers, in Jack Heart's words: "Lab That Remembers is etude cloude";
"League is, I think, closer to the face of [the Etude SDK] but what we're
trying to build now is mostly focused on the train API, which is the building
block for the League API".

- **Etude Cloud** (later): questions, findings, agents assigned to experiments.
- **Etude SDK, the face**: the league. A checkpoint is a player; an evaluation
  is matches between players. Front line: "Which checkpoint is better?"
- **Etude SDK, now**: the train API. Its job is to let any loop run unattended
  and publish checkpoints the league can seat.

## Principles

Each is something Jack Heart said on 2026-11-13; the design is held to them.

1. **The loop is the author's.** "move as much execution management outside of
   Manabot and into Etude as possible so that Loop code can be really focused
   on algorithms and parameters". Etude is a library: "we're not behind a
   closed API... You can just call our code". No Trainer: "I don't want to do
   that... not in [Etude]".
2. **A call is API only when something else in Etude reads what it writes.**
   "writing this kind of code is relatively cheap... where it starts to add
   value is just when its a shared and interplays with other parts of hte
   system... the API (and the positioning) should be designed with that in
   mind". Anything else is a snippet in the docs.
3. **You own the files.** "it's saved to your machine, it's in normal-ass
   formats that you can read, and you own it. We do not own it... Using our
   framework to write your trainer does not lock you into us". Etude publishes
   "directories, checkpoints, and JSON files".
4. **Resuming is automatic.** "why would you not be able to resume a run. that
   should be something we are saying we handle automatically".
5. **One stage, one model is the heavy default; the rest generalizes.** "the
   heavy heavy default should be 1 stage 1 model, and the API should accept
   that but generalize where needed". Jace's kind of loop must stay possible:
   "I don't want to go so far as to rule out innovative learning patterns".
6. **Pydantic is the config protocol.** "i am fine with using pydantic as basic
   a dependency / native protocol to etude". The docs must say it plainly.
7. **Jobs have no seed; runs do.** "we also have names for run (is seeded) and
   job (is not seeded)".
8. **New and resumed are different commands.** Nothing is inferred from a
   directory that happens to exist. People may choose run ids.
9. **Compatibility between attempts is the author's.** "if you change the code
   within attempts at a run, you get undefined behavior. seems ifne".
10. **Written for agents as much as people.** "100% this". One file, typed
    refusals, spending caps, commands safe to repeat.
11. **An SDK experiment is a label.** "a formal definition of an experiment
    ... sits with Etude Cloud. I think an Etude SDK experiment is just kind of
    a nice name for a label".

## Names

From the ontology review (`scratch/ontology.md`). Jack Heart, 2026-11-13,
answering its questions: "1a -- i think RunAttempt is probably better than
Attempt though", "2a", "3a", "4a, or alternatively use cohort instead of
experiment as a more general word", "5a", "6b". All of the renaming is in
EXP-56, before the release.

The sentence the nouns make:

> An experiment is a label over jobs. A job with a seed is a run. A run
> executes as run attempts on machines. While it executes it writes rows on
> its clocks, keeps state so it can carry on, and publishes checkpoints. A
> checkpoint with code that plays it is a player. Players take seats in
> matches in an environment. A report shows rows and match scores.

| Thing | Name | Replaces |
| --- | --- | --- |
| what a loop holds in `train` | `Run` | `etude.train.run.Run` (kept) |
| a job and a seed to launch | `RunRequest`, never named by authors | `etude.launch.records.Run` |
| the stored run | `RunRecord`, one class | `RunRecord`, `RunDeclaration`, `RunFile`, `TrainingRun` |
| one execution of a run on a machine | `RunAttempt`, one class | `RunAttempt` (launcher) and `Attempt` (tracking) |
| a published model with its clocks | `CheckpointRecord`, plus one reference type | `CheckpointRecord`, `Checkpoint`, `CheckpointKey`, `CheckpointRef` |
| a checkpoint and the code that plays it, or code alone | player | `Model`, `Code` |
| how a run attempt ended | `outcome.json` | `state.json` at the run's root |
| what a run keeps to carry on | state: `out/state/state.json` | (new) |
| the tracking internals package | `etude.tracking` | `etude.run` |
| groups a user sees | experiment (the label) and job | also cohort and report "line" |

- Report "line" becomes job. Cohort stays a launch option for shared limits
  and is not taught as a concept.
- The two group words, each with one meaning (Jack Heart, 2026-11-13, on
  this pair: "fine for now"). **Experiment**: the label on runs, taken from
  the file; no limits, nothing formal. **Cohort**: a named envelope of limits
  (`parallel`, `max_usd`) that runs are admitted under. A later launch adding
  to an existing cohort is EXP-61, not this work.
- One class for a stored run and one for a run attempt means the launcher's
  database and the tracked folder write the same record types. EXP-56 decides
  where each field is written; the classes are single.
- Runs and evaluations already stored stay readable: readers accept the old
  file name (`state.json` as an outcome) and old field spellings. Nothing
  stored is rewritten. Renamed Python names keep an alias that warns and
  names its removal release.
- Not renamed here: `Record` and `Storage` (each names two classes) and
  `Session`. The review lists them; no answer was given.

## The train API

### What the author writes

A file with a `train` function and a config class:

```python
# train.py
class TrainConfig(BaseModel):       # yours; any fields; no seed
    lr: float = 3e-4
    updates: int = 1000

def train(config: TrainConfig) -> None:
    with track(clocks=[Step("updates", "optimizer updates")], config=config) as run:
        torch.manual_seed(run.seed)
        model, optimizer = build(config)
        run.resume(model=model, optimizer=optimizer)
        for update in run.steps("updates", to=config.updates):
            metrics = one_update(model, optimizer)
            run.log(metrics)
            run.save()
            if update % 100 == 0:
                run.checkpoint(policy=model)
        run.checkpoint(policy=model)
```

That is the whole client experience for the default case: seven Etude lines,
no state class, no stop check, no resume branch.

Requirements, stated in the docs: the config is a Pydantic model with no Etude
base class and no required field; it survives a trip through JSON, checked at
launch and refused there; `train` takes that one annotated parameter and sits
at the top level of an importable file.

`train(config)` is one attempt at the whole job. It returns (finished), raises
`Stopped` (saved on purpose; call again), or raises anything else (failed).

### The calls

| Call | Does | Read by | PR #103 |
| --- | --- | --- | --- |
| `track(clocks=, config=)` | opens the run; takes the config model | launcher, reports | built; takes a dict today |
| `run.seed` | this run's seed | declaration, reports | new (env var today) |
| `run.resume(**objects, extra=)` | registers what to keep; loads the last save into it; returns the clocks at that save | launcher carriage, continuation | new |
| `run.steps(clock, to=)` | counts from the resumed position to the target; raises `Stopped` on the pass after a stop is asked for | launcher retry, budgets | new |
| `run.save()` | writes the registered objects when due, and always before a stop | launcher carriage | new |
| `run.log(row)`, `run.log_table(name, rows)` | rows on the declared clocks; the stepped clock is filled in | reports, queries, live view | built; clock fill new |
| `run.checkpoint(**objects)` or `(path)` or `(export_fn)` | publishes an immutable player with its clocks | evaluation, other runs, reports | path form built; rest new |
| `run.checkpoints()` | this run's published checkpoints, oldest first | (same list evaluation uses) | new; `latest_checkpoint` built |
| `run.input(name)` | a file or checkpoint folder this run was given | launcher (digest-checked), lineage | built as `ETUDE_INPUTS_DIR`; accessor new |
| `run.out`, `run.stopping()`, `run.until()`, `Stopped` | the pieces `steps`/`save` are made of | launcher | built |

Details that matter:

- **`run.steps(clock, to=)`** starts at the position `run.resume` restored
  (zero for a fresh run) plus one. Its target is the launch's (`--until`,
  a continuation) or else `to`. It must raise `Stopped`, not end quietly:
  code after the loop runs only when the target was reached. Otherwise a
  phased recipe runs its next phase on a half-trained model.
- **`run.resume`** takes anything with `state_dict()` / `load_state_dict()`.
  `extra=` is an optional Pydantic model for the author's own numbers
  (Manabot's cumulative counts), loaded and returned. A `name=` argument
  separates the state of two phases in one run. For state not shaped like
  this, `run.resume()` returns the last saved folder and `run.save(fn)` hands
  `fn` an empty one.
- **`run.save()`** is called every pass. Etude writes on the last pass before
  a stop and otherwise on an interval (default by wall time, overridable with
  `every=` in steps). The write replaces the previous state whole or not at
  all.
- **`run.checkpoint(policy=model)`** writes named objects the same way state
  is written, so "a lot of the exports... look the same". One call is one
  checkpoint and one player, however many models it holds. A single path
  still works, for projects whose checkpoint file has its own format
  (Manabot's world-bound policy file).
- **Carrying on restores what was registered.** Random streams and anything
  in flight are restored only if they are in a registered object. The docs
  say so.
- **Etude checks config and seed between attempts and names the fields that
  differ.** It does not check model shape, file format or environment.
  Source may change between attempts of a run; each attempt records its
  commit.

### What is on disk

Published as a contract in the docs. Readable with `json` and the author's
framework, without Etude.

```text
<experiment>/<run>/
  run.json                       job, seed, config, clocks, parent and inputs
  ...rows and tables              as today
  out/
    state/                        replaced each save
      state.json                  {"at": {"updates": 400}, "seed": 0, "commit": "...", "extra": {...}, "files": {"model": "model.pt", ...}}
      model.pt                    the framework's own format (torch.save of a state_dict)
      optimizer.pt
    checkpoints/
      updates-00000100/           never changes
        checkpoint.json           {"at": {"updates": 100}, "files": {"policy": "policy.pt"}}
        policy.pt
```

A checkpoint folder is digested as a whole (a manifest of its files'
digests), since `CheckpointRecord` holds one digest today.

Writers are per framework, chosen by the object's type, imported only when
used: PyTorch in this work; MLX and JAX when Manabot's other backends need
them. Etude's core stays free of framework imports.

### Jobs, runs, names, and the commands

A **job** is the function and its config, with no seed. A **run** is a job
with a seed: the thing launched, named, resumed and reported. The seed is not
a config field.

Names (Jack Heart, 2026-11-13: "maybe Run and etude.run want ot change", then
"ok, RunRequest works"): `Run` is the object a loop holds inside `train`. What
a job file declares, a job and a seed to run it with, is a `RunRequest`
(today's `etude.launch.records.Run`). The internal package `etude.run`
(declarations, delivery, storage) is renamed so it is not a second `run`
module. It becomes `etude.tracking` (Jack Heart: "or etude.track?" then
"etude.tracking also fine iether way"), which keeps the package name apart
from the `track` function.

The author never has to name `RunRequest` (Jack Heart: "still in favor of
not needing to know its called a RunRquest"). A job file returns jobs, and
seeds are one of the ways a job is launched, beside its name, hours and
dollars; they are not part of what the job is.

```python
# job.py
def runs() -> Job:
    return training(train, config=TrainConfig(), smoke=TrainConfig(updates=1),
                    seeds=(0, 1, 2), max_hours=2, max_usd=Decimal(4))
```

One job with three seeds is three runs. Without `seeds`, one run with seed 0.
A run's id is the job's name and its seed (`baseline-seed-7`) unless given.
`RunRequest` remains the record underneath for code that builds launches by
hand.

| Intent | Command | Result |
| --- | --- | --- |
| a new run, here | `etude run train.py [--config JSON] [--seed N] [--name NAME] [--until clock=value] [--seconds N]` | always new; prints the id |
| a new run, on machines | `etude launch job.py [--rent ...]` | always new |
| the same run, after a lost machine or `Stopped` | none; the launcher calls again | a new attempt of the same run |
| the same run, after it ended unfinished | `etude run train.py --resume RUN`, `etude launch job.py --resume RUN` | a new attempt of the same run, on current code |
| further, or changed | `etude continue --run RUN --until ...` | a new run descended from the first (as built) |

- A run id may be chosen (by default the job's name and seed; `--name` locally; `RunRequest(id=...)` underneath).
  It is unique within its experiment. The authority is the run's folder:
  `run.json` is created create-only, so a second claimant fails on any
  computer. The launcher also refuses early, before renting.
- A taken name is refused, never resumed. This changes `declare`, which today
  resumes a re-used name.
- `--resume` refuses a finished run and a changed config or seed.
- `etude run` replaces `local_launch` in user code and Manabot's
  `train --out`. Exit codes: 0 finished, 75 stopped, other failed.
- New commands print JSON with `--json` from the start (EXP-59 covers the
  rest).
- Bug already assigned to EXP-56: a refused or interrupted attempt must not
  mark a finished run failed.

### Beyond one stage, one model

Each rung adds one thing and changes nothing below it. No "stage" noun.

| | Case | Adds | When |
| --- | --- | --- | --- |
| 1 | one stage, one model | nothing | this work |
| 2 | several models in a run (Jace) | more names in `run.resume(...)` and `run.checkpoint(...)` | this work |
| 3 | phases in one run (Manabot belief) | `name=` on resume and save | this work |
| 4 | stages as separate runs (AlphaGo: imitation, then self-play from it) | `training(..., start=<checkpoint of another run>)`, `run.input("start")` | this work, over the built `Input` |
| 5 | wanted runs computed from what exists: `runs(state)`; runs reading each other while training | | EXP-60 |

Rung 4's rule of thumb for the docs: split into two runs when stages want
different machines or budgets, when several runs start from one, or when the
earlier result is evaluated on its own.

### What evaluation must accept (the join to the league)

"Model" is the author's word for a network. Etude's evaluation noun for a
checkpoint and the code that plays it (today's `Model`) becomes **player**;
code that plays with no checkpoint (today's `Code`) is also a player. Jack
Heart, 2026-11-13, on this proposal: "works for me". Stored evaluation records
keep their spelling; the rename is of the names people write and read.

- A checkpoint that is a folder: the playing code is handed the folder and
  `checkpoint.json`, and loads the files it needs. One checkpoint can be
  seated by two different pieces of playing code.
- Relative seats keep resolving against declared clocks.
- `run.checkpoints()` and evaluation read the same records.

Not checked: how the playing code receives a checkpoint today beyond "one
file as an input". EXP-56 must read that and extend it.

### Set up for Etude Cloud without building it

Every fact the Lab will want is already a file in the author's bucket, which a
hosted reader can read without being in the write path (STU-14): runs with
their job, seed and config; attempts with commit, machine and cost; the edges
between runs (`continue`, `start=`); checkpoints with clocks; evaluations.
The SDK adds no experiment record: an experiment stays the label those files
share.

## Manabot (MTG-329)

Rebuilt against the API above. Frozen evidence, experiment IDs and receipts
are untouched. Run directories written by the current loop (`state.pt`) are
not loadable afterwards; the docs name the last revision that reads them. No
loading path for the old layout.

### The loop

`manabot/training/loops/self_play.py` becomes settings, the collector, `step`
and this:

```python
def self_play(settings: SelfPlay, run: Run, *, updates: int) -> Path:
    """Train to the run's update target and return the final policy's path."""
    with device_settings(settings):
        learner, ema = create(settings, seed=run.seed)
        counts = run.resume(learner=learner.agent, optimizer=learner.optimizer, ema=ema, extra=Counts())
        with self_play_games(settings, run) as collector:
            for update in run.steps("updates", to=updates):
                counts, row = step(settings, learner, ema, collector, counts)
                run.log(row, training_seconds=counts.training_seconds)
                run.log_table(OUTCOME_TABLE, collector.stats.self_play.drain(), training_seconds=counts.training_seconds)
                run.save()
                if is_checkpoint(settings, update):
                    policy = publish(run, learner, ema, counts)
        return publish(run, learner, ema, counts)
```

Removed from the file: the `Tracker` protocol and the untracked path;
`LoopState` as a saved thing; `load_state`, `save_state`,
`setting_differences`, `StateMismatch`, the `format` and `world` guards; the
three hand-written resume cases; the stop check and `raise Stopped`; the
`Stopped` subclass. `resource_metrics` stays until EXP-58 records machine
metrics, then goes. `publish` keeps Manabot's own world-bound policy file and
records it with `run.checkpoint(path)`.

Aim: the file reads top to bottom like a single-file loop, and
`loops/minimal.py` (PPO, about 90 lines, resumable) is the example the Etude
README can show.

### Recipes, experiments, command line

- A recipe's `train(config)` opens the run and calls the loop:
  `with track(clocks=CLOCKS, config=config) as run: self_play(settings, run, updates=config.updates)`.
- `seed` leaves every config; experiments pass `seeds=(197,)` to `training`.
- The belief recipe becomes two jobs by the rung 4 rule: the policy run, and a
  sampler run with `start=` the policy's final checkpoint. Its
  `if out.exists(): return out` steps go.
- `manabot train --out` is removed in favour of `etude run`; the guide says so.
- `manabot/recipes/proposed_etude.py` and any `local_launch` use are deleted.

### Checks

Local, tiny, no induced load: a fresh `etude run` of the Elemental recipe at
one or two updates; interrupt and `--resume` reading back as one run with
continuous rows; the learner parity already recorded for seed 197 re-run
against the new loop; a launched smoke under the launcher's cap. Tests for the
files changed only.

## Jace adoption

What Jace needs is exactly rungs 2 and 4: `run.resume` over five models and
six optimizers, one checkpoint holding all its models, `run.checkpoints()` for
past versions at the table, and `run.input` for its human-trained drafter. Its
`limited_iteration` is unchanged; its caller becomes a `run.steps` loop
(`scratch/sketch.html`, Part 2); its `tracking.py` is deleted. The same
checkpoint seats as a limited player and as a constructed player. Etude's
docs should carry a multi-model example of this shape so the path is visible
without reading Manabot.

## Order of work

1. **EXP-56**: the calls, the on-disk layout, PyTorch writer, `etude run`,
   `--resume`, names, config and seed checks, folder checkpoints through
   evaluation, `start=`; the Training page rewritten as in
   `scratch/sketch.html` Part 1; a tiny non-Magic example. Old spellings
   (`track(seed=)`, `local_launch`) warn and name their removal release.
2. **MTG-329**: rebuild on an isolated override of the unreleased SDK, then
   pin the release.
3. **EXP-12** (launcher database migration) starts as soon as EXP-56 is done.
4. After: README led by the league (`scratch/etude-readme-draft.md`), EXP-60.

## Acceptance for EXP-56

- The default example above runs with `etude run`, is interrupted, resumes
  with `--resume`, and reads back as one run.
- A two-model example saves, resumes and publishes one checkpoint that an
  evaluation seats.
- A taken name is refused. A changed config or seed is refused naming fields.
  Changed source resumes and the attempt records its commit.
- A run started from another run's checkpoint records that link.
- A stop between phases never lets the later phase run.
- A finished run is never marked failed by a later refused attempt.

## Not in this work

A Trainer or callback API; a W&B-shaped module; the live view (EXP-57);
machine metrics (EXP-58); JSON on existing commands (EXP-59); pool seats,
population reads and ratings (EXP-60); a spending limit across launches
(EXP-61); anything hosted.

## Open

Jack Heart has not settled these; the design takes the first option and the
change is cheap either way.

1. The name `track`: "tracking doesnt clearly say like 'save logs, and
   checkpoints, ...'". Kept; alternatives `record`, `open_run`.
2. `etude continue` with nothing changed but the target: a child run (built)
   or the same run extended.
3. Whether the docs show saving games in flight (Jace keeps unfinished games
   between batches) or say an attempt starts them fresh.
4. Which public, non-Magic example the docs and the front-page clip use.
5. How `run.save()` decides it is due when not stopping: wall time (taken) or
   a count.
