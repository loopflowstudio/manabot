# Experiments

An experiment here is a module, the runs it launched and one report that stays
on the record whichever way it goes. Most are exploratory. One that will be
cited as a finding also commits its question, method and cost cap to git
before it runs, and decides what the result means after reading it. This
guide covers how a question becomes a report, what makes a result count, how
a result becomes a decision, and the tools that declare and execute a
comparison. The
[research program](../manabot/RESEARCH.md) says which questions are open; the
[experiments index](../experiments/README.md) lists every report.

- [What an experiment is](#what-an-experiment-is)
- [How a claim gets made](#how-a-claim-gets-made)
- [From a result to a decision](#from-a-result-to-a-decision)
- [A worked example](#a-worked-example)
- [Failures worth reading](#failures-worth-reading)
- [Declaring a comparison](#declaring-a-comparison)
- [Writing a report](#writing-a-report)
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
uv run etude launch experiments/elemental.py --revision COMMIT --updates 3200 \
    --hours 11.9 --usd-per-run 9 --rent l4
uv run etude eval experiments/elemental.py --source r2://manabot/jobs     # play what is new
uv run etude report experiments/elemental.py --source r2://manabot/jobs --serve  # draw what it played
```

`runs()` names the training runs, `eval(state)` returns the seatings still
wanted so repeated passes play only new ones, and `report()` draws the figures.
A seating is Etude's rule for who meets whom (`etude.eval.records.Seating`):
its players in seat order, the conditions they play under and the tags a
report selects them by. A `Benchmark` holds `seatings`, and a checkpoint says
where it stood on each clock in `Checkpoint.at`, as in
`checkpoint.at["updates"]`.

These commands need etude-sdk 0.3.2 or later, which the lock pins. Launch,
status and eval commands keep their state in a launcher database named for its
schema version: etude-sdk 0.3.2 uses `~/.etude/launcher-17.db`. A database an
older client wrote, such as the unversioned `~/.etude/launcher.db` of 0.2.x,
stays where it is and is never opened, so both clients work from one home:

```sh
uv run etude status                                        # launcher-17.db
uv run --with "etude-sdk[s3,eval]==0.2.15" etude status    # launcher.db
```

There is no migration between them. A new database starts empty: `etude
status` and `etude runs` list nothing, and launching needs its providers,
storages and on-demand machines added again. A run stays with the client that
launched it, so follow or stop it with that version, and keep the older file
while its runs are still referenced, since it is the only record of how they
ended. `etude report` and `etude launch --quote` rent nothing; a report reads
the job store and needs no launcher database. To leave `~/.etude` untouched
altogether, give a command its own home:
`ETUDE_HOME=$(mktemp -d) uv run etude ...`.
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
- **Exploratory is the default.** Calibration, throughput, capacity and pilot
  runs, and anything run to make later work possible, need no prediction. Their
  reports say what was learned, under the word "Exploratory", and their numbers
  are not cited as findings.
- **A finding asks first.** A result that will be cited as a claim about a
  method or about playing strength (in the README's "What we know", the
  research program or the paper) opens its report with one paragraph committed
  before any scored run: the question, what will be measured and against
  what, the strongest confound and a cost cap. An expectation of the outcome
  may be written there too, as an expectation. What that paragraph never
  holds is a verdict: no number chosen before the data decides whether the
  result is a win. A threshold set in advance is a guess dressed as a rule,
  and it has turned clear results into "rejected" and "unresolved". The
  decision is made after reading the result
  ([from a result to a decision](#from-a-result-to-a-decision)).
- **The freeze is the commit and the run ID.** A change that keeps the question
  is a dated line in the report, not a new protocol. Cost is the launcher's
  number for the run.
- **Seeds are the unit of evidence for a method.** Many games from one
  training run don't replace independent training runs. Three seeds are three
  data points.
- **Games are paired.** Candidates play the same deals, with both decks and
  both seats, which removes a large source of variance
  ([evaluation](evaluation.md#one-deal)).
- **Results belong to a world.** Numbers don't carry across
  [world versions](../WORLDS.md).
- **Refutations stay on the record,** dated and never silently edited.

## From a result to a decision

A report does not trigger a decision; a person makes one with the report in
hand. The report's job is to make that easy:

- **Lead with the plain answer:** the effect, its uncertainty and what it
  means for the question.
- **Compare against the incumbent, not against a bar.** What we run today is
  the default. A new backend, layout or technique has to justify its cost
  (code to keep, complexity, dollars) with an advantage where the work spends
  its time. Say how large the advantage is and where it appears and
  disappears: batch size, width, device, phase.
- **Separate what the numbers support from what they do not:** one host, a
  replayed update, a noise floor measured from a repeat run, how many seeds.
- **Recommend, with the reasoning, as a recommendation,** and name what would
  change it. Adopt, keep as a switch, drop and measure more are all
  legitimate outcomes, and inconclusive is a result.

What stays fixed after the run is the evidence, not the reading of it. Source,
decks, seeds, bounds and retained data are never changed to fit a conclusion,
and refutations stay on the record, dated.

## A worked example

[exp-04](../experiments/exp-04-potential-shaping.md) is from an earlier world,
so read it for its shape rather than its numbers. The project believed that
shaped rewards (pay per land, creature and damage) sped up learning, and an
earlier run had shown those rewards training agents into cast-everything play.
The question was whether a policy-invariant form of shaping would learn without
that fingerprint. The expectation going in was written down:

> E2c reaches ≥90% of E2b's win-vs-random while holding instants
> significantly longer. […]

The prediction held, and an addendum overturned the premise anyway: a
terminal-only control matched the potential-shaped arm and beat pay-per-event
shaping by about 14 points, so the shaping the project had been paying for was
buying nothing. Reward in every recipe since has been win, loss or draw and
nothing else. The report records the question, the prediction, the method, the
results, what they supported and the caveats, in that order, which is
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

An experiment file keeps its explicit `train(config)`, `runs()`, `eval(state)`
and `report()` functions. Training is an ordinary function that calls a loop;
Etude supplies its run directory, seed, saved state, stopping, tracking and retry. `runs()`
returns jobs made with `etude.jobs.training(train, config=..., smoke=...)`.
Put each arm in that declaration with its hours and dollar cap, and its
training seeds in `seeds=`; a config has no seed. Evaluation and reports continue to read the recorded checkpoints.

For example, `experiments/elemental.py` launches the current Elemental recipe
and evaluates its checkpoints on the existing paired-deal grid. A local run
calls the same function:

```bash
uv run etude run manabot/recipes/elemental.py --seed 197
uv run etude run manabot/recipes/elemental.py --resume elemental-seed-197
```

The first command is always a new run, named for its job and seed; the second
is another attempt of that run from its last saved state. `--resume RUN
--until updates=N` takes the same run to a later target, and `etude continue`
starts a new run from its state; neither changes the recipe's settings. See [training](training.md#recovery) for the guarantees and limits.

The nightly baseline runs `manabot/recipes/baseline.py` with `etude run` and
then calls `evaluate(config, directory)` on that run's `out`. Evaluation compares the initial and final raw
checkpoints on the same four deck/seat legs per deal, runs the existing native
choice inspection, and writes `learning.json` for aggregation across training
seeds. Training time and evaluation time remain separate.

### Retained comparisons

`manabot.experiment.config.Experiment` still resolves saved comparison
snapshots into `TrainingRegime` values and provenance receipts. The record
models and reports still read their original fields. Resolution does not
launch work, and those declarations no longer have a `runs()` method.
The local comparison scheduler, checkpoint follower and study executor are
retired. Frozen entry points and the CLI name revision
`f4e1822faf46bbc9cdc43909293f8cbc8de4a300` when asked to execute them.
Use their recorded revision and lock to reproduce old experiments; do not
translate their schedules or recovery contracts silently.

`StepCalibration.from_export(path)` remains a reader of a retained completed
single-stage export. Its `updates_for(hours)` gives an author a target; a new
recipe fixes its learning horizon before launch and leaves the stopping target
to Etude. Equal updates do not imply equal transitions, exposures or dollars.

## Writing a report

An experiment's `report()` returns an Etude `Report`. Build it from
`manabot.training.report`, which knows how manabot jobs are named and which
comparisons a training study asks, and add only what is specific to the
experiment. The frontier study's report is four questions and a schedule:

```python
from etude.report import Report, Score
from manabot.training.panels import ML, PERF
from manabot.training.report import (
    against_earlier_self,
    against_fixed,
    head_to_head,
    job_labels,
    learning_schedule,
)


def report() -> Report:
    regimes, _ = frozen()
    jobs = {_regime_id(label): regimes[label] for label in SIZES}
    small, large = jobs
    return Report(
        score=Score(table="games", value="score", reference=0.5, unit="games"),
        clock="updates",
        labels=job_labels({small: ("Small", jobs[small]), large: ("Large", jobs[large])}),
        evals=(
            against_earlier_self(
                "Model vs itself at half training time",
                tags=tuple(f"half-training-{label}" for label in SIZES),
            ),
            against_fixed(
                "Model vs heuristic",
                tags=tuple(f"heuristic-{label}" for label in SIZES),
            ),
            head_to_head("Large vs Small at fixed training time", EQUAL_HOURS_SEATINGS),
            head_to_head("Large vs Small at fixed # updates", EQUAL_UPDATE_SEATINGS),
        ),
        perf=PERF,
        ml=(*learning_schedule(jobs), *ML),
    )
```

For an experiment whose evaluations were played from the same revision,
`uv run etude report <experiment file> --source r2://manabot/jobs --serve`
draws it. A finished experiment's page is rebuilt from the revision that ran
it, or from a file with its `report()` and no `eval` over its own run prefixes:
`etude report` still calls `eval(state)`, and a seating derived from today's
source digest no longer agrees with the held runs. The page then draws the
tournament as its eval runs recorded it and says so in a note at the top
(`tournament_conflict`).

```sh
doppler run -p manabot -c prd -- \
  uv run etude report experiments/tla_frontier.py --source r2://manabot/jobs --out /tmp/tla-frontier
```

writes the frontier page from its 1,120 stored matches on any machine with
access to the bucket. Allow about half an hour: `r2://manabot/jobs` holds
several thousand runs, and a build reads each one's record. A report over a
local store takes seconds.

Over the frontier's retained evaluations, the page shows four charts under
Evals, scored from 0 to 1 with 0.5 drawn as even. The charts against the
model's earlier self and against the heuristic have a line for Small (width
64) and one for Large (width 160, 8 layers). The two Large vs Small charts, at
a fixed number of updates and at fixed training time, each show one line,
Large's score against Small at each of Large's checkpoints: above 0.5, Large is
stronger. Small has no line of its own there, since it would be the same games
read from the other seat. Fixed updates has ten points, from update 25 to 800,
and fixed training time eleven, ending at Large's final checkpoint. Hovering a
point reads like `0.66 · 95% interval 0.55–0.77 · 64 games · 16 matches`.
The learning schedule opens ML details.

| Helper | Draws |
| --- | --- |
| `job_labels(jobs)` | A display name per job: the size word you give, then the model's width, and its attention layers when they differ from the first job's. Small (width 64), Large (width 160, 8 layers). Etude appends each job's seed count. |
| `against_earlier_self(title, tags)` | Each checkpoint against an earlier checkpoint of its own job, a line per job. A point counts every such match its checkpoint sat in; a checkpoint that was also the earlier partner of a later one counts that role too, until Etude can score only the later side. |
| `against_fixed(title, tags)` | Each checkpoint against one fixed player, such as the heuristic, a line per job. The tags select the seatings that hold that player. |
| `head_to_head(title, seatings)` | Two jobs against each other, as one line: the first-seated job's score, drawn where its checkpoint stood. `seatings` are the names `eval(state)` gave the pairs, one per point, each played once. An empty sequence raises `ValueError`. |
| `learning_schedule(regimes)` | Two charts, the configured learning rate and policy damping of each job on the update clock through its damping horizon, with each run's latest update marked. These are plans, not measurements. |

Every helper returns an ordinary Etude panel, so a report mixes them with its
own `Curve`, `Matrix` or `Series` panels, and an experiment that needs a
different title or tag passes it rather than writing a new helper.
Equal training time is not yet an Etude clock: a fixed-time head-to-head is
drawn at the first-seated job's update, and the report's text gives each
pair's hours.

`head_to_head` selects by seating name because Etude draws a selection from
one seat only when every part of it is `Matches(seating=NAME)`. A tag is
always drawn per job, both seats, so a tagged head-to-head shows two lines
mirrored about 0.5. The names must be the recorded ones, character for
character, so the frontier formats them with the functions its `eval(state)`
schedule uses.

`report()` receives no state, so it cannot work out which pairs were played.
A finished experiment lists them in its file as a frozen record
(`EQUAL_UPDATE_SEATINGS` covers the ten updates both jobs reached); listing
every planned milestone instead would name pairs that were never played. A
name nothing holds does not fail the build. It is a note at the top of the page,

```text
A Curve panel names seating 'w160l8 u1200 vs w64 u1200', which nothing holds.
```

and the chart is missing that point: correct the name, or drop it if the pair
was never played. Renaming a seating in `eval(state)` changes its tournament
definition and the name reports select; the match keys still depend on the
players, groups, conditions and game count. For a finished experiment, correct
the report selection and preserve the recorded seating names.

A seating can also follow a job without listing pairs: an `Inference` first
seat for each held checkpoint to fill and a relative second seat, such as
`Aligned(job=OTHER, clock="updates", inference=...)` for the other job at the
same update. One name then covers every checkpoint. `Earlier` and `Fraction`
seat a checkpoint's own earlier self the same way, and every relative seat
names the `Inference` it plays as.

`Crosstable(matches=Matches(tournament=NAME))` shows each checkpoint of one
line against each other checkpoint of it, a tab per line. Etude has no panel
for direct games between two jobs other than these.

How a game was run is provenance, like its seed: an evaluation pass may play a
requested game on any available setup, and each result records its device,
revision and seed. A chart is meant to pool every game of its comparison
whichever tournament or machine played it, the way a job pools its seeds.
Etude does not pool setups yet: a tag names the matches of one tournament, and
one comparison played in several environments shows the environment whose
latest match is newest, with a note at the top of the page. Because each
chart names its own matches, the page shows no Matches control.

A tag no seating carries is noted at the top of the page, and a chart drawing
nothing says so; check those notes before reading the charts.

## Reading a report

Reports select over Etude jobs and average across seeds, and their metric
definitions are in [experiment-metrics.md](experiment-metrics.md). When reading
a training curve, keep the three kinds of progress apart: `rl/loss`,
`rl/collection_kl` and `rl/entropy` in the report (`loss`, `collection_kl` and
`entropy` in the raw Etude rows) describe optimization; monitoring scores
against the latest opponent describe relative progress; only held-out
evaluation on paired deals describes durable progress
([reading a run](learning.md#reading-a-run)).
