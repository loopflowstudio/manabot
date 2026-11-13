<!-- Draft of Etude's README.md, 2026-11-13. Proposal; not applied to the Etude repository.
Evaluation and report names follow Etude main at v0.3.1 (Seating, relative seats, Crosstable).
The training section uses run.out, run.until, run.stopping and Stopped, which are in EXP-56 (PR #103), not yet released.
Unverified: that this exact file runs as written; no command was run against it. -->

# Etude

Train on machines you own or rent, and find out which checkpoint is better.

Etude is a Python library and command line for running reinforcement learning
experiments. You write the training loop. Etude runs it on whatever compute
you have, keeps what it records in your own bucket, and plays its checkpoints
against each other and against opponents you choose.

## Which checkpoint is better?

A number logged from inside training cannot answer that. A match can. In
Etude a checkpoint is a player, and an evaluation is a set of matches between
players under conditions you name.

```python
# ppo.py: one experiment, in one file, at one commit
from decimal import Decimal

from etude.eval.match import environment
from etude.eval.records import Benchmark, Code, Earlier, Inference, Seating
from etude.experiment import ExperimentState
from etude.jobs import evaluation, training
from etude.launch.records import Job
from etude.report import Crosstable, Curve, Matches, Report, Score

from game import play                      # your code: plays one match, returns rows
from train import TrainConfig, train       # your code: the training loop

world = environment("my-game", files=["game/"])
deals = tuple({"deal": d} for d in range(25))
policy = Inference(source=["players/net.py"])           # a checkpoint, run by your code
heuristic = Code(name="heuristic", source=["players/heuristic.py"])

progress = Benchmark(
    name="progress",
    environment=world,
    seatings=(
        # A fixed opponent: comparable for as long as the heuristic does not change.
        Seating(name="vs heuristic", players=(policy, heuristic), conditions=deals, games=8),
        # A relative opponent: each checkpoint against itself 400 updates earlier.
        Seating(
            name="vs earlier self",
            players=(policy, Earlier(clock="updates", by=400, inference=policy)),
            conditions=deals,
            games=8,
        ),
    ),
)


def runs() -> Job:
    return training(
        train,
        config=TrainConfig(),
        smoke=TrainConfig(updates=1),
        max_hours=2,
        max_usd=Decimal(4),
    )


def eval(state: ExperimentState) -> list[Job]:
    checkpoints = [c for run in state.runs for c in run.checkpoints]
    return [evaluation(progress, play=play, checkpoints=checkpoints, max_hours=1)]


def report() -> Report:
    return Report(
        score=Score(table="games", value="score", reference=0.5),
        clock="updates",
        evals=(
            Curve(title="Against the heuristic", matches=Matches(seating="vs heuristic")),
            Curve(title="Against itself, 400 updates ago", matches=Matches(seating="vs earlier self")),
            Crosstable(),
        ),
    )
```

```sh
etude smoke ppo.py            # one tiny update and one tiny match here, spending nothing
etude launch ppo.py --rent l4 # train, capped at the job's hours and dollars
etude eval ppo.py             # play the matches eval(state) asks for and nobody has played
etude report ppo.py --open    # the page report() describes
```

Run `etude eval` again after more training and it plays only what is new.
Add a seating and it is played for every checkpoint so far. Results are
tables in your bucket, queried with DuckDB.

## Your loop, on any machine

Etude owns no part of learning. It gives a loop the few things it needs to
run unattended:

```python
# train.py
from pydantic import BaseModel

from etude.train.records import Step
from etude.train.run import Stopped, track


class TrainConfig(BaseModel):
    lr: float = 3e-4
    updates: int = 1000


def train(config: TrainConfig) -> None:
    with track(clocks=[Step("updates", "optimizer updates")]) as run:
        state = load(run.out) or new(config)                 # where the last attempt left it
        target = run.until("updates", default=config.updates)
        while state.update < target and not run.stopping():  # a deadline or a request to stop
            state, metrics = one_update(state)
            run.log(metrics, updates=state.update)
            save(run.out, state)
            if state.update % 100 == 0:
                run.checkpoint(save_policy(state), updates=state.update)
        if state.update < target:
            raise Stopped("saved; call again to continue")
```

- **A run survives its machine.** When a rented machine is taken away, Etude
  restores `run.out` on another and calls `train` again. It reads back as one
  run.
- **A run has a budget.** Hours and dollars are limits the launcher enforces
  across every attempt.
- **Clocks are declared.** A run states its x-axes once, each with a line of
  meaning, and every row gives all of them. There is no single step counter
  to fight.
- **Compute is whatever you have.** This computer, hosts you reach over SSH,
  and machines rented by the hour, behind one command.

## What you keep

Everything a run or a match records goes to your own directory or S3 bucket,
as files. There is no server to run and no account to make. An experiment's
page is its `report()` function at a commit, so everyone who opens it sees the
same charts in the same order.

## What Etude is not

It is not a trainer: bring your own loop and model. It sells no compute. It
has no hyperparameter search service, no saved dashboards to rearrange, and no
model serving.

## Install

Etude needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv tool install etude-sdk          # the `etude` command
uv add etude-sdk                   # the library, in a training project
uv add 'etude-sdk[eval]'           # with DuckDB, for evaluation tables
uv add 'etude-sdk[s3]'             # with S3 or R2 storage
```

## Read more

- [Evaluation](docs/evaluation.md): players, seatings, relative seats, benchmarks and round robins.
- [Tracking](docs/tracking.md): runs, clocks, row tables, checkpoints, reports.
- [The launcher](docs/launcher.md): typed jobs, pools, SSH hosts, rented machines, limits and recovery.

## Develop

(unchanged)

## Licence

MIT. See [LICENSE](LICENSE).
