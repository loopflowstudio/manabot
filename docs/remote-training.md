# Remote training

Manabot trains on rented machines through [Etude](https://github.com/loopflowstudio/etude).
An experiment's planned runs become Etude training runs; Etude rents the machine,
enforces hours and dollars, ships metrics and checkpoints while the run goes,
and deletes the machine at its deadline.

```python
from decimal import Decimal
from pathlib import Path

from etude.launch.client import queue_cohort, wait

from manabot.experiment.config import Baseline, Case, Experiment
from manabot.experiment.execution import PlannedRun
from experiments.batched_baseline import batched_baseline

experiment = Experiment(
    "capacity",
    Baseline.capture("baseline", batched_baseline()),
    cases=(Case("control"),),
    jobs=(
        PlannedRun(case="capacity-control", seed=1, max_hours=6, max_usd=Decimal(5), milestones=(1000, 2000)),
        PlannedRun(case="capacity-control", seed=2, max_hours=6, max_usd=Decimal(5), milestones=(1000, 2000)),
    ),
)
runs = experiment.runs(repo=Path("."), revision="<commit>")
wait(*queue_cohort("capacity", runs, rent="l4", parallel=1, max_usd=Decimal(10)))
```

`Experiment.runs` is pure: it only builds the runs. Each one runs
`manabot.experiment.job` on the machine, which trains the case's regime to its
update target and records a checkpoint at every milestone, so Etude evaluation
jobs can follow them before the run ends. `validate_numerics=True` runs the CUDA
numerics gate (`manabot.experiment.numerics`) first and refuses to train if it
fails.

A job can also continue another run's learner. `TrainingJob.continuation`
(`LearnerContinuation`) names the parent's `run.json` and its published
raw/EMA/Adam exports as objects in the machine's Etude storage; they arrive as
job inputs, the worker binds them as the stage's `LearningStateImport`, and
training resumes at the parent's absolute update toward the job's absolute
target. Game streams and RNGs restart, and the seed must be the parent's.
A parent interrupted before its final exports continues only by declaration
(`from_monitoring`): from the raw weights of one of its monitoring exports,
with the EMA reset to them and a fresh Adam. A job stops at an update boundary
before its watchdog (`stop_reserve_seconds`), exports its learner state and
ends `paused`; Etude records that attempt as succeeded at the update it reached.
`etude continue` is different: it resumes from a policy checkpoint, which
cannot carry Adam or collector state, so the worker refuses it.

Pick the machine when launching (`--pool`, `--rent`), not in the experiment.
Read progress with `etude status`, `etude logs` and `etude usage`; continue a run
from one of its checkpoints with `etude continue`. See Etude's
[launcher guide](https://github.com/loopflowstudio/etude/blob/main/docs/launcher.md).
