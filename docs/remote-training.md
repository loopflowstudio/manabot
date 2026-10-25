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
fails. The job's setup installs Rust 1.98.1 with rustup and runs the locked
`uv sync`, which builds the native engine; stock RunPod images supply uv but
no Rust.

Pick the machine when launching (`--pool`, `--rent`), not in the experiment.
Read progress with `etude status`, `etude logs` and `etude usage`; continue a run
from one of its checkpoints with `etude continue`. See Etude's
[launcher guide](https://github.com/loopflowstudio/etude/blob/main/docs/launcher.md).
