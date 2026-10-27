# Remote training

Manabot trains on rented machines through [Etude](https://github.com/loopflowstudio/etude).
An experiment's planned runs become Etude training runs; Etude rents the machine,
enforces hours and dollars, ships metrics and checkpoints while the run goes,
and deletes the machine at its deadline.

```python
# experiments/capacity.py: experiment "capacity"
from decimal import Decimal
from pathlib import Path

from etude.launch.records import Run

from experiments.batched_baseline import batched_baseline
from manabot.experiment.job import TrainingJob


def runs(revision: str) -> list[Run]:
    job = TrainingJob(
        regime=batched_baseline(),
        experiment=Path(__file__).stem,
        milestones=(32, 64, 96),
    ).job(repo=Path("."), revision=revision, max_hours=6, max_usd=Decimal(5))
    return job.runs(1, 2)
```

```sh
uv run etude launch experiments/capacity.py --revision COMMIT --rent l4 --max-usd 10
uv run etude report experiments/capacity.py --source r2://manabot/jobs --open
```

A job's runs track under its `experiment`, the stem of the file that declares
them, so the file's `eval(state)` and `report()` find them
([experiment files](../experiments/README.md#etude-experiment-files)). A
case-by-seed `Experiment` gives the same runs with
`experiment.runs(experiment="capacity", repo=Path("."), revision=...)`.

Building a job is pure: nothing is submitted until `etude launch`. Each one runs
`manabot.experiment.job` on the machine, which trains the case's regime to its
update target and records a checkpoint at every milestone, so Etude evaluation
jobs can follow them before the run ends. `validate_numerics=True` runs the CUDA
numerics gate (`manabot.experiment.numerics`) first and refuses to train if it
fails. The job's setup installs Rust 1.98.1 with rustup and runs the locked
`uv sync`, which builds the native engine; stock RunPod images supply uv but
no Rust.

A job can also continue another run's learner. `TrainingJob.continuation`
(`LearnerContinuation`) names the parent's `run.json` and its published
raw/EMA/Adam exports as objects in the machine's Etude storage; they arrive as
job inputs, the worker binds them as the stage's `LearningStateImport`, and
training resumes at the parent's absolute update toward the job's absolute
target. Game streams and RNGs restart, and the seed must be the parent's.
`etude continue` is different: it resumes from a policy checkpoint, which
cannot carry Adam or collector state, so the worker refuses it.

Pick the machine when launching (`--pool`, `--rent`), not in the experiment.
Read progress with `etude status`, `etude logs` and `etude usage`; continue a run
from one of its checkpoints with `etude continue`. See Etude's
[launcher guide](https://github.com/loopflowstudio/etude/blob/main/docs/launcher.md).
