# Run an Experiment comparison

`manabot.training.experiments.Experiment` is the readable declaration of a
comparison. Each real experiment lives in its own Python file with a thin entry
point. `resolve()` remains side-effect free; `run_experiment()` explicitly executes
its schedule on configured hardware. `ExperimentRun` records what actually ran,
separately from the declaration. VerifyStore owns both ExperimentRun and
TrainingRun persistence; JSON exports and optional trackers are projections.

The primary report is **one editable `comparison.ipynb` per execution directory**.
It loads every regime's learning metrics, milestone win/draw/score rates and
uncertainty, compatible cross-regime comparisons, resource diagnostics, failures
and cost. Data discovery and plotting are ordinary editable cells. Run All reads
retained files only: it never starts training or evaluation and needs no tracker.
Report refresh creates the notebook only when absent, preserving existing edits
and outputs. To adopt a newer template, explicitly choose another filename.

## Declare, configure and launch

A declaration adds an `ExperimentSchedule` to the existing components/cases API:

```python
from pathlib import Path
from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.experiment_execution import ExperimentSchedule, HardwareInventory
from manabot.training.experiment_runner import run_experiment
from manabot.training.experiments import Case, Experiment
from manabot.training.presets import ataraxos_mtg_v1

experiment = Experiment(
    name="my-comparison",
    baseline=ataraxos_mtg_v1(),
    cases=(Case("control"),),
    schedule=ExperimentSchedule(
        seeds=(117, 118),
        hardware="authorized-local-cpu",
        wall_seconds=900,
        process_seconds=900,
        monitoring=MonitoringBudget(seconds=360, attempt_seconds=180),
        checkpoint_seconds=3600,
    ),
)
# Reading configured resources does not provision hardware or authorize science.
hardware = HardwareInventory.model_validate_json(Path("hardware.json").read_text())
result = run_experiment(experiment, hardware, Path(".runs/my-comparison"))
print(result.status, result.notebook)
```

The configuration file explicitly lists resource names, exact hostnames,
`backend: "local-cpu"`, and available `cpu_threads`. An optional
`dollars_per_hour` is the declared occupied-host price. The implementation admits
only this host and CPU execution; foreign hosts, unconfigured names and unsupported
backends fail before launching. There is no SSH discovery, cloud provisioning,
arbitrary device fallback or inferred spend authorization. Reserve at least one
thread for evaluation in addition to the learner's declared threads. The supported
placement executes one learner and at most one evaluator concurrently; other
regime/seed jobs wait. Account-level host/evaluator leases span worktrees.
Separately operated accounts require resource coordination.

`wall_seconds` bounds the execution including setup and reporting;
`process_seconds` bounds additive learner and evaluator process time, so overlap
is not free. The full monitoring allocation is reserved before admitting each
learner. An attempt must fit its declared ceiling before launch. Resource
admission may leave jobs pending rather than shrink their scientific counts.
Host dollars use recorded occupied wall time, not summed overlapping process
seconds. Without pricing, dollars remain unavailable. This does not measure
native peak memory or correct throughput for contention.

## History and depth are declarations

The first real consumers are
[`experiment_history.py`](../experiments/runners/experiment_history.py) and
[`experiment_depth.py`](../experiments/runners/experiment_depth.py). They share
entry-point admission and the same runner, queue, persistence and notebook.
They resolve to the original calibrated regime digests, seeds and balanced order.
Their existing `ResolvedStudy` admission retains scientific protocol ownership;
this software does not redefine historical plans or reopen seed families.

For a separately authorized **future** comparison with a newly admitted plan:

```bash
uv run --extra notebook python -m experiments.runners.experiment_depth \
  --plan /path/to/current-source-resolved-plan.json \
  --schedule /path/to/approved-execution-schedule.json \
  --hardware /path/to/configured-hardware.json --out .runs/depth-comparison
```

Use `experiment_history` for the history declaration. The schedule must match the
plan's seeds/order, reserve all its scientific deals, and fit training plus
monitoring inside the original training envelope. Plan admission verifies the
calibration artifacts and exact clean source/runtime bindings before execution.
This entry point does not run calibration or grant another allocation. Existing
live ETU-103/106 campaigns continue under their frozen sources; do not attach this
runner to those output directories. Historical launch scripts remain for their
frozen evidence. Their scientific scoring remains separate from live monitoring;
these new entry points do not silently consume scientific endpoint deals.

## Milestones and evidence

The queue discovers completed stage raw checkpoints and admitted periodic exports
while learning continues. The executor owns periodic export cadence and update
boundaries. Frozen jobs retain the TrainingRun snapshot, artifact hash/size/path,
original cumulative coordinates and MonitorProtocol. Changed identities fail.
Duplicate discovery does not launch duplicate work. The existing ETU-101 arena
owns admission, legal Commands, four seat/deck legs, timeouts and exact replay.

Default monitoring uses 25 reserved deals / 100 games against source-pinned
scripted greedy. Declare smaller cohorts only for explicit bounded fixtures.
`scientific_deal_seeds` prevents overlap with frozen evaluation families.
Monitoring results cannot change seeds, cohorts, budgets, counts or selection.
A missing/failed/truncated/replay-failed leg makes aggregate rates unavailable.
Intervals resample whole four-leg deals conditional on the checkpoint; they are
not independent-training-seed uncertainty or evidence of general strength.

The notebook generates one read-only HTML dashboard from editable loading and
plotting cells. Its default page starts with current progress, freshness, latest
applicable loss and evaluation, hardware, costs and failures, then shows matched
milestone strength curves and two diagnostics. No inline plots are emitted.
Compatible stage/update milestones are compared across every expected run;
unmatched latest checkpoints remain status only. Metric links explain definitions,
units, uncertainty and deeper library analysis in [the metric guide](experiment-metrics.md).
ETU-101 owns the source diagnostics; W&B remains an optional projection.

## Failures and explicit continuation

Each learner process and evaluator job has a retained attempt, including launch,
admission, timeout and replay failures. Failed learners do not discard already
exported checkpoints. Independent pending cells may proceed if their full
allowance still fits. Stopped evaluation attempts are not retried automatically.
Workers run in isolated process sessions with watchdogs; teardown kills and reaps
the session. Inherited leases prevent a second execution while an orphan worker
remains live.

```python
# Continue only pending work under the same frozen intent/runtime and budget.
result = run_experiment(experiment, hardware, output, resume=True)
# Explicitly recover one stopped TrainingRun through the existing recovery owner.
result = run_experiment(experiment, hardware, output, resume=True, recover=(0,))
```

Recovery requires a valid existing snapshot and the same host/source/runtime.
Unsupported recovery fails visibly; it never restarts training from scratch under
the old attempt. Original records and paths remain. A failed recovery child is its
own attempt; no parent can be silently forked twice. Resume grants no extra budget.
After abrupt death, running attempts conservatively charge their full reserved
allowance, and unobserved wall time includes downtime. There is no byte-identical
or cross-machine recovery promise. A budget-exhausted execution may remain
incomplete with pending jobs; that is evidence, not a retry signal.

Execution receipts include process startup and supervisor accounting. Arena cost
is a subset of evaluator process cost and must not be added again. TrainingRun
cost is a subset of the corresponding learner process cost. The notebook presents
these source measurements separately and displays execution totals. Imported
producer cost, if present, remains a separate receipt; fresh execution cost does
not claim to include historical production. Missing measurements are unavailable.

## Interpretation and durable knowledge

An experiment skill invokes the Python interface, reads the notebook and evidence,
distinguishes measured observations from causal conclusions, updates the research
ledger and appropriate repository knowledge, and chooses the next experiment.
The runner and report never rewrite that knowledge automatically. Notebook cells
provide exact evidence paths/hashes and editable interpretation space.

The bounded acceptance demo uses multiple small regimes and independent seeds,
real checkpoint reload and arena replay, then executes the notebook headlessly.
It establishes scheduling, persistence and report behavior only, not method
improvement, challenger strength or completion of Trained Challengers.

### Editable demo

[`experiment_demo.py`](../experiments/runners/experiment_demo.py) contains the full
small declaration: two model widths, two seeds, one stage per run, one reserved
four-leg monitoring deal per checkpoint and a 750-second maximum allocation.
Configure a local CPU resource named `fixture`, then run:

```bash
uv run --extra notebook python -m experiments.runners.experiment_demo \
  --hardware /path/to/configured-hardware.json --out .runs/experiment-demo
uv run --extra notebook jupyter lab .runs/experiment-demo/comparison.ipynb
```

The notebook is the editable report generator; `comparison.html` is the default
read-only viewing surface. Headless execution checks HTML figures and absence of
inline notebook plots. Refresh preserves existing notebook bytes and edits.

The [revised retained generator](../experiments/study/experiment-demo/report-generator.ipynb)
writes the [HTML demo](../experiments/study/experiment-demo/comparison.html).
Jack Heart's original edited notebook and checkpoint copy remain unchanged.
The [evidence note](../experiments/study/experiment-demo/README.md) records the
original fixture costs, failures and limits. This revision reruns reporting only.
