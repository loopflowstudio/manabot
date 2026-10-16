# Training time reports from selected Etude runs

Status: selectable report implemented; historical long-run cohort remains incomplete.
Direction supplied by Jack Heart on 2026-10-15.

The report should answer **where time went across a chosen set of real training
runs**, starting with some of the longest runs recorded since adopting Etude.
It should connect that account to focused benchmarks whose ability to predict
training improvements is measured, rather than assumed.

## First experience

Open a training report over one or more explicitly configured Etude log sources
or retained manabot artifact directories. Browse the discovered runs, select a
cohort, and see its combined time accounting, per-run variation and timeline.
The report reads existing evidence; opening or refreshing it never launches work.

The run picker provides:

- Experiment and run identity, regime/model, seed, hardware, status and dates.
- Recorded training duration, learner transitions, updates, measured cost when
  available, and timing-data coverage.
- Search, sorting, individual selection and a “select largest five” action.
- A visible filter for labeled benchmark, smoke and fixture runs. These are
  excluded from the initial training selection; failed or interrupted training
  remains eligible.

The initial interpretation of “largest” is **observed active training duration**,
not planned rental hours, file size or model size. Alternative sorts expose work
counts and measured cost. Missing values remain unavailable; sorting must not
substitute a different quantity without saying so. If active time is unavailable,
the run stays discoverable for manual selection.

Selecting a logical training run includes its attempts and required recovery
ancestry. Preserve an explicit distinction between a scientific run, its process
attempts, and an Etude launcher job that may execute several training cases.
A launcher job is not automatically one training run.

## What changes with the selection

Show the following views over exactly the selected cohort:

1. **Summary:** selected logical runs, attempts and statuses; observed training
   hours; useful work; measured cost and coverage where available.
2. **Aggregate phase hierarchy:** accumulated exclusive phase seconds, with
   duration-weighted shares. Collection, learning, setup, checkpoint/export work,
   diagnostics and unattributed time appear only to the depth recorded.
3. **Per-run comparison:** individual phase distributions and duration/throughput
   variation, with drilldown to the source run, stage and available update window.
4. **Training timeline:** attempt lanes on one calendar axis, including overlap,
   gaps and partial attempts. This is distinct from the aggregate flamegraph.

The same selector controls all four views. Save exact selected identities and
source hashes/versions in an exportable selection manifest. The page may preserve
view state in its URL; a saved manifest additionally pins the evidence snapshot.
Refreshing discovery must not silently add new runs to an explicitly saved cohort.

## Accounting rules

An aggregate flamegraph shows accumulated duration, not execution order, CPU
utilization or GPU utilization. A nested interval is already included in its
parent. Never add host forward time to device activity contained within it.

Sum exclusive phase seconds first, then divide by the appropriate observed-time
denominator. Do not average per-run percentages: a ten-second run and a ten-hour
run contribute different amounts of time. Keep unmeasured residuals visible.
Missing historical fields are not measured zero. Show coverage counts and the
denominator for each statistic; a partial total must be labeled partial.

Keep these quantities separate:

- Active collection/learning time, when explicitly recorded.
- Total recorded regime/attempt time, including its measured overhead.
- Additive process occupancy across attempts.
- Union of observed calendar intervals and first-to-last calendar span.
- CPU work, device activity and rental cost, where separately measured.

Running attempts end at their last saved observation, never at report-render time.
Calendar spans that include downtime cannot be relabeled process occupancy.
Overlap alone does not establish a dependency, wait or critical path; cross-host
clock uncertainty remains a limitation unless recorded.

Recovery needs two different treatments. Timing measures the actual attempts,
including failed or superseded work. Scientific progress counters may include
inherited work or rewind to a checkpoint. Do not add cumulative endpoints, copied
completed stages, or a parent's sunk cost again. Preserve attempt costs even if a
later attempt replaces its learning-curve rows. If ancestry or branch ownership
cannot establish a safe total, report that counter as unavailable with a reason.

For comparable work, throughput is `sum(work) / sum(corresponding time)`, with
both quantities drawn from the same covered runs. Group by recipe, work-unit
meaning and hardware before pooling rates. Keep different hardware/model regimes
visible. Arbitrary selected runs are not independent seed replicates; do not
attach a scientific confidence interval to their combined duration or rate.
Rental ceilings are not measured spend, and shared rental cost needs explicit
allocation before it can be presented as training-only cost.

## Existing owners and integration boundary

manabot already owns the domain accounting and an offline report:

- [`time_profile.py`](../../manabot/training/time_profile.py) partitions
  `TrainingRun` records and creates attempt intervals, preserving recovery and
  missing-data warnings.
- [`experiment_report.py`](../../manabot/training/experiment_report.py) reads
  retained evidence and exposes `write_time_profile`, producing standalone HTML
  and timing JSON with source hashes.
- [`models.py`](../../manabot/training/models.py) and
  [`experiment_execution.py`](../../manabot/training/experiment_execution.py)
  own training and attempt records.

The public [Etude client](https://github.com/loopflowstudio/etude) already owns
log discovery, columnar metric reading, clocks, attempts and continuation rules.
Relevant interfaces are `etude.track.read.runs`, `read_all`, `RunData`, and
`etude.track.storage.storage_from_uri`. Its report has run checkboxes and URL
state. Reuse its reader rather than implementing a second Etude storage format.

Etude owns generic logging, discovery and general report capabilities. manabot
owns the meanings and projection of its collection/learning counters. The private
Etude Studio owns its managed research/report experience. This implementation
extends the existing report in this repository; it does not require changing
Studio or moving domain-specific training logic into Etude.

Native `TrainingRun` and Etude `RunFile` both use the filename `run.json`, but
they are different schemas. Detect the schema and retain an explicit identity
mapping. Do not infer phase semantics from similar metric names. Generic logs
without a known timing contract can still supply a selectable run and their
recorded clocks; missing detailed attribution stays visible.

Read only configured sources, using existing credentials. Retain source identity,
fetch time and hashes of the bytes used. Do not embed credentials, signed URLs,
checkpoint payloads or arbitrary raw configuration in the report. Metadata and
the needed metric columns should suffice; no model download is required.

Delete — do not maintain: replace the standalone report's fixed all-runs view
with cohort selection on the same accounting path. Keep one phase-accounting
owner and preserve the existing dashboard, per-run detail, provenance and tests.
Do not add another training executor or a parallel experiment registry.

## Initial evidence and coverage limits

The current report was exercised on 24 ordinary timing cases from GPU run022.
These are short optimization trials, not a mature training campaign. Their
availability does not satisfy the request for the largest historical runs.

A read-only inventory on 2026-10-15 found accessible Etude journal records,
including three local PPO tracking runs with checkpoints at 9,984 environment
steps and launcher live checks. The inspected native records were short
benchmarks. Access to listing the older training-artifact bucket was denied.
The inventory did not expand all archives or parse every large record.

Consequently, the largest historical training cohort has **not been established**.
The report must expose source coverage and unavailable sources. A permission
failure is not an empty history, and benchmark trials must not silently replace
long training runs. Existing ETU-126 protocols identify longer candidate work,
but planned endpoints and budgets do not prove actual completed training time.

## Acceptance checks

- Real Etude-written logs and native retained records can be discovered and
  selected through the report's supported entry point.
- Selection updates aggregates, per-run detail and timeline together; the default
  top-five choice and explicit saved selections are reproducible.
- A long and a short run produce time-weighted shares and pooled rates with
  correct denominators. Missing counters remain missing.
- Duplicate exports, resumed attempts, copied stages, rewinds, missing parents
  and branching ancestry do not silently double-count work or duration.
- Overlapping attempts produce a calendar union distinct from their summed
  duration. Incomplete attempts stop at saved evidence.
- Failed source reads, malformed/conflicting records and incomplete coverage are
  visible. Rendered data is escaped and selection exports retain provenance.
- Validate against accessible real logs without claiming that they constitute
  the complete historical cohort. Tests require no GPU rental or training run.

## Later expansion

First deliver training only. Then include **parallel evaluation at the regime
level**, with checkpoint dependencies, evaluator attempts, transfers, contention
and overlap on the same timeline. Evaluation is not required for this first view.

Subsequent instrumentation can add low-overhead phase totals and sampled detailed
traces at early, steady-state and later training windows. Host timers and CUDA
activity must remain distinct, and profiling must not perturb training RNG.

Map each observed area to a faster focused benchmark using retained real inputs:
native replay, ordered inference requests, rollout materialization, saved learner
batches, checkpoint payloads and, later, evaluation matches. Validate benchmark
before/after effects against the parent phase and complete training regime across
multiple independent changes and held-out workloads. Run017's 1.844× serving gain
with only about 1.008× ordinary large-run throughput is a retained counterexample
to assuming that an isolated benchmark predicts training speed.
