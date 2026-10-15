# Training time profile and benchmark calibration

## Accepted delivery order

2026-10-15, Jack Heart: start with training only, without evaluations. First
deliver regime/stage/update time accounting and a timeline across training runs
and attempts, backed by their real records. Include training-owned checkpoints,
transfers and waits where measured; leave unavailable intervals explicit.
Then expand the same regime-level view to include parallel evaluation work and
its overlap with training. Evaluation is deferred from the first deliverable,
not required to validate it. The broader model below is the eventual scope;
it must not enlarge the first implementation into a campaign-wide eval project.

## Implemented first report

`manabot/training/time_profile.py` reads existing TrainingRun/ExperimentRun records
into aggregate phase hierarchies and training-only attempt intervals. The existing
experiment dashboard uses it; `experiment_report.write_time_profile` also writes
a standalone HTML report plus timing JSON with source hashes. No training or
tracker is invoked. Root/stage residuals remain explicit; absent counters are not
measured zero. Copied completed recovery stages are excluded, missing parents and
inconsistent totals remain warnings, and running intervals stop at saved evidence.
Monitoring-export and recovery totals are already inside stage time, not siblings
to add again. Detailed subphase spans remain unimplemented.

Delete—do not maintain: the old flat per-run stage-cost table in
experiment_report.write_dashboard is replaced by this hierarchy. Existing run
records, costs/resources/failures and editable notebooks are preserved. There is
one report renderer for the standalone page and the dashboard.

The real run022 artifact is scratch/training-time-report.html, selecting all24
ordinary timing cases (both shapes); deterministic diagnostic cases remain in
retained evidence and are not timing samples. All24phase trees reconcile without
accounting warnings and all24calendar intervals are present. The same schema also
tests against experiment-demo's explicit process timestamps without importing its
evaluation costs into the training-only view. Structural SVG/HTML checks are
available here; no visual browser rendering is claimed.

Validation: `uv run --no-sync pytest tests/training/test_time_profile.py
tests/training/test_experiment_report.py -q` passes19tests(10.74s); Ruff and
diff checks pass. Real24run standalone report has no attribution warnings.

2026-10-15. Jack Heart requested a high-level, data-backed account of where time
goes across a training regime and across runs, including evaluations, followed by
focused benchmarks whose predictive relationship to real training is measured.
This steers the inference optimization goal toward whole-system evidence before
further isolated optimization. Run022 finished with verified artifacts, budget
accounting and provider-confirmed cleanup. No new rental is
selected by this note. The original total GPU budget remains $50.

## The report to build

Start at an experiment's elapsed calendar time. Show concurrent lanes for training
attempts, evaluation attempts and machine lifecycle. Drill into a selected regime,
stage and update window. Alongside that timeline, show a flamegraph-style hierarchy
whose widths represent accumulated time for a named accounting basis. Keep wall
time, occupied process time, CPU work, device activity and rental dollars separate.
The question at each level is: what work or waiting accounts for this cost, and
which focused benchmark can investigate it?

Proposed hierarchy (conceptual, not measured percentages):

```
Experiment / campaign
  Machine lifecycle: acquire, setup, artifact transfer, teardown, paid idle
  Training run / attempt / regime / stage
    Collection: native step + encoding, input materialization + transfer,
                policy forward, sampling/log processing, rollout assembly
    Learning: targets, forward/loss, backward, clipping + optimizer, EMA
    Checkpoints / recovery snapshots / artifact exports
    Diagnostics / persistence / other observed work
    Unattributed time
  Evaluation: queue, checkpoint load, games/search, reduction, export
  Reporting / orchestration / failed attempts / unobserved gaps
```

These categories do not imply that evaluation is sequential with training or that
device work is exclusive of host work. A wall-clock report must show overlap; a
resource-cost report can add distinct process occupancy. It must never add GPU
activity to a host forward interval that already contains that activity. Critical
path claims require dependency/wait evidence, not just overlapping timestamps.

## What exists now

- `manabot/training/models.py`: TrainingRun has setup, elapsed/calendar,
  recovery/downtime and monitoring export counters. StageRecord has collection,
  learning, export, diagnostic, CPU seconds and work counts.
- `manabot/training/execution.py`: timers surround real collection/update calls.
  Monitoring exports and recovery checkpoints have different owners. On failure,
  remaining unaccounted stage time is assigned to diagnostic_seconds; that field
  cannot universally be interpreted as pure diagnostic computation. Historical
  residuals must stay honestly labeled, not reverse-engineered into invented spans.
- `experiment_execution.py` and `experiment_runner.py`: ExperimentRun retains
  attempt start/end, process occupancy, evaluator occupancy, elapsed/calendar,
  uncertainty and cost. Recovery lineage already exists.
- `checkpoint_queue.py`: evaluation attempts retain start/end, charged seconds,
  status and checkpoint binding. `monitor_evaluation.py` retains evaluation
  duration, per-game rows and CPU measurements. Monitoring and scientific
  evaluation are distinct purposes and must remain distinct in the report.
- `experiment_report.py`, `docs/experiment-metrics.md`: the offline report already
  reads these records and shows a stage-cost table, execution costs and failures.
  Extend that evidence path; do not introduce another training executor.
- Public Etude client at `/Users/jack/src/etude-sdk`: `etude.track.run`,
  `track.records` and `docs/tracking.md` expose named clocks, numeric metrics,
  checkpoint identities, evaluations and cross-run reports. Numeric logging can
  carry coarse phase totals now. The inspected API does not expose a generic
  nested timing-span/profile contract. Metric rows alone do not recover nesting,
  concurrency or gaps. Its bounded logging queue can drop rows; accounting must
  expose missing coverage rather than silently treat missing rows as zero.
- Etude Studio at `/Users/jack/src/etude` is a different repository. Generic
  tracking/profile storage belongs in the public client; managed cross-run report
  UX belongs in Studio. Domain phases and counters remain in manabot. This note
  changes neither repository and does not import private content into the client.

## A measured first picture

`training-time-profile.svg` is an aggregate flamegraph-style picture generated
from one actual run.json: run020's first predeclared ordinary 4096-stream
single-worker reference, 3 updates and98,304learner rows. The selection rule,
source path, SHA256 and exact values are in `training-time-profile.json`.

The run took22.210724s: collection16.750875s, learning3.943399s,
exports0.062984s, setup0.249799s, unattributed stage1.196760s,
other run0.006907s. Collection is75.4% of this run. These are aggregate widths,
not reconstructed chronological spans. This is real short-run GPU training,
not a representative long campaign; this recipe exercises no evaluation.
Do not graft run018's profiled subphase percentages onto this run's collection.

The retained experiment-demo does contain real executions and overlapping evals,
but explicitly calls itself a tiny software fixture:50.78wall seconds versus
61.30additive learner/evaluator process seconds, including47.17evaluator seconds.
It is useful for validating overlap accounting, not for estimating production eval
share. A real long-running training/evaluation cohort is still needed to support
that conclusion. Missing campaign evidence is an output of this research.

Run022 is now complete and verified. `training-time-comparison.svg` shows every
ordinary4096-stream trial (12bars), partitioned using that trial's own recorded
collection/learning/export totals and explicitly named remaining time. Its JSON
binds all12source files by hash. The paired complete-run throughput gain is27.0%
for borrowed4versus cloned1 and18.5% for borrowed4versus cloned4. Large collection
time falls while learning time remains similar. All32cells completed and full
deterministic proofs passed; ordinary exact-state parity failed, including raw/EMA
differences in two512borrowed4cases. This gives a real phase-to-whole-run positive
example, not yet a calibrated fast-benchmark predictor or an accepted adoption.

## Map each area to a fast benchmark

| Area in the real report | Focused benchmark and retained input | What it can predict; required parent check |
|---|---|---|
| Native producer / observation encoding | Existing pipeline_benchmark replay and worker_scaling, with recorded policy outputs, active masks, seeds and realistic game states | CPU producer cost/scaling. Validate against the same collection window with a live GPU; no direct claim about full training. |
| Input preparation, transfer, serving | Replay an ordered real request trace through production preparation + model + response handling, with the actual checkpoint and request-size/arrival pattern | Per-request service cost and collection impact. Existing worker_inference/worker_service_profile cover small requests; their transfer to ordinary collection failed in017. |
| Sampling, rollout materialization and batch assembly | Retained real transitions and row/mask ordering through the production collector; existing pipeline replay as starting point | Collection data handling; validate complete collection and regime. Direct storage020 is a negative calibration example, not a retained fast implementation. |
| Learner update | Exact collected batches and starting model/optimizer/EMA state through the actual objective/update function; cuda_performance learner mode only where its path matches | Time per update, gradients and state evolution. Validate several real collect/update cycles; shape, retained fraction and loss recipe matter. |
| Evaluation | Frozen checkpoint, declared decks/opponents/deals and production evaluator, including load and export | Evaluation latency/cost and queue service rate. Validate with concurrent training because CPU/GPU contention and queue delay can change the answer. |
| Checkpoint/persistence | Real model/optimizer/recovery payload sizes through existing save/hash/copy/upload paths, same storage class | Blocking export cost and transfer demand. Validate at the actual checkpoint cadence and with background uploads. |
| Orchestration/reporting | Recorded attempt/evaluation events and representative artifact counts through existing scheduling/report readers | Queue/report/storage overhead; validate campaign elapsed time and resource spend. Exclude claims about inference throughput. |
| Complete regime / experiment | Shortened real recipe retaining collection, updates, checkpoint cadence and evaluations; then an unchanged production canary | Parent validation for all focused benchmarks. A shortened recipe is itself a proxy until calibrated against sustained runs. |

Keep warm/cold start, small/large requests, sparse tails, early/later checkpoints,
hardware CPU quota, GPU/driver, precision, world/decks, worker count and contention
as explicit strata. Do not average away a regime in which the proposed change
regresses. Existing replay captures untrained CPU policy outputs; deriving traces
from actual GPU training/checkpoints is an explicit gap, not already implemented.

## Establish predictive validity

For each implementation change, measure a pair: focused benchmark before/after,
and its actual parent phase before/after on the same source pair and workload.
Also measure complete regime duration and fixed useful-work counts. Interleave
controls/candidates, include failures, and keep instrumentation outside ordinary
timing arms. Use several independent changes, not repeated samples of one change,
to test whether benchmark improvements predict real improvements.

Report the predicted versus observed effect size, direction agreement, ranking
agreement across changes, and uncertainty by workload/hardware stratum. Correlate
paired deltas rather than absolute durations across different workload sizes. A
benchmark that ranks changes correctly but exaggerates their total benefit needs
that limitation recorded. Test calibration on held-out changes/checkpoints;
one successful pair does not establish correlation. Existing017 supplies a clear
counterexample:1.844x worker serving, only1.0082x ordinary large-run throughput.

For strictly sequential, exclusive phases, their measured baseline shares give a
conditional whole-run prediction; report the prediction error. For overlapping
training/evaluation or pipelined CPU/GPU work, use observed waits/dependencies and
validate the critical path instead of blindly applying additive phase fractions.
Benchmark validity belongs to a specific target and scope: uncalibrated,
supported within named strata, or contradicted. Never give a benchmark a global
"representative" label.

## Minimal instrumentation and next delivery

1. Extend the existing offline report from retained records first: training-only
   regime/stage accounting with visible residuals and failures, and training-run/
   attempt lanes wherever real timestamps exist. Show provenance and coverage per
   row. Parallel evaluation lanes belong to the later regime-level expansion.
2. Add low-overhead phase totals to normal execution and publish them through
   existing Etude numeric metrics. Validate timing overhead against uninstrumented
   runs. Separate always-on accounting from scheduled sampled Torch/native traces.
3. Specify one optional generic Etude span stream: stable span/parent IDs,
   run/attempt/process/thread identity, phase, monotonic start/end on a named clock,
   wall-clock anchor with uncertainty for cross-machine display, outcome and work
   coordinates. Retain checkpoint/dependency links and coverage/drop indicators.
   Count ancestors/descendants and resumed attempts once. A CPU duration, host
   interval and GPU activity interval are different units/views. Existing metric
   storage stays the owner of scalar histories; detailed trace artifacts remain
   referenced, not expanded into a metric per kernel.
4. Capture a bounded profile from actual training at early, steady-state and later
   policy windows. Keep full accounting throughout. Choose benchmark trace sets
   from those windows, including request order and tails. Later add concurrent
   evaluation to the same regime-level accounting and benchmark validation.
5. For the next optimization, run the mapped benchmark plus the paired regime
   check. Accumulate calibration evidence before trusting the benchmark to screen
   changes cheaply; retain a complete-regime acceptance check for adoption.

The training-only report is implemented as described above. Further phase spans,
Etude integration, real later-policy captures and benchmark calibration remain
outstanding; the initial report does not establish that the optimization goal is
complete. No new isolated optimization cohort is selected.
