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

## Implemented report

The accepted cross-project design is committed at11902c4f in
`docs/plans/training-time-cohorts.md` and pushed to PR292's branch. That stable
revision is suitable for Etude to reference; the implementation extends the same
training-only scope.

`manabot training-report --source DIR_OR_S3 --out report.html` discovers native
training artifacts and Etude logs through the optional reports extra. The offline
page selects logical runs, defaults to the five longest with observed active time,
and updates weighted phase totals, coverage, descriptive distributions, stratified
rates and calendar lanes. It exports/restores exact source-bound selections.
`write_time_profile` now uses this page; existing dashboard accounting remains.
Phase partitioning stays in time_profile.py. time_cohort.py owns native lineage
projection and rendering; time_sources.py uses Etude's reader for log storage.

Delete—do not maintain: the standalone fixed-all-runs page is replaced. Preserve
native phase accounting, recovery warnings, the experiment dashboard and provenance.
No second training executor or tracker format is introduced. New sources with
unknown duration, historical missing counters, conflicting exports and unreadable
archives retain explicit gaps. Etude active-clock totals never become invented
regime overhead. Measured rental cost remains unavailable in these projections.

The repaired scratch/training-time-report.html opens with an aggregate flamegraph
above the picker, selecting completed41min etu103-recovery-v2-pilot-1 and79min
live etu103-recovery-v3-science-2 snapshots. A static companion is training-time-flamegraph.svg.
The page has39logicalruns: four historical native snapshots,32labeledGPU cases
and three older Etude PPO logs. Historical S3 access is now authorized and verified.
Four run.json files were hash-verified against current job manifests; larger
hundreds-of-megabyte records were not fetched in this bounded refresh. No complete
historical ranking is claimed. Some running snapshots have contradictory stage
counters and remain unattributed. No new GPU work or browser rendering occurred.
Receipt: .runs/historical-training-metadata/receipts.json; regeneration script:
.runs/render-training-time-report.py.22focusedtests pass, including browser event
wiring in a headless DOM and hierarchy accounting. Source snapshots remain local.

2026-10-15 latest direction: basic phase accounting and flamegraph only. Detailed
PyTorch tracing is deferred; the instrumentation discussion below is prospective.

## Active GPU bottleneck investigation

2026-10-15: new goal authorizes shipping and up to$25for new GPU training runs,
separate from the previous$50campaign. Basic timers only; PyTorch tracing deferred.
Budget: .runs/gpu-bottleneck/budget.json. Historical startup evidence and exact
source limits are published in experiments/training-time.md and its data directory:
97.5%post-first-update active coverage,82.5%active collection; modelw384-d8.
Admission-wait and optional collection timers are committed in839eba2d/79ea7a0e.
Collection uses the existing Profiler, resets per stage/attempt, records typed
seconds/calls and nests in the report without double counting or new CUDA waits.
29instrument/report and14native/recovery/study tests pass; fullCPUoff/on rollout,
RNG/raw/EMA/Adam/diagnostics exact. Learning subphases remain unimplemented.

001/002 are settled operational failures; all missing measurements and costs are
retained in experiments/training-time.md. Study003 completed all eight cases;
all40objects/106members verified (2.477GB), including retained outlier checkpoints.
Source7d00992a; run d51e85f1a805; provider deletion confirmed. No GPU rerun for recovery.

2026-10-16 delivery: PR292 synced with main at16306b87; only uv.lock conflicted,
resolved against merged dependency definitions (uv lock --check passes). Nine
Rust debug vector tests, fmt and Clippy pass. Initial Python batch108pass/1disk-
reserve failure; freed3.75GB of exactly reproducible source archives. Rebuilt native
extension;80selected native/recovery/model tests pass, including failed test.
Isolated remote suite175pass/2skip; Ruff passes. Added producer/profiling tests to
CI and refreshed PR copy. CI pending; manual merge only. No005 admission occurred.
Raw commands/results: .runs/gpu-bottleneck/pr292-*.log. Continue bottleneck research
after this delivery boundary; no new paid run is needed to land these instruments.

w384-d8/64streams/16updates: off/onmedian18.777/19.079s,436/429learnerrows/s;
steady554.5/542.5. On/offratio1.0161passesdescriptive5%gate. Instrumentedcollection
62.0%run/75.0%active; combinedpolicy86.8%collection/53.8%run; native4.55%collection/
2.82%run. Requestsmean10.95rows,67nativesteps/update. HostintervalsnotGPUutilization.
ExactproofFAILED: initialweights+firstrollout exact; all3actions/buffers/seeds/
samplingRNGexact, firstlearningvaluelossdiffers; laterrollout/weights/Adam differ.
Proofrawmaxdifference6.37e-5. Ordinaryoffcontrolsalsodiffer: firsthasoneweight
293.4111at extra_attention.6.mlp.2.weight[362,1461], tensorinitialmax0.11701;
otheroffcontrolmax0.11725. Firstordinarycontrol395optimizerexposures/18240decisions,
others390/18261; all8192learnerrows. Firstsamplednumericalhealthmatchescontrols;
laterperstepparametermovementunmeasured. Causeunknown, notdismissedasrounding.
Allgates/everycase/failedproof/stateaudit andstandalonereport retained in
experiments/data/training-time/basic-profile-003. No acceptance/tolerancewaiver.

scratch/gpu-training-report.html +gpu-training-flamegraph.svg showdiagnostic003;
ordinary historicalreport nowselects41mincompleted+79minrunningrecords. Live
stage.secondsstays0untilfinalization inexacthistoricalsource45940fe3; reporter
showsobservedphasecounterswithoutinventingstagewall, rejectsparentcontradictions.
79minsnapshot95.8%steadyactive/81.1%collection, sourceboundJSONpublished;25report
tests pass. Packager/groupchecks5pass. Publicreportsstatefullrecoveryandnumerical
failure prominently. No browserrenderingclaimed; SVGbounds+headlessDOMchecked.

Outlier follow-up is published in41fc1cea and outlier-audit.json: intended Adam
movement bound0.0130134 versus293.454; one element across133parameter tensors.
EMA is also abnormal. Causeunknown; update loss is last nonempty minibatch.

004complete: run7d6c152708bc/source5a99c7e4, all6diagnosticcases exact within
comparisons.24checkpoint payloads independentlyrehashed;31objects/126members
verified. Fourdeterministic3updatecontrols match003offproof; twoordinary16update
controls matchallfive non-anomalous003ordinarycases. Largestmovement0.0001733.
Noanomalyreproduced. Addedwitnesssynchronization anddifferentEPYC7543host confound
causalinference; no throughputclaim or003gatewaiver. Reports/data published under
experiments/data/training-time/basic-profile-004. Providerdeleted04:18:14UTC;
$0.238estimated, campaign$1.4055estimated/$2.21conservative plus$2ancillary.
NoliveGPU. Privatehelpers/result-004 retainfullreceipts; cachemayberemovedafter
verification. Next: matchedmodel/batchattribution withfresh ordinarycontrols.

## Next instrumentation boundary

The existing manabot.infra.profiler.Profiler already owns nested host timers and
bounded duration samples. Reuse that owner for optional ordinary-training phase
accounting; do not introduce a second timing framework. Before enabling it, a
regression reproduced that its reservoir sampling consumed global Python RNG
after max_samples calls. TimingNode now owns a fixed-seed private Random instance.
The new enabled/disabled RNG-invariance test fails on the original implementation;
all8profiler tests pass after the fix. This does not explain022CUDA parity failures:
ExperimentHypers.profiler_enabled defaults false in the measured ordinary path.

The intended instrument is opt-in at the training execution configuration, with
typed phase totals/call counts retained on the existing StageRecord. Root phases
must bracket the same collection/learning calls and synchronization as current
stage counters. Nested collection scopes cover input preparation, forward host
intervals, policy logs/sampling, native step+encoding and final assembly. The
single collector owns the profiler; producer threads do not share its stack.
Reset at the attempt/stage boundary, preserving existing recovery cost ownership.
The report can expose nested measured phases under their matching parent and keep
the unmeasured remainder. Host forward intervals are not device execution time.
Detailed CUDA traces remain separately sampled artifacts; do not synchronize every
subphase or present their overhead as ordinary throughput. Test exact rollout/
learning/RNG equivalence with profiling toggled and quantify instrumentation cost
before using it to choose another paid optimization experiment.

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

## Run selection and cohort statistics

2026-10-15, Jack Heart: select runs from Etude training logs, aggregate the
selection, and start with some of the largest historical training runs. Proposed
default: top five logical training runs by recorded active training duration,
with alternative sorting by learner transitions, updates and measured cost.
This is a reversible interpretation of “biggest,” not a completed selection.

The picker should expose experiment/run identity, dates, regime/model, hardware,
status, training duration, useful-work counts and timing coverage. Exclude labeled
smokes/benchmarks by default; retain failed/interrupted training and expose the
filter. Selecting a logical run includes its attempts and required ancestry,
without duplicating copied stages or cumulative continuation counters. Save exact
IDs, source versions/hashes and selection rules so a later refresh cannot silently
change the cohort. Native TrainingRun and Etude RunFile both use run.json but are
different contracts; discover by schema and retain an explicit identity mapping.

For the selected cohort show total observed training hours, summed exclusive phase
seconds and their duration-weighted shares, per-run distributions, useful work,
and stratified throughput (sum comparable work / sum corresponding seconds).
Calendar coverage is the union of observed attempt intervals, separately from
first-to-last span and additive process hours. Show missing duration/phase counts
and coverage denominators; missing phase data never becomes zero. Preserve
per-run drilldown and hardware/recipe strata. No pooled scientific confidence
interval over arbitrary heterogeneous runs. Parallel eval remains later scope.

Sources: etude-sdk/src/etude/track/read.py runs/read_all already discovers runs;
track/report.html already has per-run visibility checkboxes and URL state.
manabot/training/experiment_report.py load_evidence/write_time_profile consumes
local TrainingRun artifacts; it has no Etude picker or selected-cohort aggregate.
Reuse those owners rather than adding a training executor.

Historical inventory limitations and resolved access failures are retained in
experiments/training-time.md and the published design. The implemented report's
current cohort and coverage are described above; it is not a complete ranking.

## Minimal instrumentation and later delivery

1. Connect run selection and cohort statistics above to retained evidence. Keep
   training-only accounting, provenance, residuals and attempt lanes explicit.
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
