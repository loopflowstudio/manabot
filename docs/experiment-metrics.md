# Reading the experiment report

The notebook is the editable report generator. Run All reads local evidence and
writes one HTML dashboard. HTML is the default read-only surface; it does not
poll, train, evaluate, or access a tracker. Change the notebook's question,
selected metrics and plotting code to regenerate that surface. Existing notebooks
are create-once: use a new filename to adopt a newer generator without losing edits.

## Freshness

Rendered time is the UTC time of report generation. Evidence as-of is the
execution coordinator's saved `last_seen_unix`, not the JSON file modification
time (copies change modification time). Age is seconds since that heartbeat.
A running status with an old heartbeat is stale, not proof that training is still
running. A completed execution is labeled final; otherwise status is in-flight.
Checkpoint age is lag in recorded training seconds from the latest diagnostic,
not wall-clock time since export. Missing time coordinates are unavailable.

## Learning

Progress is shown separately for each variant and independent training seed.
Updates are collect/update iterations, including empty-filter skips; planned
updates come from declared stages. Learner transitions, native decisions and
optimizer exposures count different work. SPS explicitly means cumulative
learner transitions divided by recorded training seconds at the latest diagnostic.
It includes measured training overhead, not just model inference. Host contention
is uncorrected. There is no uncertainty interval for these process measurements.

RL loss is the latest retained optimized minibatch objective, with policy/value
and regularization semantics defined by the recipe. It is not predictive log
loss, is not necessarily comparable across recipes, and can be absent on an
empty-filter update. Distillation cross-entropy is labeled as such. Do not infer
strength from either loss. Latest rows without a loss remain unavailable rather
than carrying a stale optimized value forward without a label.

Deeper example in a notebook cell (figures go into the HTML `sections` list):

```python
sections.append(('Collection KL', 'learning',
                 [metric_figure(evidence, 'rl/collection_kl')]))
# Inspect all retained learning/resource scalars before selecting other panels.
rows = evidence.metrics(evidence.runs[0])
```

## Evaluation

Win fraction counts wins over all valid games; draws are not wins. The source
monitor also retains score (win + half draw) and draw fractions. Latest evaluation
is checkpoint status, not a ranking of variants. Each row identifies the original
checkpoint's updates/transitions, training-time lag, opponent and completed/planned
game counts. Incomplete or failed evaluation attempts cannot supply rates.

The saved 95% percentile interval resamples complete four-leg deals, retaining
both deck and seat assignments. It conditions on one checkpoint and does not
measure training-seed uncertainty. One deal can yield a zero-width interval,
including in the tiny demo; this does not establish precision. Monitoring deals
are inspected development evidence, not the scientific held-out endpoint cohort.

Deeper analysis: `compatible_panels(evidence)` returns complete MonitorResult
records, retaining raw game rows, score/draw intervals, protocol, artifact digest
and opponent identity. Report paired independent-seed effects under the scientific
protocol before making a method claim; never pool games as independent fits.

## Comparisons

Default strength curves include only stage/update coordinates present exactly
once for every expected run in a compatible cohort. Protocol, opponent, world,
ABI and inference envelope must agree. Missing runs or unmatched milestones
produce unavailable panels. No interpolation, cherry-picked best checkpoint,
or unequal-latest ranking is performed. Points and intervals remain per seed.

The time axis is original checkpoint cumulative recorded training seconds. The
work axis is cumulative native environment decisions, explicitly a work proxy,
not FLOPs or hardware-normalized compute. Equal updates need not mean equal
transitions, exposures, seconds or dollars. Curves show these differences; they
do not assert an equal-cost treatment effect. Sparse single-point fixtures do
not establish learning trends.

For deeper analysis use `matched_milestones(evidence)` and inspect
`coordinates.optimizer_exposures` or `coordinates.learner_transitions`. Equal-cost
analysis needs overlapping observed support and a predeclared cutoff/selection
rule; do not extrapolate absent checkpoints. Keep regime-specific configuration,
independent seeds and held-out scientific cohorts attached to any conclusion.

## Costs

Elapsed wall seconds and additive worker-process seconds come from ExperimentRun.
Process seconds include concurrent learner and evaluator duration; evaluator
seconds are already included and must not be added again. They are occupied
process time, not CPU utilization or hardware-normalized compute. Host dollars
are available only when an hourly rate was configured; absent prices are unknown.

TrainingRun stage costs give collection, optimization, export and diagnostics.
Resource curves expose sampled process RSS in bytes; the summary uses MiB.
Sampling misses peaks and does not measure the whole host. Load averages measure
contention but do not correct throughput. Failures, interrupted attempts and
pending work remain visible; a missing cost record is not zero cost. Timing has
no statistical uncertainty estimate; repeat calibration for performance claims.

Deeper examples: `evidence.executions` retains attempt allowances, failures,
conservative crash charges and costs; `evidence.runs` retains per-stage costs,
imported producer references and artifacts. Add `progress/host_load_1m` or
`progress/process_cpu_seconds` through `metric_figure`. Do not sum ancestor costs
again when recovery receipts already charge them. Source preparation and external
costs absent from these receipts remain unavailable.
