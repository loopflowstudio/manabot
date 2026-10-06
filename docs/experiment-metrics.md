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

## Sampling

The default diagnostics are separate per regime and metric, with independent seeds
identified by color. Solid lines use a trailing arithmetic mean of **up to 25 saved
diagnostic records**, not 25 games, samples, or elapsed seconds. Faint lines are raw
values. The window restarts at missing values, missing update coordinates and stage
boundaries. Smoothing shows local variation; it is not a confidence interval or a
reconstruction of unrecorded minibatches. Raw scalar JSON is downloadable from HTML;
`metric_figure` and `evidence.metrics(run)` remain available to edited notebooks.

For Ataraxos move updates, each learner timestep is an optimized minibatch across
selected environments. The loop overwrites `loss`, `policy_loss`, `value_loss`,
`entropy`, `collection_kl` and `reference_kl` each time. The saved values therefore
come from the **last optimized timestep minibatch**, evaluated before that
minibatch's optimizer step. They are neither update-wide means nor evaluations on
fixed positions. The sampled positions and number of legal offers change over time.
Empty actor/critic filters skip optimization and retain no loss or entropy.

`policy_loss` is the clipped surrogate **plus** weighted collection/reference
reverse KL in this recipe. The unregularized surrogate was not saved separately;
the report does not subtract guessed penalties. The KL panels show unweighted
nats. Scalar value loss is MSE against lambda targets; categorical value loss is
cross-entropy against outcome-distribution lambda targets. Neither is an independent
terminal-outcome test. PPO and distillation retain their own objective semantics.

`rows` and `retained` cover the collection batch, unlike the last-minibatch losses.
The report totals these only where both are present, names the record count, and
separately counts explicit skips and records without a loss. Missing retention is
unavailable, never zero. Optimizer exposures can differ from selected rows because
recipes can reuse rows or select actor and critic samples differently. Stage/run
counters and diagnostic samples cannot reconstruct absent historical updates.

Entropy is `−sum(p log p)` over legal offers, in nats. A uniform binary distribution
has entropy ln(2), about 0.6931, but that value alone cannot establish two legal
actions: larger skewed distributions or averages of different supports can have
the same entropy. Normalizing by log(number of legal actions) requires the actual
sampled support sizes, which these historical diagnostics did not retain. Entropy
measures dispersion, not competence; a confident wrong policy can have low entropy.
See the [saved history investigation](evidence/history-report-learning-2026-10-06.md)
for measured counts and the remaining explanatory limit.

## Scientific studies and software demonstrations

`load_study` reads the existing `study/study.json` export as a scientific cohort,
separately from MonitorResult. The notebook explicitly selects this path; it does
not relabel monitoring games. `study_strength_figures` separates opponent, phase and
raw/EMA variant, requiring one complete cell per declared regime/seed at each shared
cutoff, matching world/inference identity and four-leg deals. It checks summary
scores against terminal arena rows. Incomplete cohorts have no comparative point.
Thin lines retain seed variation; black diamonds show means with the existing
training-seed/common-deal bootstrap's descriptive 95% intervals when at least three
seeds and a shared x coordinate exist. These are score fractions (win + half draw);
monitoring plots are explicitly win fractions. No line extrapolates to initialization.
Zero-work evaluations appear if retained and matched; otherwise baseline is unavailable.

Cost plots retain each seed's actual checkpoint time. An equal cutoff is not an
equal-cost effect; different information, parameter counts and optimizer exposures
can change both collection and learning cost. The saved experiment's cost analysis
owns common-window comparisons. Two sparse checkpoints cannot explain learning
speed. Scientific *development* cohorts remain distinct from held-out endpoints,
and both remain distinct from earlier single-seed monitoring.

Passing pipeline, artifact reload and exact-replay smoke checks shows software
execution. A positive control must separately demonstrate a predeclared behavioral
improvement under a suitable training/evaluation design. This reporting change
neither runs that study nor treats changing weights/loss/entropy as its substitute.
