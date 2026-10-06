# Capacity: early progress and a bounded terminal allocation

ETU-103 software preparation, 2026-10-05. The original three-capacity WDL
study below remains proposed and unexecuted. Jack Heart subsequently authorized
the separate mini depth screen recorded here.
No default-model or promotion-model decision is supported yet. Keep the existing
default unchanged until this comparison supplies evidence. ETU-91 and the frozen
ETU-106 value-token checkout are untouched; their evidence is not capacity data.

## Authorized mini depth screen (2026-10-06)

Jack Heart selected capacity as the first self-contained mini experiment,
superseding the filter-scope selection. ETU-105's worker was interrupted before
mini training. The laptop history campaign in ETU-106 stays untouched. This
allocation is eight hours on `jack@100.96.227.95`, checkout `/Users/jack/src/etude`;
no laptop training, paid compute, retry, evidence replacement or Task completion.

`depth-screen` reuses the greedy-only screen supervisor, TrainingRegime,
TrainingRun, EvaluationProtocol and replayed arena. It compares `depth-1` and
`depth-2`: width 64, heads 4, scalar value-token, no history. Everything except
attention depth and recipe ID is identical, including authored Allies/Lessons,
semantic input, float32 CPU/one thread, current self-play, Adam, actor/critic
quantile filtering, four streams × 64 transitions and the move-learning recipe.
This deliberately departs from the older WDL three-capacity proposal. Extra depth
changes both policy and value representations. Paired seeds do not imply shared
weight bytes, identical training trajectories, or equal inference cost.

The supervisor charges preflight/calibration ≤1,800 s, training ≤21,600 s,
evaluation ≤4,500 s and reporting ≤900 s to one monotonic 28,800 s ceiling.
Calibration seed 10340 runs 40 updates per arm, split 20/20, with raw/EMA reload
and 400 s per-process / 190 s per-stage caps. Both timing receipts must complete.
No score enters allocation: rate is the maximum of process seconds/40 and each
stage seconds/20 across both arms. The shared terminal count is
`min(1600, 100 * floor(21600 / (6 * 1.25 * rate * 100)))`; below 400 stops the
attempt. Three paired training seeds 10341/10342/10343 use alternating orders
1→2, 2→1, 1→2. Each run continues Adam/collector/EMA across two equal halves,
with 3,600 s process and 1,790 s half-stage caps. Fixed count is the screen's
terminal allocation, not convergence or the full-capacity study endpoint.

Before scientific training the supervisor writes exact recipes, seeds, counts,
source/native/environment/input/world identities and timing receipts to
`resolved-plan.json` and `calibration.json`. Clean-source checks precede each
phase. A two-hour feasibility check and between-run timing checks preserve the
remaining phase reserves. Child process groups are killed on deadline/disk/failure.
The allocation never expands in response to scores or partial runs. All failed
attempts and bytes remain retained. The mini's existing native binary is reused
when native sources match; PR240/242 `.runs` evidence is preserved.

Raw midpoint and endpoint checkpoints each play 100 games per seed-arm versus
source-pinned greedy: common deals 963410–963434, all four seat/deck legs. The
1,200-game cohort uses paired seed/deal resampling with each four-leg block intact;
EMA exports remain unscored. Both checkpoints use these screening deals; there is
no untouched promotion cohort. All games require exact replay, ordinary legality
and viewer-boundary admission. Invalid or incomplete cohorts stop without a
comparative disposition. No score-dependent stopping or checkpoint selection.

Existing notebook/report output includes strength versus cumulative training
seconds (collection/export included), native decisions, learner transitions and
optimizer exposures. Common-support areas and contrasts never extrapolate; two
scheduled checkpoints cannot establish a detailed early learning curve or exact
time to useful progress. The descriptive effect is depth 2 minus depth 1. The
existing screen rule recommends a larger confirmatory allocation only with ≥5
percentage-point endpoint gain, no negative seed effect, positive common-cost
effect and ≤1.25× pooled inference latency. Consistent losses of ≥5 points reject
only this recipe/budget; other outcomes are unresolved. Neither decision promotes
a default model. Bootstrap seed 10634 and 10,000 draws are analysis constants.
Exposure comparisons are descriptive post-treatment accounting.

Launch the delivered clean source using the shared supervisor:

```bash
uv run --extra dev --extra notebook python -m experiments.runners.run_history_input --study depth-screen --campaign COMMIT --out .runs/etu103-mini-depth-20261006-1
```

The detached launch retains stdout, parent/child PIDs, phase receipts and budget
in that output. `supervisor.json` owns total charged time including calibration,
training, evaluation and reporting; nested `study/study.json` records evaluation
only. Do not add those totals. TrainingRun retains sampled resource observations;
host load is observation, not a correction for contention. `study/report.md`,
`study/history-contrast.{json,md}` and the executed notebook retain the shared
report schema; arm labels identify the depth comparison. Launch/result status is
recorded separately below; this protocol is fixed before execution.

## Executable plans

```bash
uv run python -m experiments.runners.capacity_study --write-plan /tmp/capacity-plan.json
```

This exports an ordinary ResolvedStudy and adjacent `.provenance.json`. It does
not construct a model or execute a workload. The default is a 900-second,
one-seed workflow plan with two tiny checkpoints, not a scientific allocation.
The declaration uses ETU-109's pinned Ataraxos baseline and ETU-102's existing
64/1, 64/2, 128/2 ladder, four attention heads. Only width and depth vary:
semantic viewer inputs, value-token pooling, categorical WDL, action capacity,
move-learning rule, Adam, batch/stream sizes, self-play opponents and raw artifact
evaluation stay fixed. Layer/width changes affect the shared policy/value core.
No history, pooling or learning-rule feature is crossed into this comparison.

The existing executor is the only execution path. After separate authorization:

```bash
uv run python -m experiments.runners.run_training_regimes --study model-capacity --profile scientific --plan /path/capacity-plan.json --out /path/new-capacity-run
uv run python -m experiments.runners.run_training_regimes --report-only /path/retained-capacity-run
```

Report regeneration verifies retained protocol, recipe, run-export, checkpoint
and trace digests. It writes the existing metrics/report/uncertainty/notebook plus
`capacity-analysis.json` and three-axis plots. Install the notebook extra for
report execution. Failed attempts remain in the ordinary study and VerifyStore;
partial cohorts do not become a model-selection result. No alternate runner,
checkpoint adapter, score computation or W&B dashboard is introduced.

## Proposed calibration and scientific protocol

First reserve an uncontended laptop window with ETU-91's operator; no worker may
infer resource availability from host load. Run a separately authorized bounded
pilot using the same token baseline, one CPU thread, fixed native world and
source, recording actual model receipts, complete-run costs, sampled process-tree
RSS, host load and any interruptions. Existing concurrent historical-mean
capacity-calibration timings cannot calibrate these token arms. Suggested pilot
ceiling: 30 minutes total, including evaluation/recovery. Preserve every failure.
If even the pilot cannot fit, stop and revise before scoring.

Freeze conservative update counts per arm targeting **at most five minutes**
per unit from the pilot, including collection, learning and checkpoint export.
The authoring input is `CapacityWorkload` JSON: `updates_per_unit` (three positive
integers in ladder order), `calibration_path`, its `calibration_sha256`, and
`prior_campaign_seconds` including all earlier allocations/attempts. The digest
binds the reviewed calibration evidence; the exporter does not certify its
scientific adequacy. Export with `--workload /path/workload.json`. Retain that
file and the authoring receipt with the plan. Also supply `runtime_identities`
with the seven ordinary calibrated runtime/source digests, `projected_disk_bytes`
and `projected_evaluation_seconds`. The exporter checks that evaluation plus maximum
training and a 30-minute report/recovery reserve fit the 24-hour ceiling. Freeze code, runtime/world, content,
setup, ABI and storage projections before the separately authorized launch.

Proposed full allocation: three capacities × three independent training seeds
1031/1032/1033 × 12 calibrated units. Checkpoints follow 1, 2, 4 and 12 cumulative
units. Stage watchdogs are 300, 300, 600 and 2400 seconds; each run has a
3660-second total ceiling. **Full training means the final frozen update count**,
not convergence or exactly one elapsed hour. The common per-arm budget is about
one hour, with 60 seconds of setup allowance. Unused time is not filled with
extra updates. Slow arms may fail; do not replace them or increase limits after
seeing scores. The initial 20-minute progress window and final endpoint are
separate estimands. Early traces remain useful if the final comparison is negative.

The proposed study process/allocation cap is 24 laptop hours (roughly 9.15 hours
maximum training plus evaluation/report reserve), within the unchanged 168-hour
campaign ceiling after charging prior attempts. Calibration and storage/resource
projections must show the evaluation fits before allocation approval; otherwise
reduce the proposed cohort before freezing or retain a bounded inconclusive pilot.
This document grants no allocation. No paid compute is proposed.

At each checkpoint, 25 common deal blocks × four seat/deck legs yield 100 games
per cell. All three fixed anchors (random, scripted greedy, PUCT-64) and paired
small-arm versus each larger arm comparisons reuse EvaluationProtocol. Untouched
endpoint deals use disjoint reserved families and paired training seeds. Both
seats and deck assignments stay together in uncertainty resampling. The existing
paired-seed/common-deal bootstrap supplies descriptive intervals; three seeds
remain a small cohort. One-thread, one-pass inference fixes the execution rule,
not equal latency or FLOPs across capacities. Record those costs; never claim
matched wall-clock inference strength without that qualification.

## Analysis and decision rule

The preregistered descriptive initial-progress threshold is score ≥0.60 against
the fixed random anchor within 1200 measured training seconds. First observed
crossings retain the preceding observation and first hit, not an interpolated
exact time. No hit is right-censored at the last observed checkpoint in that
window; no observation is unavailable. Sparse monitoring and score noise mean
these are scheduled observations, not confidence-certified or monotonic learning
crossings. Monitoring neither stops training nor selects held-out deals.

Curves show score against cumulative training wall time (including collection),
native environment decisions, and optimizer sample exposures. Repeated minibatch
samples are exposures, not decisions or optimizer steps. Each axis uses the
intersection of observed supports across all frozen arms/seeds. The last available
checkpoint supplies a step function; no pre-first-point backfill or post-last-point
extrapolation. Early mean score/area clips that shared time support at 1200 s.
Empty overlap and historical absent exposure counters remain unavailable. Failed
cohorts keep diagnostic crossings/terminal rows but suppress comparative curves.
Collection/export costs stay in measurements; study comparisons retain evaluation
and replay costs, with replay a subset. TrainingRun retains identities, phase costs,
resource observations and all attempts. Report total actual charged study time,
not only optimizer duration, when interpreting the later experiment.

Terminal observations use only the last checkpoint on untouched endpoint deals;
monitoring scores cannot replace endpoint evidence. Compare complete endpoint
cohorts with seed/deal uncertainty and actual training/inference costs. Prefer a
fast experimental model only when early progress and common-support area agree
across seeds without a material competence/legality regression; name a promotion
model only from terminal evidence. Conflicting estimates or intervals compatible
with no useful difference mean inconclusive, not a throughput-based winner.
Repeat a promising or important-negative learning idea at the next capacity in
a separately frozen follow-up, leaving its architecture-feature owner unchanged.
No such follow-up or empirical default selection has run in this software pass.
