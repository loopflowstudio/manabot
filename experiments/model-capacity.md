# Capacity: early progress and a bounded terminal allocation

ETU-103 software preparation, 2026-10-05. Jack Heart authorized software delivery
and synthetic checks; **all empirical work below is proposed and unexecuted**.
No default-model or promotion-model decision is supported yet. Keep the existing
default unchanged until this comparison supplies evidence. ETU-91 and the frozen
ETU-106 value-token checkout are untouched; their evidence is not capacity data.

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
