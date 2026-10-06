# History report: learning trends and evidence limits

Jack Heart requested readable reports and an investigation of repeated entropy
near ln(2). The reporting demonstration uses saved evidence only; it launches no training or
evaluation. The validation scope error described below is separate from that demonstration.
The shared notebook generator preserves existing files and writes read-only HTML.

## Retained result

The completed `etu106-history-input-continuation` study contains two evaluated
checkpoints (64,000 and 128,000 learner transitions) for history-off/on across
training seeds 10631–10633. Each scientific cell contains 100 complete games;
1,200 games have exact replay. Both checkpoints are labeled **development** in the
frozen export. No initialization evaluation is retained. This study does not
establish improvement over initialization or a positive learning control.

| Arm | Midpoint seed scores | Endpoint seed scores |
| --- | --- | --- |
| History off | 39%, 26%, 26% | 26%, 26%, 28% |
| History on | 30%, 25%, 24% | 15%, 34%, 31% |

The saved on−off contrast is −4 points [−11.34, +1.33] at midpoint and 0 points
[−11, +9.33] at endpoint (paired training-seed/common-deal 95% percentile bootstrap).
The saved disposition is unresolved. Equal endpoints do not establish equivalence;
three seeds and one scripted opponent do not establish general strength. The
history-on arm also changes information and parameter count. The observed common
training-cost window is 735.31–1151.19 seconds; cutoff comparisons are not
matched-cost effects. Sparse checkpoints cannot resolve learning-speed dynamics.

The earlier `etu106-live-monitor` evidence has one training seed per arm and a
separate monitoring deal namespace/protocol. Its midpoint/endpoint records are shown on a separate dashboard; it also lacks
an evaluated initialization. Monitoring registrations have different observation
ABIs, so histories stay in separate per-run panels, without a cross-arm comparison. Monitoring is inspected development evidence, not independent method
uncertainty. The generator preserves the original coordinates and per-checkpoint
deal intervals without pooling either cohort.

## Entropy investigation

The six scientific run exports contain 3,000 update records and 2,965 entropy
values. **1,044 / 2,965** fall within 0.001 nat of ln(2). Counts by arm/seed:

| Arm / seed | Near ln(2) / entropy records | Empty-filter updates | Selected rows / collected rows |
| --- | ---: | ---: | ---: |
| off / 10631 | 121 / 496 | 4 | 14,103 / 128,000 |
| off / 10632 | 153 / 496 | 4 | 14,352 / 128,000 |
| off / 10633 | 238 / 492 | 8 | 10,958 / 128,000 |
| on / 10631 | 199 / 494 | 6 | 9,566 / 128,000 |
| on / 10632 | 171 / 496 | 4 | 19,379 / 128,000 |
| on / 10633 | 162 / 491 | 9 | 16,225 / 128,000 |

Measured explanation of the **instrument**: `ataraxos.update` overwrites entropy
and losses at each optimized timestep; the final retained value describes only
the last optimized minibatch. The 35 empty-filter updates retain no optimized
entropy/loss. Update-wide selection-group counts cannot identify the last
minibatch's legal support or its action mix.

Hypothesis about the **policy**: many last minibatches may contain nearly uniform
binary decisions. The observed numbers are consistent with this, but do not prove
it. The exports do not retain last-minibatch support sizes or probabilities;
no retrospective normalization or update-wide entropy can be recovered. Entropy
alone says nothing about whether the chosen actions win games. A future instrument
could retain sample-weighted update diagnostics and legal-support counts, but that
is separate work, not an invented historical aggregate.

## Reproduce the report without playing games

Source evidence remains read-only at:
`/Users/jack/src/etude.test-ataraxos-inspired-model-representations/.runs/etu106-history-input-continuation`.
The prototype/user-edited notebooks in
`/Users/jack/src/etude.agent-9039d61b/.runs/history-retrospective-report` are untouched.

```bash
uv run --extra notebook python -m experiments.runners.report_history \
  /Users/jack/src/etude.test-ataraxos-inspired-model-representations/.runs/etu106-history-input-continuation \
  .runs/etu117-demo \
  --monitoring /Users/jack/src/etude.test-ataraxos-inspired-model-representations/.runs/etu106-live-monitor
```

Open `scientific/comparison.html` first: independent seeds, descriptive uncertainty,
missing initialization, then optional per-regime diagnostics and raw downloads.
Open `monitoring/comparison.html` separately to review the earlier monitoring.
Each directory has its own editable `report-generator.ipynb`. Run All regenerates
HTML; creating or refreshing the report never replaces an existing notebook.
Changed source copies fail rather than silently replacing evidence. The output
`source-manifest.json` binds copied bytes to original paths/hashes.

The supervisor charged 10,945.66 seconds including the prior 91.16-second failed
calibration; the 2,888.45 evaluation seconds are already included. Run exports keep
collection, learning, export, diagnostics and sampled resources. The report does
not reinterpret these costs as normalized compute or retroactively create a
heartbeat for the completed legacy study.

Headless notebook execution and HTML structure/link validation are software checks.
No rendering environment was supplied, so visual presentation awaits Jack Heart's
review. Jack Heart subsequently accepted the W&B version and authorized shipping; see below.

A broader inherited pytest selection accidentally ran tiny training fixtures and
started a live monitoring attempt. It was interrupted after 18 checks passed;
that attempt has zero completed monitoring rows and is retained separately in
`.runs/etu117-interrupted-validation`. This exceeded the intended saved-data-only
validation scope. It supplies no scientific evidence and did not modify source
experiments. Subsequent checks select only offline reporting tests.

## W&B graph version

Jack Heart requested a new version using W&B graphs after configuring
`loopflow-studio/etude`. The native
[W&B report](https://wandb.ai/loopflow-studio/etude/reports/History-input-%E2%80%94-learning-and-evidence--VmlldzoxODA2NzM3Ng==)
leads with each arm's three scientific seed curves at matched transitions and
recorded cost, then the saved paired effect with explicit 95% bounds. Cost plots
do not imply matched-cost effects. Seven separate scientific projection streams
contain fourteen retained score/effect rows; they do not rewrite the six training
or two monitoring histories. API readback matched every projected scalar.

Collapsed per-arm diagnostics use native W&B EMA smoothing 0.8 with original
curves available; retention remains raw. W&B lines can bridge unavailable values,
so the report states that limitation and keeps empty-filter retention visible.
This differs from the earlier Matplotlib trailing-25-record view; neither
reconstructs historical minibatch averages. Earlier monitoring remains separately
labeled and filtered by exact run IDs. Default W&B aggregation is disabled;
saved statistical bounds are explicit series, not automatic standard-error bands.

The separate create-once notebook owns editable report choices. Run All only
reads saved data and writes `report-plan.json` plus a read-only HTML viewer;
network publication is an explicit step. The HTML embeds the native report with
a direct link if authentication or embedding prevents display. Project visibility
was not changed. Original notebooks and reports remain untouched.

```bash
uv run --extra notebook python -m experiments.runners.report_history_wandb \
  .runs/etu117-demo .runs/etu117-demo/wandb-report
# Run All in the new report-generator.ipynb, then publish its saved plan:
uv run --extra notebook python -m experiments.runners.report_history_wandb \
  .runs/etu117-demo .runs/etu117-demo/wandb-report --publish
```

Publication creates a new report, preserving manually edited prior reports.
Scientific streams reuse content-bound IDs and prefix checks. Model bytes remain
in S3; this operation uploads only metric projections and report configuration.
The reporting extra installs `wandb-workspaces`; its dependency updates W&B to
0.30.0. No training or evaluation ran. Headless screenshot capture returned a blank
W&B page, so browser rendering and interaction remain for Jack Heart's review;
successful API verification is not visual approval.


## Review and artifact delivery — 2026-10-06

Jack Heart responded “good enough” to the new W&B report and requested shipping.
This is product acceptance and landing authorization, not a claim of particular
browser interactions or new scientific evidence.

The private, versioned `etudefantasia` bucket in `us-west-2` holds six archived
history runs under `manabot/`: 48 objects / 93,001,518 bytes, including 36 artifact
blobs, exact producer exports and manifests. Full readback verified every upload;
all references pin version IDs. An independently downloaded raw checkpoint passed
ordinary world/schema/model admission. Six W&B training streams retain 500 history
rows each and matching S3 metadata. No model bytes were uploaded to W&B.
Receipts remain in `.runs/etu117-demo/s3/`, including bucket, roundtrip and W&B
reference verification. Publication remains explicit; originals are preserved.
