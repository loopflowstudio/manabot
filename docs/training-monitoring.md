# TrainingRun dashboards

The training monitor projects existing TrainingRun diagnostics and arena receipts
into the `manabot` W&B project. VerifyStore remains the training authority;
`run.json`, `dashboard.json` and W&B are exports. No W&B account is needed to
train or retain diagnostics. These commands do not modify a running trainer,
its checkout, or its frozen recipe.

## New training and live dashboards

For an independently authorized step-target run, freeze a positive update interval:

```bash
uv run manabot train --regime ops/examples/step-target.json \
  --seed 197 --out .runs/example --checkpoint-updates 1000
```

A step is one completed self-play collection/learning iteration, including an
empty-filter skip. It is not an optimizer call or a fixed number of samples.
The interval is recorded as `TrainingRun.monitoring_checkpoint_updates`; every
exact absolute multiple exports a raw checkpoint. The final raw export owns the
endpoint, including an endpoint on that grid. Exports preserve Torch RNG and do
not reset Adam or the collector. Export failures retain rejected bytes and an
explicit missing milestone; they do not invent a replacement evaluation.

Portable continuation preserves the absolute offset: resuming at 26,000 with
interval 1,000 first exports at 27,000. It does not re-evaluate the inherited
boundary. Step continuation rejects `require_initial_admission`; that separate
initialization gate is for fresh runs. Exact CPU recovery requires the unchanged
cadence and retains the committed monitoring prefix and artifact bytes. The
existing evaluator deduplicates those artifacts across attempts/restarts, charges
failed attempts and never retries them silently. Retain every attempt directory.

`ExperimentSchedule`, `PlannedRun`, and deploy jobs default to 1,000 updates when
no interval is supplied. This is an authoring default, not an hourly estimate.
Calibrate throughput before launch, freeze the target and shared interval, then
keep them fixed regardless of observed runtime. Ataraxos keeps its absolute
iteration formulas; PPO step recipes select `schedule_clock="iteration_fraction"`.
Step cadence currently requires one ordinary step-target self-play stage.
Multi-stage recipes, active-time endpoints, supervised epochs and compound-game
training retain their explicit legacy cadence; their differing iteration/epoch
clocks are not silently combined into one step schedule.

Explicit `checkpoint_seconds` / `--checkpoint-seconds` preserves historical
elapsed-time cadence at update/epoch boundaries. Frozen launch JSON and historical
receipts retain their original meaning. Do not migrate live runs in place.
Monitoring allocation, leases, watchdogs, credentials, upload/freshness timers
and actual cost remain time-based; `JobSpec.checkpoint_seconds` is an export
reserve, not a monitoring interval. Stage-end raw/EMA exports remain available
when monitoring is disabled. Watchdog/resource budgets charge all export work.

Once `run.json` exists, launch the read-only dashboard follower separately:

```bash
uv run python -m manabot.training.monitoring \
  --run .runs/example/run.json --out .runs/example-dashboard --watch-seconds 30
```

Add `--online --entity TEAM` to upload using existing W&B credentials. Alternatively
read the authoritative store with `--store .runs/training.sqlite --run-id ID`.
Default operation writes local `dashboard.json` only. W&B network/auth failures
retain this file and a `delivery.json` failure type; they cannot stop training.
Run the same command later with `--online` to sync. This is JSON backfill, not an
assumption that the W&B SDK can resume an offline run.

Each TrainingRun has a stable W&B ID, grouped by resolved regime digest. Run
configuration retains regime/stages, seeds, resolved hyperparameters, hardware,
source/runtime/world identities. Stage summaries retain status, admitted and
rejected checkpoint digests, errors and actual device/thread counts. W&B resumes
at its acknowledged next history step. A changed published prefix or an older,
shorter export is rejected. Keep one publisher per run; the output directory has
a local writer lease. Recovery attempts retain separate run identities rather
than pretending an interrupted attempt never happened.

The default workspace includes ready-made `dashboard/` panels for teacher CE/KL,
RL objectives, regularization, entropy, retention, residuals and schedules.
Scalar sections also provide `elapsed/` copies against original training seconds,
`progress/` counters and `throughput/` rates. Missing metrics have explicit
`availability/` flags and summary explanations, not fabricated zeros. W&B's
[custom metric axes](https://docs.wandb.ai/ref/python/experiments/run/) and
[line-series charts](https://docs.wandb.ai/guides/track/log/plots/) own rendering.

## Reading the curves

* Distillation records epoch-average training cross-entropy and teacher KL in
  nats, growing-data validation CE/KL and fixed-cohort validation CE/KL. The first
  supervised stage freezes whole validation games and exact dataset bytes in
  `fixed-validation.npz`; its digest, source shards and game IDs are retained.
  Every later stage excludes those games from training and evaluates the same
  rows against the first stage's fixed reference targets. Later stages may change
  their training/growing-validation targets; those identities are labeled separately. TrainingRun's globally
  assigned game IDs are preserved when combining shards; generic historical
  shard loading keeps its original per-round offset behavior.
* RL policy loss is the clipped optimization objective, **not log loss**. PPO's
  value loss is half squared lambda-target residual; categorical move learning
  uses its own recorded value loss. Existing per-update loss, entropy and KL
  observations describe the last optimized minibatch (Ataraxos KL/entropy use
  its recorded post-update batch). Empty-filter updates label unavailable losses.
  Advantage retention, absolute advantages, lambda-target residuals, learning
  rate, reference regularization and collection-KL coefficient remain separate.
* Progress distinguishes learner iterations (or supervised epochs), native
  environment decisions, learner transitions, optimizer exposures, games and
  elapsed run cost. Teacher generation remains on the cost axis. RSS, process CPU
  time and host load are resource observations, not evidence of learning quality.
  Throughput is cumulative work divided by observed elapsed cost. Checkpoint age
  appears only when an earlier admitted monitoring export exists. W&B automatic
  system charts describe the publisher process; use the recorded progress fields
  for learner resource observations.

## Fixed-competitor checkpoint monitoring

Run evaluation in a separate process, explicitly accounting for its compute:

```bash
uv run python -m manabot.training.monitor_checkpoint \
  --run .runs/example/run.json --out .runs/example-monitor --watch-seconds 30 \
  --concurrent-activity 'training running concurrently' --online --entity TEAM
```

The follower evaluates each newly admitted monitoring export once, sequentially.
It preserves existing attempt directories on restart; stopped/incomplete attempts
are not automatically replaced. Saved results can be synced later. For an
individual export or a stage-end checkpoint:

```bash
uv run python -m manabot.training.monitor_checkpoint \
  --run .runs/example/run.json --checkpoint 0 --out .runs/checkpoint-0-monitor
uv run python -m manabot.training.monitor_checkpoint \
  --run .runs/example/run.json --stage policy-0 --out .runs/stage-0-monitor
```

Default monitoring is 100 games: 25 fixed deals (`1910101000`–`1910101024`), each
with four seat/deck combinations, versus source-pinned scripted greedy. Reserve
these deals for repeatedly inspected monitoring; exclude them from scientific
held-out cohorts. A custom `--protocol protocol.json` accepts MonitorProtocol
fields for bounded fixtures or an explicitly frozen monitoring cohort. No command
here authorizes scientific scoring or paid compute. Selected Allies/Lessons
checkpoint/setup admission is required; this runner does not relabel custom
training worlds as compatible.

Arena owns game execution, timeouts, legal Commands and exact replay. Each deal
block keeps `rows.json`, command traces and replay receipts. A failed game, missing
leg, admission failure or replay failure remains inspectable. Aggregate win/draw/
score rates are unavailable for incomplete cohorts. Complete cohorts report 95%
percentile bootstrap intervals resampling entire four-leg deals. These intervals
measure uncertainty conditional on a checkpoint; they do not quantify variation
across training seeds. The W&B monitoring stream has ready-made rate/interval
panels against update count and training seconds, separate from training history.

`monitor.json` records evaluation wall time, coordinator CPU/RSS, reaped-child CPU,
host load before/after and declared concurrent activity. Replay is included in
wall cost and retained separately in replay receipts. RSS is not native peak
memory; host load cannot identify contention or correct throughput for it. Arena
work does not advance the recorded training coordinates.

## Historical backfill

No retraining or new evaluation is needed to publish saved diagnostics:

```bash
uv run python -m manabot.training.monitoring \
  --run /path/to/saved/run.json --out .runs/historical-dashboard --online
uv run python -m manabot.training.monitoring \
  --evaluations /path/to/checkpoint-0/monitor.json /path/to/checkpoint-1/monitor.json \
  --out .runs/historical-monitor-dashboard --online
```

Supply evaluation attempts in immutable append order, including failures. Runs,
protocols and opponents must agree within a curve. Existing arena rows can be
imported against a saved MonitorResult manifest with original checkpoint,
coordinates, registrations and protocol:

```bash
uv run python -m manabot.training.monitor_checkpoint \
  --import-manifest /path/to/manifest.json --import-rows /path/to/rows.json \
  --out .runs/imported-monitor
```

Import preserves source path/digest, original training coordinates and recorded
costs, and validates every deal/leg and registration. It does not reload old model
bytes or claim a new replay. Historical diagnostics without per-update timestamps
retain their original stage/ordinal and are explicitly marked incomplete; final
stage duration is never spread over earlier updates. Missing fixed validation,
losses or resource observations cannot be recreated from aggregate results.
Historical schemas still need to be readable by the ordinary TrainingRun model.

## Experiment comparisons

New comparisons can schedule stage and periodic monitoring through the shared
[Experiment execution interface](experiment-execution.md). Its primary report is
one editable Jupyter report generator for learning, milestone, comparison and cost
plots in a read-only HTML dashboard. The existing evaluator and dashboard APIs remain the evidence projection
owners; trackers are optional. Frozen live campaigns are not retrofitted.

## S3 model and artifact storage

Jack Heart selected `s3://etudefantasia/manabot/` for retained bytes and
`loopflow-studio/etude` for W&B metrics on 2026-10-06. The bucket was created in
`us-west-2` with bucket-owner-enforced ownership, all public-access blocks,
AES256 default encryption and versioning enabled. No bucket creation, lifecycle
deletion, or public access is performed by the publication commands below.

Use the standard AWS credential chain (`AWS_PROFILE` or `--profile`); never put
keys in recipes or manifests. Publish a completed or stopped TrainingRun:

```bash
uv run --extra artifacts python -m manabot.training.artifacts publish \
  --run /path/to/run.json --destination s3://etudefantasia/manabot/ \
  --out .runs/storage/run-artifacts.json
uv run python -m manabot.training.monitoring \
  --run /path/to/run.json --artifact-manifest .runs/storage/run-artifacts.json \
  --out .runs/storage/dashboard --online --entity loopflow-studio --project etude
```

The publisher validates all retained stage exports, including rejected artifacts,
monitoring checkpoints, fixed validation and recovery snapshots. Role names retain
that distinction. Equal digests share a content-addressed S3 key. Exact `run.json`
bytes and the storage manifest are also stored in S3; the adjacent `.receipt.json`
pins the manifest's URI, digest, size and version when available. Original receipts
and absolute paths remain unchanged. This archives the named exports, not every
file under a run directory or a portable process-recovery environment. Referenced
external datasets and ancestor runs need their own publication.

Publication is a separate explicit operation: it does not delay an optimizer or
retrofit live campaigns. Interrupted uploads can leave unreferenced blobs; retry
verifies existing bytes instead of replacing objects. Local files are retained.
The implementation uses conditional single PUT with SHA-256 and full readback;
artifacts over 5 GiB fail explicitly. Transfer costs are separate from historical
training costs. Bucket encryption and access policies are managed outside this CLI.

W&B receives only the manifest metadata through the run summary. Model, optimizer
and dataset bytes stay out of W&B. Its chart tables remain part of visualization.
The older trainer also stops uploading model artifacts; its existing local save
and historical W&B reader remain intact.

Fetch a named role from the manifest (use its exact `artifacts` key):

```bash
uv run --extra artifacts python -m manabot.training.artifacts fetch \
  --manifest .runs/storage/run-artifacts.json \
  --artifact stages/policy-0/artifacts/raw --cache .runs/artifact-cache
```

This prints a verified local path for ordinary checkpoint consumers. Python callers
can pass a manifest's `StoredArtifact` directly to `load_checkpoint_agent`; it uses
`MANABOT_ARTIFACT_CACHE` (default `.runs/artifact-cache`) and the default AWS credential
chain. Downloads install atomically, and cache hits are rehashed before loading.
Corruption fails closed. S3 locations do not relax world, architecture, setup or
belief admission. Only load manifests/checkpoints from trusted producers; digest
integrity does not make arbitrary Torch serialization safe.

The first live backfill archived the six retained history runs: 36 model/optimizer
artifacts plus six exact producer exports and six manifests, totaling 48 objects
and 93,001,518 bytes. All uploads passed full readback; every reference includes
an S3 version ID. One separately downloaded trained checkpoint passed the ordinary
loader without training or games. All six W&B training runs expose matching S3
metadata; scientific scores remain in the separate scientific report. Local
receipts are under `.runs/etu117-demo/s3/` in the reporting checkout.

## Numerical health

Ordinary self-play now retains stable collection logs, bounded private failure
state and sampled gradient/parameter effects. See [numerical health](numerical-health.md)
for metric populations, expected zeros, overhead and bounded CUDA admission.
