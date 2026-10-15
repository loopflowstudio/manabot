# GPU validation allocation — 2026-10-14

Jack Heart authorized **$50 total** GPU time to validate directions on an actual
machine. This replaces the earlier no-paid-compute boundary for this goal only.
It does not allocate money from other experiments or authorize changing them.

Question: do current model service curves and ordinary CUDA collection support
the laptop replay/queue model's bottleneck ranking? Predict before execution:
at width 64/depth 1, device-resident batch 256 throughput exceeds batch 64 by
at least 20%, but ordinary collection fails to reach 50% of the best model-only
decision throughput. These are exploratory systems predictions, not playing
strength claims. Three independent initial seeds are used for collection;
no trained-checkpoint representativeness or multi-GPU result is claimed.

First allocation: one RunPod L4 through the existing Etude client, independent
launcher database under `.runs/gpu-validation/launcher`, dedicated existing-bucket
prefix `etude/fantasia-gpu-validation-20261014`. Live API quote: L4 and A40 each
$0.59/hour; H100 SXM $3.99/hour. No existing pods observed. Recheck admission price.
One GPU maximum, $0.80/hour ceiling, job maximum 30 minutes, Etude's additional
setup allowance at most 15 minutes, maximum one infrastructure attempt, $1.50
job cap and $3 cohort reservation. Reserve another $5 of the total for storage,
retrieval and delayed billing; at most $42 remains unallocated after this job.
Use actual provider bills when available and retain conservative estimates until
cleanup is confirmed. Uncertain creation/deletion prevents another rental.

The stock image and Etude source package are pinned. Read credentials from
Doppler etude/prd only in-process; use its dedicated Etude journal issuer and
existing private bucket, no IAM or account changes. Worker self-deletion and
its independent deadline bound rental lifetime; poll actual rental state.
No GPU launch until source, protocol, command and job limits are frozen in git.

Reuse `cuda_performance` for eight fresh-process model cells: width64/depth1 at
batches 1,4,16,64,256,1024; width384/depth8 at 64,256. Preserve OOM/failure cells.
Each measures synchronized float32 inference and diagnostic Adam service;
TF32 disabled. Use actual collected observations, record their hash and shapes.
Measure ordinary collection at 16/64/256 streams, 4096 learner transitions per
sample, seeds 10831/10832/10833, cold and repeated instances with exact same-device
replay hashes. One separate profiler collection at 64 streams x16 transitions
attributes kernel/copy intervals; do not mix profiled times with throughput.
Two existing ordinary TrainingRun cells (batch512, streams16/64) measure actual
filtered learning, export and persistence. No strength evaluation or promotion.

Kill criteria: missing CUDA, wrong GPU, source/schema mismatch, nonfinite outputs,
or replay mismatch stops the affected probe. 90-second cell, 600-second total
measurement and 900-second dependency/build ceilings. Failed/OOM/timeout cells
are retained, never relabeled successful or silently replaced. Source and all
result hashes, hardware/software identity, commands, elapsed time and costs
are retained. Local CPU checks may validate schema/orchestration only.

Follow-up allocation requires inspecting these results and confirmed cleanup.
Keep the full $50 ceiling inclusive across all failed and successful attempts;
do not interpret one run's smaller cap as a fresh budget on retry.

## First allocation status

Protocol/driver commit `7f93210e2328003d2734c2f33dd358c8dbd191c1` was
committed before launch. Etude published only that explicit revision; other
untracked research artifacts were excluded. Client source is
`d04d4996e16ca064eab127da39cb9375c5a511c2` from the existing private client.
Run `49c55761a6d3`, cohort `l4-validation-001`, session `5c4dee556dac`, pod
`td9idyvn0cg2wg` was created at 2026-10-15 02:18:56.538 UTC for $0.59/hour.
Its worker started; provider status was RUNNING on the subsequent exact-ID read.

Launch and one observation command timed out delivering the 182,743,040-byte
uncompressed Git archive through Etude's 60-second single-request upload. The
run remains recorded. Recovery uses the same source object, commit, run and pod,
uploading the exact archive through multipart S3 followed by full SHA-256
readback. No replacement rental was requested. Credential values stay in process.
Local budget/allocation state is `.runs/gpu-validation/budget.json`; the local
launcher home is `.runs/gpu-validation/launcher`. The authenticated CLI adapter is
`.runs/gpu-validation/cli.py`, invoked through Doppler and uv with the existing
Etude client package. Resume observation of the exact run, never relaunch to
recover an uncertain response. Confirm deletion and settle cost before allocating
any follow-up.

First allocation ended before measurement: the bootstrap installed `dev`, but
the repository declares maturin in `play`. The ordinary build command therefore
failed before CUDA admission. Source readback was verified after 153.397 seconds;
the failure log is `.runs/gpu-validation/observed.log`. Etude reported about
$0.07 estimated. Exact provider lookup confirmed pod absence at
2026-10-15 02:27:57 UTC (`first-cleanup.json`); reserve $0.10 pending final billing.
This is a bootstrap failure, not negative performance evidence.

Second allocation amendment: add the existing locked `play` extra; preserve all
measurement cells and predictions. One new attempt, same $1.50 job/$3 cohort
ceilings, cohort `l4-validation-002`; all costs remain inside the original $50.
Upload the exact frozen archive to the same private prefix with multipart and
full SHA-256 readback before renting. A local transport adapter may copy that
verified object server-side to Etude's session-specific source key; bytes, Git
source identity, worker execution and cleanup remain owned by the existing
launcher. No changes to the Etude checkout or credential scope.

Second source commit `724da6e1ab87381c65d1e85ed54cb4176df25ed5` was prepared
and fully read back before renting (102.352 seconds). SHA-256 of its tar is
`b6714a4476a4b98de52b74d0fce0ee901c6059dc89ded3e44340b5eb8e50204b`.
Server-side copy to the new session key succeeded and matched size/hash metadata.
Run `a7df3b8d1a77`, session `de4b7799f228`, pod `lhu3ok84ordvqw` is the active
second attempt; worker start observed at 2026-10-15 02:34:42 UTC. Its observation
process is the exact run's `etude wait`, not a new submission. Use the CLI adapter
with `uv run --no-sync --with /Users/jack/src/etude/client --with boto3 python`
under Doppler etude/prd. Logs are retained locally as `observed-002.log`.

## Second allocation findings and next discriminating probe

All 53 manifest entries verified on retrieval. Both exact provider pod IDs are
absent; second machine estimated $0.08, total rental estimate $0.15. Reserve
$0.25 conservatively plus the separate $5 ancillary reserve. Raw small records
and failures are in `gpu-probes/l4-002`; the full archive is retained under
`.runs/gpu-validation/result-002` (archive SHA in the portable summary).

Width64 inference median rates at batches1/4/16/64/256/1024 were
314/1,314/5,151/15,071/26,316/28,064 observations/s. Batch256 versus64 is1.75x,
meeting the preregistered20% prediction. Width384/depth8 is942/934 at64/256:
larger batching is not universally helpful. Repeated whole collection across
three seeds delivered972–1,210 decisions/s at16streams,2,269–2,707 at64, and
3,785–4,970 at256. All nine exact same-seed rollout comparisons passed, with
no truncations. 256stream traces ended after16 learner steps and1–2 games;
these short trajectories are not steady mature-policy traffic.

The sole host profile interval is1.389588s, with0.207562s kernel/copy union
(14.94%),68,114 kernels and4,955 copies. Three same-named GPU annotations are
nested duplicate windows, not independent repetitions. CPU GRU inclusive time
was0.2447s over119forwards; launch runtime0.5246s, overlapping those operations.
Busy intervals do not measure SM occupancy. CPU catalog dominance must not be
transferred quantitatively to GPU. Current attention length is202, unlike the
historical203-slot curves, so old rates remain historical sensitivity evidence.

Both actual-learning cells failed before any learning because a Git archive
has no`.git`; `source_commit()` supports a verified`.manabot-source.json` bundle.
This is source admission failure, not a measured learning rate. Preserve both
failures and use the existing source-bundle contract on the next run.

Third allocation preregistration: same L4/pool/price and caps, cohort
`l4-validation-003`, maxoneattempt, no automatic retries. Compare reference and
research-only catalog reuse within a frozen-policy collection call. Three seeds,
64/256streams,4096learnertransitions, four paired measurements in ABBA/BAAB
order after one short warmup. Cache first computation is included; destroy it
before learning. Require exact rollout, journal and reset-seed equality and
unchanged parameter versions. Never supply cached values with gradients enabled.
Prediction: at least15% median collection gain at64streams, smaller relative gain
at256. Losing that prediction is valid evidence. This does not select a production
cache API or invalidate-on-update design.

Rerun ordinary three-update TrainingRun cells at16/64streams, reference and cached
collection, preserving actual learning/skip counters and comparing final tensor
states where available. Keep gradients on the original uncached path. The source
bundle binds the prepared Git archive's complete file hashes plus the freshly
built native module; verify before invoking the existing source owner. Setup900s,
measurement600s, eachchild90s, same$1.50job/$3cohort caps. One local small CPU
smoke checks reference/cache parity within the remaining local probe allowance;
it is not GPU performance evidence. No production code changes.

Preflight: the4stream x16step CPU paired smoke passed all four ABBA trajectory
hash comparisons in2.846commandseconds, sampled peak324.2MiB. This consumes the
existing local cap, not GPU budget. Ruff check/format passed. The GPU experiment
retains separate baseline/candidate processes for whole-training comparisons.
