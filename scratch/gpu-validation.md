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
