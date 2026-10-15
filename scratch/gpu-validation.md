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
