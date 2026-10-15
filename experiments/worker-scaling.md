# Laptop projections and actual collector scaling

These research instruments distinguish native game decisions, collected learner
rows, optimizer exposures and whole training time. A vector environment currently
steps games serially; increasing its stream count does not allocate CPU workers.
The new worker benchmark gives independent collectors to persistent processes.
It changes no production rules, policy or learning algorithm.

## Actual workers

```bash
uv run --no-sync python -m experiments.runners.worker_scaling --tape experiments/data/worker-scaling/tape-w6 --out .runs/workers-new --actors 64 --repeats 1 --seconds 14.5
```

The fixed workload is64actors x64streams x8learner steps. All actors exist before
timing; all32768learner rows remain resident until the completion barrier. One,
two and four workers perform identical work. `prepare`, `collect`, `verify` and
`release` receipts are separate; the supervisor accounts for the complete process
tree and kills its owned group on failure/deadline. Do not sum workers' wall times
as elapsed time. Summed RSS conservatively counts shared pages more than once.

On an M4 Max with the pre-sync w4 source, the first retained screening sample measured:

|Workers|Collection seconds|Decisions/s|Learner rows/s|Speedup|
|---|---:|---:|---:|---:|
|1|1.9641|35,811|16,683|1.00x|
|2|0.9927|70,850|33,007|1.98x|
|4|0.5164|136,203|63,454|3.80x|

All192actor results matched their tape. The whole attempt took9.848s; sampled
summed peak RSS4.278GB. This is one ordered sample with recorded neural outputs,
not real GPU or training throughput. All actors repeat one untrained trajectory;
load, memory and thermals are not isolated. Full receipts live in
[data/worker-scaling/laptop-screening](data/worker-scaling/laptop-screening).
This demonstrates useful coarse process parallelism in this workload. It does
not reverse the historical finding that tiny per-step thread tasks lost time.

Add `--device cuda` on a CUDA host to collect through a shared frozen GPU policy.
The controller captures a same-device tape, then serves real observations through
one mmap slot per worker. Workers block for actual outputs. Exact original batch
sizes are retained, so numerical kernel differences do not contaminate the worker
comparison. This currently serializes GPU requests; coalescing is a separate
experiment. Collection includes copies/IPC/service and CPU sampling. No learning
runs inside this worker curve.

For the complete bounded physical comparison, run:

```bash
uv run --no-sync python -m experiments.runners.worker_experiment --out "$ETUDE_OUTPUT_DIR/worker-evidence" --expected-gpu "NVIDIA GeForce RTX 5090"
```

The [prospective GPU protocol](protocols/gpu-worker-scaling.txt) defines limits,
repeats, parity and hardware economics. It includes ordinary training controls.
A successful command alone proves neither a strength gain nor a universal hardware
recommendation. Compare costs using the actual rental and whole-training receipts.

## Laptop queue and learner models

`pipeline_benchmark` captures or replays ordinary collection, validating acting
inputs, rollout arrays, final buffers, journals and reset seeds. `PreparedReplay`
separates lifecycle phases, retains arrays, and restores each actor's CPU sampling
state. Actors in one process must collect sequentially; use processes to overlap.

`accelerator_queue` models finite actors/CPU workers/GPU servers and measured
request timing; `cohort_flow` adds a synchronous central learner and publication
barrier. These are conditional projections, not physical distributed benchmarks.

```bash
uv run --no-sync python -m experiments.runners.accelerator_queue check
uv run --no-sync python -m experiments.runners.accelerator_queue scale --request-timings --serving-costs experiments/data/worker-scaling/h100-serving/service-model.json --out .runs/queue-new.json
uv run --no-sync python -m experiments.runners.cohort_flow --global-streams 4096 --serving-costs experiments/data/worker-scaling/h100-serving/service-model.json --out .runs/cohort-new.json
```

The retained H100 full-serving calibration passed a prospective nine-cell check
(all median timing errors<=9.1%, correct best-batch rankings). Its initial idealized
model-plus-link predecessor failed by1.35–1.90x. The model therefore includes host
preparation, transfer and validation; it assumes independent CPU/memory/link
capacity per GPU and reports extrapolated request sizes. The copied service record
only relocates its overhead-source path; the original record and evidence hashes
remain alongside it. No timing coefficient was refit during relocation.

The central-learner model predicts8874rawrows/s on one H100 versus16272 with64
servers at4096global streams. That is a conditional stationary result, with no
actual network/all-reduce or mature-policy validation. It must not be used as a
measured ceiling. Current physical ordinary three-update H100 evidence delivered
98304rows in22.481s (~4373rows/s), with different collection architecture.

Prior catalog reuse and serving-buffer gains were workload-dependent. Ordinary
collection staging failed its performance gates; a huge-page toggle made assembly
slower. These negative findings remain in the research history. Do not apply a
universal optimization multiplier to the model or erase stalled measurements.

PR submission synchronized upstream world w6. The current worker command uses
`data/worker-scaling/tape-w6`; the initial laptop curve and queue calibration
remain w4 historical evidence. The paid comparison captures w6 policies on each
GPU. No w4 timing coefficient is claimed to calibrate w6.

## Physical single-GPU results (w6)

Three repetitions per worker count, fixed 32,768 learner rows per case:

| Host / mode | 1 worker rows/s | 2 workers rows/s | 4 workers rows/s | 4-worker speedup |
|---|---:|---:|---:|---:|
| H100 host, recorded CPU outputs | 5,712 | 10,788 | 21,838 | 3.82x |
| RTX 5090, shared CUDA | 1,160 | 1,418 | 1,389 | 1.20x |
| H100, shared CUDA | 716 | 1,375 | 1,593 | 2.22x |

These are medians of collection wall times, with exact verification of all 576
actors in each completed mode. The Linux CPU result confirms that coarse process
parallelism works. The shared service occupies about 98% of four-worker wall time
on both hosts. Its interval includes host work and transfers; it is not a GPU
utilization measurement. Original request sizes are small (at most 64 rows), and
requests are serialized. These results do not establish a single-GPU ceiling.

Ordinary three-update training at 4,096 streams delivered 2,277 learner rows/s on
the RTX 5090 and 2,546 on H100. At actual rental rates of $1.19/h and $3.99/h,
respectively, these project to 6.89 million versus 2.30 million rows per rental
dollar, excluding setup. The RTX 5090 is the economic choice among these measured
hosts for this width-64, one-layer policy. Peak allocated GPU memory was about
10.2 GB on each. Short runs, different host CPUs, and untrained policies limit
extrapolation; this is neither a global hardware optimum nor a strength result.

The first RTX CPU replay failed the Mac tape's final parity check. H100 diagnostics
subsequently isolated its portability mismatch to the batch hash; acting inputs,
final buffers, journals and reset seeds matched. Fresh host-local CPU capture
passed every worker check. This does not establish which batch field differed or
its cause. The failed portability result remains in the receipts.

Run012 used source `1f86d136a4e071ac34a9449705bfc225297eb185`; run013 used
`a8574ddb056f6af54ef43b7e2d9e00303d8b5ba8` with named mismatch diagnostics and
host-local CPU capture. CUDA and ordinary training algorithms were unchanged.
Readable summaries, exact raw structured receipts, output manifests, retrieval
verification and deletion records are retained in
[data/worker-scaling/gpu-012](data/worker-scaling/gpu-012) and
[data/worker-scaling/gpu-013](data/worker-scaling/gpu-013). `receipts.tar.gz` preserves
original JSON/log files; model binaries and exchange slots remain in the verified
Etude outputs identified by the included lifecycle manifests.

## Inference serving optimization

The optimization target is inference serving. `--device cuda-local` gives
each persistent worker its own model on the same GPU, keeping the original request
shapes and exact-output checks. The ownership comparison below passed its screening
gate; it is not yet a production training change. `--profile --device cuda` records
separate synchronized transfer/forward attribution and a CPU/CUDA trace.

`worker_experiment --serving-ownership` interleaves three complete curves per
ownership strategy, then runs the separate profile and ordinary training controls.
See the [prospective protocol](protocols/inference-serving.txt). The shared service
is currently a benchmark implementation; ordinary TrainingRun uses its collector's
GPU model directly. An isolated ownership win does not prove a training gain.

The separate CPU-only RTX5090 run014 used a different host (Xeon Gold6530,
13.6-core quota) and measured 1,581 / 2,396 / 3,473 learner rows/s at 1/2/4workers,
a 2.20x four-worker gain. All576actor witnesses matched. Its sampled host load
was232–249 and timings varied substantially: one-worker16.17–22.81s, four-worker
6.79–13.19s. Load alone does not identify the cause. This CPU curve must not be
paired with run012's GPU curve as a same-host decomposition. It demonstrates why
GPU type alone cannot determine a training rental's throughput. Raw evidence is in
[data/worker-scaling/gpu-014](data/worker-scaling/gpu-014). All three rentals were
deleted; their receipts distinguish complete-rental costs from job-only estimates.

## Same-GPU ownership result (run016)

On one RTX 5090 / EPYC 7543 host with a 12.75-core CPU quota, three interleaved
trials per design measured these median collection rates. Each case collected
32,768 learner rows through 4,288 policy requests, preserving original shapes:

| Active workers | Shared model rows/s | Worker-local models rows/s | Local / shared |
|---|---:|---:|---:|
| 1 | 1,035 | 1,157 | 1.118x |
| 2 | 1,509 | 1,591 | 1.054x |
| 4 | 1,503 | 1,736 | 1.155x |

All 1,152 actor results matched across designs: row/decision counts, rollout and
buffer hashes, journals and reset seeds. The candidate passes the prospective
15% four-worker improvement gate and the one-worker non-regression gate, narrowly.
Four-worker shared times were 20.143 / 22.122 / 21.795 seconds; local times were
18.871 / 19.571 / 17.830. These three observations are not a population estimate.
Peak summed RSS was 6.76–6.98 GiB shared versus 9.88–10.09 GiB local. Local models
consume more host execution capacity: in the last four-worker trial, workers used
about 70.5 CPU-seconds during 17.83 seconds of wall time. Removing central dispatch
helps, but this is a modest gain and does not clear the entire serving bottleneck.

The separate synchronized profile points toward fixed work per request. At
1 / 16 / 32 / 64 input rows, median forward time was 7.130 / 6.995 / 6.973 / 7.166 ms.
H2D was 0.411–0.640 ms and D2H 0.101–0.108 ms. Inside each forward, the inclusive
catalog host span was 2.91–3.03 ms; the trace contained 15 scalar reads and three
equality checks. Traced kernel/copy/memset intervals occupied only 0.882–1.150 ms
of each forward window. These are profiler-instrumented initial observations,
not unprofiled throughput or SM utilization. Catalog spans include host scheduling
and synchronization; the figures do not assign every gap to a particular cause.

This favors reducing repeated per-request work and amortizing it through larger
inference batches as the next investigation. Merely adding model processes is
insufficient. Catalog reuse alone remains unproven for whole training: its earlier
w4 large-batch benefit was only 0.85%. Preserve that contrary evidence when testing
the current w6 path. Admission checks, policy-version boundaries, sampling, and
gradient-enabled learning must remain correct when changing serving execution.

Unchanged ordinary training controls on this host reached 1,655 rows/s at 512
streams and 3,454 rows/s at 4,096 streams (98,304 rows in 28.457 s). The latter
projects to 10.45 million rows per rental dollar at $1.19/h, excluding setup.
It is a host control, not a gain caused by worker-local inference. Its collection
took 22.219 s and learning 4.269 s; neither worker architecture is integrated into
this training path. Different trajectories and request shapes also prevent treating
its rate as a matched comparison with the worker benchmark.

Run `45a304b45d64` used published source
`91b42f4677ee64a2ddfeb4f3861cdd7619cdfff6`. All 757 uploaded outputs and 756 manifest
entries were verified; raw profile trace, timings and structured receipts are in
[data/worker-scaling/gpu-016](data/worker-scaling/gpu-016). The provider confirmed
pod deletion; estimated whole rental cost was $0.477. Admission015 failed before
any experiment started, and the provider inventory found no pod. Its reservation
remains conservatively held while cleanup observation continues; run016 was an
explicit replacement admission with the same source/protocol, not a measurement
retry. No training speedup or playing-strength improvement is claimed.

The next candidate is now implemented in ordinary collection: `PolicyInference`
admits definition references on CPU and reuses semantic catalog embeddings during
one unchanged behavior-policy scope. It preserves original request shapes and
sampling. Normal learning retains fresh differentiable catalog computation.
`worker_experiment --serving-preparation` compares this path against ordinary
tensor inputs, with paired complete TrainingRuns and exact raw/EMA/Adam witnesses.
See the [prospective preparation protocol](protocols/inference-preparation.txt).
The candidate passes local forward, gradient, rollout, recovery and whole-training
parity tests. The GPU comparison below rejects adoption for ordinary training:
its complete-training speed target and strict learned-state parity gate failed.

## Scoped preparation result (run017)

On one RTX 5090 / EPYC 7543 host with a 14.375-core CPU quota, six interleaved
trials compared ordinary inputs with scoped preparation. All 18 worker cases
completed and all 1,152 actor results matched exactly across both modes.

| Active workers | Reference rows/s | Prepared rows/s |
|---|---:|---:|
| 1 | 944 | 1,428 |
| 2 | 1,203 | 2,150 |
| 4 | 1,238 | 2,283 |

Four-worker serving improved **1.844x**, passing its 1.5x timing target. Median
collection duration fell from 26.465 to 14.353 seconds. Peak summed process RSS
remained below 7.0 GiB. This is a measured small-request serving benefit, not a
complete-training benefit or a single-GPU throughput ceiling.

All twelve ordinary three-update TrainingRuns completed, but their medians were:

| Streams | Reference seconds | Prepared seconds | Reference rows/s | Prepared rows/s |
|---|---:|---:|---:|---:|
| 512 | 6.282 | 6.324 | 1,956 | 1,943 |
| 4,096 | 24.601 | 24.401 | 3,996 | 4,029 |

The large case improved only **0.82%**, missing the prospective 10% target; the
small case took 0.68% longer. All samples remain included, including the first
512-stream reference run at 8.820 seconds. Reference large-case times were
23.510 / 24.601 / 25.136 seconds; prepared times were 23.963 / 24.671 / 24.401.
Three timing repeats cannot distinguish this small change from run variation.

The strict raw/EMA/Adam/diagnostic parity gate also failed. At 4,096 streams,
all raw and EMA parameters and scientific diagnostics matched exactly; one
prepared repetition differed in one Adam moment element by 3.55e-15. At 512
streams, the first reference repetition differed from later references in raw
parameters (maximum absolute difference 2.38e-7), EMA, Adam state and diagnostics.
One prepared repetition also had four differing Adam elements (maximum 1.82e-12)
despite matching raw/EMA parameters. This establishes variation within a mode;
it does not identify its cause or prove the intervention harmless. The original
exact gate remains failed, with no relaxed tolerance or excluded repetition.

The decision is to retain scoped preparation as an experimental serving result,
not accept the production collector change on these results. Small-request
serving and ordinary large-batch collection need separate bottleneck models.
The next discriminating measurement is phase attribution of ordinary collection
at the actual 512/4,096-stream shapes, including inference, observation assembly,
transfers, sampling and synchronization. Existing service traces do not answer
that question. Larger requests or coalescing remain possible interventions,
not measured training wins. No simulation rewrite is justified by this result.

Run `71bfad424b47` used published source
`61e7149a707940e56eece42836cd912a6be4247e`. The complete 32.6 MB output archive and
all 783 internal manifest files were verified. Structured raw receipts, the
post-hoc state-difference audit and all timing samples are retained in
[data/worker-scaling/gpu-017](data/worker-scaling/gpu-017). The provider confirmed
deletion; estimated whole rental cost was $0.307. This does not establish playing
strength or clear the full single-GPU training bottleneck.

## Ordinary collection attribution (run018)

The [attribution protocol](protocols/collection-attribution.txt) ran ten ordinary
TrainingRun cells on an RTX 5090 / Xeon Gold 6530 host with a 13.6-core quota.
All cells completed. Four controls had no profiler; four Torch profiles observed
only the second collection, after a real update; two separate cProfile runs
observed the same collection at 4,096 streams. These are diagnostic observations,
not replicated throughput estimates. The one large unprofiled reference/prepared
pair took 28.463 / 28.740 seconds for 98,304 learner rows, including 22.448 /
22.670 seconds of collection. It supplies no new training-speedup claim.

The aligned Torch trace measured these host windows at 4,096 streams:

| Second-collection phase | Reference seconds | Prepared seconds |
|---|---:|---:|
| Complete collection | 8.922 | 8.132 |
| Native stepping and observation encoding | 4.241 | 4.073 |
| Input slicing, preparation and transfer | 0.945 | 1.239 |
| Model forward | 1.369 | 0.616 |
| Policy log processing | 0.306 | 0.308 |
| Sampling | 0.046 | 0.045 |
| Final batch assembly | 0.991 | 0.894 |
| Unmarked host intervals | 1.022 | 0.956 |

These windows include host scheduling and synchronization. Semantic catalog
encoding is nested inside forward: 199 calls / 0.629 seconds in reference versus
one call / 0.003 seconds in prepared. It must not be added to the table's forward
total. Traced device kernel/copy/memset union occupied 0.718 / 0.579 seconds of
the complete collection, respectively; that is activity time, not SM utilization.
Both modes made 199 policy calls for 67,275 acting rows. Of those calls, 142 had
at most 64 rows despite 4,096 resident streams; the largest call had 4,096 rows.
Resident stream count therefore does not describe every actual inference batch.

Independent cProfile observations agree on the large native boundary (4.099 /
4.098 seconds) and batch assembly (0.922 / 0.944 inclusive seconds), inside
7.643 / 7.910 seconds of profiled collection. Prepared inputs also add CPU
admission and repeated policy-version inspection: their input helper took 1.340
seconds versus 0.839 in reference. These are separately instrumented observations,
not coefficients that can be subtracted from the unprofiled controls.

At 512 streams the traced reference/prepared collections took 1.459 / 1.167
seconds, with forward 0.517 / 0.250 and native stepping/encoding 0.324 / 0.294.
This reinforces the workload dependence. At 4,096 streams, the optimized model
forward host window is about 8% of collection, while native stepping/encoding is
about half. Input materialization, transfer, batch assembly and other host work
remain substantial. Improving how CPU producers feed a GPU service is therefore
more promising than another model-only serving change. This is a design direction,
not evidence that a parallel training pipeline is already implemented.

The native boundary is not pure rules simulation. It includes stepping, observation
construction/encoding and bindings. Source inspection of `VectorEnv::step_into`
also shows active rows are skipped before stepping; these measurements do not
support a claim that all inactive games are simulated. A new pipeline must preserve
stream/seed identities, sampling order or explicitly admitted numerical differences,
policy-version barriers and ordinary checkpoint recovery. Existing worker benchmark
parity is insufficient to establish those production contracts.

All initial/raw/EMA hashes matched across controls and profile modes at each
stream count. Adam state had two distinct hashes at each count, so exact learned
state remains unproven and run017's failed gate is unchanged. The full 55 MB output
archive and all 137 manifest files were verified; raw traces, profiles, receipts
and summaries are in [data/worker-scaling/gpu-018](data/worker-scaling/gpu-018).
Run `2953f94c4d13` used source `3afa1ac26f277f70dea997e288c28dcf3b77e66f`.
The provider confirmed deletion; estimated whole rental cost was $0.139.

## Persistent production producers (run019)

The [producer protocol](protocols/producer-scaling.txt) measured the ordinary
TrainingRun with persistent native CPU shards feeding the same GPU collector.
It preserved global inference batches, sampling order and reset identities.
The RTX 5090 host had an AMD EPYC 7543 CPU, a 14.375-core quota, PyTorch2.10/cu128
and driver570.153.02. This differs from run018's host; comparisons below use
only run019's matched conditions.

Thirty of34cells completed. Four hit the unchanged training wall deadline:
the large deterministic proofs with two workers, four workers and the final
reference, plus the first large ordinary reference. All prepared ordinary
timing cells completed. Their complete-training medians were:

| Streams | Producers | Seconds | Learner rows/s | Speedup vs prepared1 |
|---|---:|---:|---:|---:|
|512|1|7.285|1,687|1.000x|
|512|2|6.757|1,819|1.078x|
|512|4|6.699|1,834|1.088x|
|4,096|1|29.359|3,348|1.000x|
|4,096|2|26.806|3,667|1.095x|
|4,096|4|26.233|3,747|1.119x|

Each median includes all three declared repetitions; large prepared1 durations
were26.659–33.467s and prepared4 durations24.367–27.922s. The four-worker point
estimate falls below the20% target versus prepared1. The missing large reference
repetition prevents the complete25%-versus-reference comparison; no median of
surviving reference samples substitutes for it. At512streams reference median
was8.110s, so the candidate passed the small-run regression criterion. Summed
process CPU time increased from29.480s to39.527s at4,096streams, while collection
wall medians fell22.580→20.060s. That demonstrates additional CPU work with a
modest wall-time gain, without identifying its cause or establishing four-core
efficiency. These measurements do not clear the training bottleneck.

All five512-stream deterministic proof runs exactly matched full rollouts,
buffers, action/active journals, game seeds, sampling RNG, raw/EMA parameters,
Adam state and scientific diagnostics. At4,096streams every available collection
witness matched: five samples for each of the first two collections and three
for the third. Only the first reference and prepared1 completed all learning,
and their final states matched. Matching prefixes do not certify the unfinished
large proof runs. Diagnostic durations are excluded from throughput estimates.

Completed ordinary timing runs had identical raw/EMA parameters and scientific
diagnostics within each stream count, but Adam state varied. The512-stream
outlier differed in71moment elements, maximum absolute difference3.64e-12; at
4,096streams the two groups differed in one element by2.84e-14. Both single-worker
reference and prepared runs participate in the large-run variation. The
[post-hoc audit](data/worker-scaling/gpu-019/parity-audit.json) preserves exact
differences; it does not relax the strict gate or prove the cause. Deterministic
execution is promising diagnostic evidence, not a waiver of ordinary parity.

The implementation remains provisional. The native profiler is disabled on
this path and its lock is per environment, so source inspection does not support a
shared profiler-lock explanation. Host variability, producer overhead and serial
collection work remain distinct possibilities. The existing collector also
copies learner observations into per-row pending dictionaries, then copies them
again into the final batch through nested Python loops. Run018 measured substantial
batch assembly and input materialization costs. Direct rollout storage is a
concrete next data-flow intervention; its benefit is not yet measured.

Run `978556688139` used published source
`a07b3cd3794df658368918742ef25b661b28bb38`. The complete83MB archive and452manifest
entries were verified. All failed cells and timing repetitions remain in
[data/worker-scaling/gpu-019](data/worker-scaling/gpu-019). The provider confirmed
deletion; estimated whole rental cost was$0.380. The original$50authorization
remains unchanged, and no additional experiment was launched for this analysis.

## Direct rollout storage result (run020)

The [prospective storage protocol](protocols/rollout-storage.txt) compared the
verbatim frozen collector from a07b3cd3 against direct writes into final rollout
arrays. Both use prepared inference and the same native implementation. All32cells
completed: eight deterministic proofs and24ordinary timing cells. Each ordinary
condition below has all three declared repetitions, with fixed work of12,288 or
98,304learner rows. No failed or slow sample was discarded.

The host was an RTX5090 / AMD EPYC9554, with a31.13-core quota, approximately62GB
memory limit, driver610.57.04 and PyTorch2.10/cu128. This is a different host from019;
the comparisons below are all within020, not cross-host speedup claims.

| Streams | Storage | Producers | Training seconds | Learner rows/s |
|---|---|---:|---:|---:|
|512|frozen|1|5.308|2,315|
|512|direct|1|4.805|2,557|
|512|direct|4|4.830|2,544|
|512|frozen|4|5.189|2,368|
|4,096|frozen|1|23.084|4,258|
|4,096|direct|1|24.913|3,946|
|4,096|direct|4|23.690|4,150|
|4,096|frozen|4|24.935|3,942|

Direct4/frozen1 throughput is0.9744x, below the1.25x target; direct4/frozen4 is
1.0526x, below1.15x. The small-run duration ratio is0.9098, passing its1.05 ceiling.
Thus the candidate passes the small workload criterion but fails the large
whole-training performance gates. Direct1 also regresses against frozen1 at4,096.
Collection medians for large frozen1/direct1/direct4/frozen4 are
17.604/19.115/18.140/19.573s; summed process CPU time is
23.256/25.077/37.819/39.019s. Removing intermediate copies did not reliably improve
the measured large workload, and adding producers did not help the frozen baseline
on this host. Neither result identifies the underlying allocation/cache/scheduling
cause. Three repetitions of one seed and recipe do not establish general capacity.

All eight deterministic proofs match exactly within shape, including complete
rollouts, action/active journals, buffers, seeds, sampling RNG, raw/EMA/Adam state
and scientific diagnostics. All24ordinary timing runs have exact initial/raw/EMA
and scientific-diagnostic agreement within shape, but Adam has four distinct groups
at each shape. Differences also occur between frozen single-worker repetitions.
The [post-hoc audit](data/worker-scaling/gpu-020/parity-audit.json) retains every
distinct optimizer group against the first frozen1 baseline:512groups differ in
4/5/71moment elements, with maximum absolute difference7.28e-12;4096groups each
differ in one moment element, at most9.09e-13. No tolerance or acceptance gate
changed. Deterministic proof does not waive ordinary-state variation.

**Adoption is not accepted, and the single-GPU training goal remains incomplete.**
The direct-storage candidate has no demonstrated benefit for the large target
workload. A subsequent implementation should remove it as the production default
before measuring another intervention; its source and this negative result remain
frozen evidence. The unresolved numerical mechanism is a separate question from
the failed performance gate.

Run `a22592d08d8b` used source `e41edb26be24b0de9d68b46f9edaadfb28c6fcb6`.
The84,061,387-byte archive (SHA256
`be13accb98333705c4d2d943f4b3ddf87ab2ba570ebc1c466f707d125c9de905`) and all443manifest
entries were verified. Structured receipts and all cells are retained in
[gpu-020](data/worker-scaling/gpu-020); full model binaries remain in the immutable
Etude output. Provider deletion is confirmed. Estimated whole rental cost is$0.251;
the original$50 budget has about$6.31 estimated rental usage, with$31.68unallocated
after conservative charges and reserves. No new rental is active.
