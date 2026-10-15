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

The next optimization target is inference serving. `--device cuda-local` gives
each persistent worker its own model on the same GPU, keeping the original request
shapes and exact-output checks. It is a candidate for comparison, not an accepted
speedup. The shared mode remains the baseline. `--profile --device cuda` records
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
