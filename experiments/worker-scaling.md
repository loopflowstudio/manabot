# Laptop projections and actual collector scaling

These research instruments distinguish native game decisions, collected learner
rows, optimizer exposures and whole training time. A vector environment currently
steps games serially; increasing its stream count does not allocate CPU workers.
The new worker benchmark gives independent collectors to persistent processes.
It changes no production rules, policy or learning algorithm.

## Actual workers

```bash
uv run --no-sync python -m experiments.runners.worker_scaling --tape experiments/data/worker-scaling/tape --out .runs/workers-new --actors 64 --repeats 1 --seconds 14.5
```

The fixed workload is64actors x64streams x8learner steps. All actors exist before
timing; all32768learner rows remain resident until the completion barrier. One,
two and four workers perform identical work. `prepare`, `collect`, `verify` and
`release` receipts are separate; the supervisor accounts for the complete process
tree and kills its owned group on failure/deadline. Do not sum workers' wall times
as elapsed time. Summed RSS conservatively counts shared pages more than once.

On an M4 Max, the first retained screening sample measured:

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
