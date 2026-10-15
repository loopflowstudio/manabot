# Laptop benchmark for accelerator-scale decisions

Jack Heart's active goal is a laptop benchmark representing the conditions that
matter at high GPU scale, followed by continued optimization. This is a working
research instrument, not a production distributed learner. The separately
accepted$50 GPU budget and actual measurements are in[gpu-validation.md](gpu-validation.md).
No PR, merge, Task operation or other experiment is selected by this work.

## System model

1. `experiments/runners/pipeline_benchmark.py` captures a real frozen policy and
   replays the ordinary collector with recorded outputs. It checks exact acting
   inputs, rollouts, journals, final buffers and reset seeds. This exposes the
   CPU/rules/storage work when inference is fast, without replacing game rules.
2. `manabot/infra/accelerator_model.py` gives conditional resource bounds from
   host demand and measured device-resident service points. Failed/missing GPU
   cells are omitted; no extrapolation is substituted for unavailable evidence.
3. `scratch/accelerator-queue-probe.py` simulates finite actors, collector CPUs,
   GPU inference servers, FIFO batching/timeouts, padding and serialized links.
   Each actor waits for its request before continuing. `--request-timings` uses
   actual request intervals and final host tails; default proportional timing
   remains a control. Explicit `--tape`, `--replay`, `--hardware` select evidence.
4. Physical CUDA collection and ordinary TrainingRun probes validate selected
   directions. Neither replay nor queue output is a playing-strength claim.

The queue requires one additional inference-serving CPU thread per GPU: measured
synchronized model calls include host dispatch/validation.512collector CPUs plus
64GPU servers therefore assume576 CPU thread reservations. Service curves are
not pure kernel timings or a guarantee for a different host/GPU pair. New queue
outputs call this inference-server busy time; older device-busy labels also
include host dispatch and must not be read as measured GPU utilization.

## Current evidence

The original64stream x64step tape(seed10831) has8,888decisions,4,096learner
transitions,31games and491policy requests. The second independent seed10832 has
8,371decisions,29games and358requests. Both pass exact audited and timed replay,
with no truncations. They are untrained policies; phases rotated within one tape
are synthetic sensitivity, not independent game trajectories.

Request timing preserves exact replay and reconstructs collection wall time.
Seed10831's declared repeat1 took0.25064s, with36.793ms final host tail(14.68%);
seed10832 took0.23346s with38.185ms tail. Tails include last-response processing,
finalization and assembly, not isolated assembly cost. Three sequential repetitions
with/without instrumentation show no obvious penalty, not precise zero overhead.
Full intervals and controls are in`request-timing-summary.json` and raw replays.

Current L4 service comes from`gpu-probes/l4-002/capacity-report.json`, float32,
TF32off,202attention slots. Historical203slot reports remain separate. The small
model's median batch64/256/1024 rates were15,071/26,316/28,064 observations/s.
Actual whole collection is much slower; see GPU evidence for its boundaries.

At4096virtual actors/512collector CPUs/64GPU servers:

| Phase / batch | Proportional host timing | Measured intervals + tail |
|---|---:|---:|
|aligned /256|1.435M decisions/s|1.420M|
|aligned /1024|1.273M|1.277M|
|rotated /256|1.473M|1.467M|
|rotated /1024|1.630M|1.616M|

The preregistered batching reversal survives. Rate changes are at most1.08% in
these GPU-limited cases; this supports the simpler approximation for this tested
comparison, not for host bottlenecks generally. Measured total CPU work is
1026.605s across4096actors in every cell, including their final tails.

The independent seed10832 64actor sensitivity matrix also completed. At batch256,
doubling collector speed moved22,172→22,244decisions/s(0.33%). Reserving one
model-only diagnostic Adam observation per decision lowered the original speed
to3,777/s and raised mean request latency66.5→395.2ms. This reserve still lacks
an actual learner queue, sample reuse and policy-age feedback.

Physical catalog reuse improved collection median1.49x/1.25x at64/256streams.
See gpu-validation.md for exact trajectory/raw/EMA parity, the small Adam-state
discrepancy, costs and proof limits. This remains a research prototype.

## Reproduction

Use fresh output directories and the existing bounded CLI(one thread,90s maximum,
6GiB RSS guard). Capture at most64streams x64learner steps for this local protocol.

```bash
uv run --no-sync python -m experiments.runners.pipeline_benchmark capture --recipe scratch/cpu-probes/update-recipe.json --seed 10832 --streams 64 --transitions 64 --out scratch/accelerator-probes/capture-new
uv run --no-sync python -m experiments.runners.pipeline_benchmark replay --tape scratch/accelerator-probes/capture-new --request-timings --out scratch/accelerator-probes/replay-new
uv run --no-sync python scratch/accelerator-queue-probe.py host --tape scratch/accelerator-probes/capture-new/tape.json --replay scratch/accelerator-probes/replay-new/replay.json --hardware scratch/gpu-probes/l4-002/capacity-report.json --request-timings --out scratch/accelerator-probes/queue-new.json
```

Six focused tests passed; ten closed-form/conservation checks cover queue timing,
finite completion, padding and host tails. All measured attempts used bounded
supervision. The legacy wrapper has consumed238.718of240aggregate probe seconds;
its original600second setup cap is separate. The accelerator-specific300second
measurement/180second focused-test limits also remain unexhausted. New measured
work needs an explicit protocol and adequate remaining allowance. The GPU ledger
records$0.255941 billed/estimated rentals,$0.45 conservative charges+$5reserve,
$44.55 free; all four machines are confirmed deleted.

## Limits and next discriminating work

CPU replay normalizes/samples on CPU, unlike production CUDA. Coalesced transfers,
independent GPU links and reserved server cores are assumptions; network,
all-reduce, target CPU contention and shared-memory interference are unmodeled.
Retained arrays alone are151,443,136bytes/actor(577.7GiB at4096), excluding native
state, Python objects, staging, models and allocator slack. Finite startup/drain
and synthetic phase rotations must remain visible. Exact replay of one policy
does not predict how a changed trained policy changes its game distribution.

Next: mixed independent/mature-policy traces, actual learner/backpressure and
policy-age accounting, and coupling to measured learner service. Reference/reference
controls reproduce tiny Adam differences without caching (gpu-validation.md).
Optimize host code by paired semantic replay; change GPU
service only with matching hardware measurements; require actual collection and
learning before a system-speed claim. The active goal is not complete.

Raw evidence and source snapshots are indexed by`accelerator-probes/artifact-manifest.json`.
Complete previous notes and exact historical/current source snapshots are retained
by hash in that manifest; no old result was overwritten.

The learner model now couples the existing inference queue to measured8×64/8×512
central updates under a zero-lag cohort barrier. It counts idle actor capacity,
retained arrays, partial horizons and publication. The27case result reverses the
preferred inference batch between one and eight servers; extra actors beyond the
measured learner shape stay idle. See `learner-flow.md` for protocol, reproduction,
evidence and limits. This is a stationary scheduling control, not a distributed
training implementation.8×4096 whole-loop service remains unavailable.
