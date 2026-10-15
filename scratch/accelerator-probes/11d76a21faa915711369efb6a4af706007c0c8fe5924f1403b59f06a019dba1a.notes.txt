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
4. Physical CUDA collection, ordinary TrainingRun and serving probes validate
   selected directions. Original model-only serving predictions failed; adding
   measured preparation/transfers passed a fresh H100 check: all nine median
   errors<=9.1%, correct best batches. See gpu-validation.md. The full-service
   model is selectable in both high-scale and learner CLIs with `--serving-costs`.

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

Request timing preserves exact replay and collection wall time, including final
tails. Seed10831/10832 repeat1 took0.25064/0.23346s, with36.793/38.185ms tails.
Full intervals and instrumentation controls remain in`request-timing-summary.json`.

Current L4 service comes from`gpu-probes/l4-002/capacity-report.json`, float32,
TF32off,202attention slots. Historical203slot reports remain separate. The small
model's median batch64/256/1024 rates were15,071/26,316/28,064 observations/s.
Actual whole collection is much slower; see GPU evidence for its boundaries.

At4096virtual actors/512collector CPUs/64servers, measured intervals preserve the
preregistered aligned/rotated batching reversal; modeled rates change at most1.08%
from proportional intervals. Full historical rates remain in archived notes.
This timing comparison does not validate the physical serving boundary.

The independent seed10832 64actor sensitivity matrix also completed. At batch256,
doubling collector speed moved22,172→22,244decisions/s(0.33%). Reserving one
model-only diagnostic Adam observation per decision lowered the original speed
to3,777/s and raised mean request latency66.5→395.2ms. This reserve still lacks
an actual learner queue, sample reuse and policy-age feedback.

See gpu-validation.md for physical catalog-reuse results and parity limits.

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
supervision. The original probe phase closed at238.718of240seconds;
its original600second setup cap is separate. The accelerator-specific300second
measurement/180second focused-test limits remain unexhausted. The new20second
calibration-analysis slice used19.589804seconds; conservative combined usage283.973775/300. New measured
work needs an explicit protocol and adequate remaining allowance. The GPU ledger
records$3.707791 billed/estimated rentals,$5.11 conservative charges+$5reserve,
$39.89 free; all eleven machines are confirmed deleted.

## Limits and next discriminating work

CPU replay normalizes/samples on CPU, unlike production CUDA. Coalesced transfers,
independent GPU links and reserved server cores are assumptions; network,
all-reduce, target CPU contention and shared-memory interference are unmodeled.
The earlier64-step trace retains151,443,136array bytes/actor(577.7GiB at4096), excluding native
state, Python objects, staging, models and allocator slack. Finite startup/drain
and synthetic phase rotations must remain visible. Exact replay of one policy
does not predict how a changed trained policy changes its game distribution.

Full serving integration completed in2.570s; the4096actor/64server sweep took3.506s,
~302MB laptop RSS. Exact commands and results are in`serving-integration-results.txt`.
Paired H100 staging passed:1.234×/1.421× gains at128actors, exact serving witnesses;
see`staging-validation-results.txt`. Ordinary collection did not pass: seed-dependent
gains, one5.13% regression and two4096stream timeouts. See`collector-staging-results.txt`.
Run010 localized a25s reference timeout inside rollout assembly; completed assembly
costs varied .863–7.194s, largely in system CPU. Native stepping was41–45% in completed
10350 GPU collections. See`collector-phase-results.txt`; no speedup ratio follows.
Run011 separated assembly costs: allocation5–8ms, copy .704–1.060s. Disabling process
huge pages lost both seed comparisons; no pressure stall recurred. Keep default
policy; see`collector-assembly-results.txt`. Next: actual1/2/4-worker replay scaling,
then learning/mature-policy traces and backpressure. Multi-GPU rates remain conditional.
Reference/reference controls reproduce tiny Adam differences without caching.
Optimize host code by paired semantic replay; change GPU
service only with matching hardware measurements; require actual collection and
learning before a system-speed claim. The active goal is not complete.

Raw evidence and source snapshots are indexed by`accelerator-probes/artifact-manifest.json`.
Complete previous notes and exact historical/current source snapshots are retained
by hash in that manifest; no old result was overwritten.

See `learner-flow.md` for the measured central learner and zero-lag cohort model,
its explicit hardware/service selection, memory accounting and proof limits. It
remains a stationary scheduling control, not distributed training.
