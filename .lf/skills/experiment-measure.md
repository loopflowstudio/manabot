---
requires: a training or evaluation speed question and its workloads
produces: a measured bottleneck and comparison plan in `scratch/plan.md`
action_style: procedural
---
Measure where training and evaluation spend time, then plan only a software speedup backed by numbers.

1. Read [training and evaluation performance](../../docs/workflows.md#training-and-evaluation-performance),
   `experiments/reports/training-update-time.md` and the relevant run
   receipts first. Separate observations from suspected causes. Stay on this
   Task's workload and model sizes—this is not a campaign and grants no
   spending authority. Frozen worlds, observation versions and recipes don't
   change. Never induce load.
2. Fix the comparison on one machine before running: `TrainingRun.host` (not
   just the GPU label), revisions, build, device, model size, precision,
   input capacities, vocabulary, recipe or checkpoint, seeds, warmup,
   included updates and repetitions. Interleave baseline and candidate runs
   to expose host drift. State whether the goal is one seed's progress or
   aggregate throughput; for packed runs measure both from overlapping
   post-warmup samples and matched solo ordinals.
3. Read the **Update time** Perf series and per-update rows. Split collection
   (stepping, encoding, preparation, inference, recording, assembly) from
   learner work, alongside inference calls and rows per call. Nested clocks
   don't add; across a recovery read per-update rows, not the last attempt's
   timers. Check `StageRecord.idle_stream_steps` before blaming small batches
   on serving (older receipts default it to zero).
4. Use the maintained tools:
   - Training: `scripts/bench_training_update.py` with `--recipe`, `--seed`,
     `--device` and separate `--json` paths (or the size-ladder command once
     documented in `docs/workflows.md`). Stop on a capacity failure; don't
     shrink the workload.
   - CUDA diagnostics: `--device cuda --cuda-counts PATH` (whole-run counts,
     not per phase) and `scripts/diagnose_training_transfers.py` for transfer
     probes; neither measures throughput.
   - Evaluation: `manabot.eval.serving_benchmark` with a declared
     `BenchmarkPlan` ([evaluation performance](../../docs/evaluation.md#performance-evidence-and-calibration)),
     keeping `performance.json`.
5. Keep behavior digests beside timings: raw and EMA weights, optimizer state
   and sampled games for deterministic training; matched decisions for
   evaluation; declared tolerances for nondeterminism. Precision, policy-lag
   or batch-composition changes need separate learning runs outside this
   workflow.
6. Write the bottleneck, slices, **Delete — do not maintain** list, focused
   checks and exact commands into `scratch/plan.md`, with receipt paths and
   limits beside each claim. Plan no slices if no real win is supported.
   Leave unavailable hardware as a deferred command; don't rent, substitute
   CPU for CUDA or claim a gain.
