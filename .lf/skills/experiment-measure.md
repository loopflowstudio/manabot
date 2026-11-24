---
requires: a training or evaluation speed question and its workloads
produces: the measured bottleneck and comparison plan in `scratch/plan.md`
---
Measure what training and evaluation spend time on, then choose a software improvement worth making.

1. Read [training and evaluation performance](../../docs/workflows.md#training-and-evaluation-performance),
   `experiments/reports/training-update-time.md` and the retained receipts
   first. Keep observations apart from suspected causes. Use this Task's
   workload and model sizes: a speed question is not a campaign, and this
   skill grants no resource or spending authority. Frozen worlds, observation
   versions and recipes don't change. Never induce load.
2. Declare the comparison before running it, on one machine: revisions,
   build, CPU and quota, device, world, input version and capacities,
   vocabulary, precision, model size, recipe or checkpoint, seeds, streams,
   batch and serving settings, warmup, included updates, repetitions and
   limits. Read `TrainingRun.host`; a GPU label isn't host identity.
   Interleave baseline and candidate runs to expose drift. Say whether the
   goal is one seed's progress or aggregate throughput or cost; for packed
   runs keep per-seed latency and aggregate throughput apart, from overlapping
   post-warmup samples and matched solo ordinals.
3. Start from the **Update time** Perf series and its per-update rows. Split
   collection (native stepping, encoding, preparation, inference and action
   selection, recording, assembly) from learner work, with inference calls and
   rows per call beside seconds, decisions and games per update. Nested clocks
   don't add; missing older series stay missing; across a recovery read
   per-update rows, not the last attempt's timers. Check complete-game closure
   and `StageRecord.idle_stream_steps` before blaming small batches on serving
   (older receipts default it to zero).
4. Measure with the maintained tools:
   - Training: `scripts/bench_training_update.py` with explicit `--recipe`,
     `--seed` and `--device` and separate `--json` paths. A capacity failure
     stops the reading; don't shrink the workload.
   - Once Task 855bff's size-ladder benchmark ships, its documented command is
     the reference; until then measure each size with the current benchmark
     and build no second ladder.
   - CUDA: a separate `--device cuda --cuda-counts PATH` run counts kernels
     for the whole run, not per phase; `scripts/diagnose_training_transfers.py`
     probes recording and transfers. Neither is throughput.
   - Evaluation: `manabot.eval.serving_benchmark` with a declared
     `BenchmarkPlan`, per [performance evidence and calibration](../../docs/evaluation.md#performance-evidence-and-calibration).
     Keep `performance.json` with its identities; cell, allocation and
     evaluator intervals overlap and exclude preparation and billing.
5. Keep behavior digests beside the timings: raw and EMA weights, optimizer
   state and sampled games for deterministic training; matched decisions and
   action agreement for evaluation; declared tolerances for what isn't
   deterministic. Precision, policy-lag or batch-composition changes need
   learning evidence this workflow doesn't produce.
6. Write the plan: the bottleneck, the change as slices, the **Delete — do not
   maintain** list, focused checks and exact commands, with receipt paths and
   limits beside each claim. When no useful change is supported, say why and
   plan no slices. Unavailable hardware is a deferred check with its exact
   command; don't rent, substitute CPU for CUDA or claim a gain.
