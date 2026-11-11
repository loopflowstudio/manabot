---
requires: training or evaluation speed question and representative workloads
produces: measured bottleneck, bounded comparison plan and checks in the existing scratch design
---
Measure what training and evaluation spend time on, then select a worthwhile software improvement.

1. Read `docs/workflows.md` (training and evaluation performance),
   `experiments/training-update-time.md` and the relevant retained receipts before
   running anything. Separate observations from suspected causes. Use the current
   task's workload and model sizes; do not expand a speed question into a campaign.
   This skill grants no campaign, resource or spending authority. Preserve frozen
   worlds, observation versions and recipes. Never induce artificial load.
2. Declare a bounded baseline/candidate comparison on the same machine: source
   revisions, build profile, CPU and quota, GPU/device, world, input version and
   capacities, effective record vocabulary, precision, model size/parameter count,
   recipe/checkpoint, seeds, streams, producers and batch/serving settings. Read
   `TrainingRun.host`; a GPU label alone is not host identity. Keep warmup,
   included update ordinals, repetitions, memory and time limits explicit. Use
   interleaved comparisons to expose shared-host drift. Do not compare wall time
   across machines or pool different sizes into one gain.
   State whether the objective is one seed's progress or aggregate throughput/cost.
   For packed runs, retain per-seed latency and aggregate throughput separately,
   using overlapping post-warmup samples and matched solo ordinals.
3. Start with the **Update time** Perf series and its retained per-update rows.
   Separate collection's native stepping, observation encoding, preparation,
   policy inference/action selection, recording and assembly from learner targets
   and minibatches. Retain inference calls, inferred rows and rows per call beside
   seconds/update, decisions/update and games/update. Missing older series stay
   missing; nested clocks are not additive. Across recovery, read per-update rows:
   receipt timers can cover only the last attempt while counts span attempts.
4. Use the maintained tools on available, authorized hardware:
   - Training: use `scripts/bench_training_update.py` through the retained-output
     command in [Training and evaluation performance](../../docs/workflows.md#training-and-evaluation-performance).
     Select the declared `--recipe`, `--seed` and `--device` explicitly, with
     separate baseline/candidate output paths. The tool re-pins its in-memory
     recipe to the current world; it cannot reproduce frozen experiments.
     Capacity failure stops the reading; do not truncate or shrink the workload.
   - When Task 855bff's standing size-ladder benchmark and retained baseline land,
     read its shipped documentation and use that command as the reference
     reading. Until then use the current benchmark per declared size; do not
     invent the future command, block on it or implement a second ladder here.
   - CUDA diagnostics: run the training command separately with `--device cuda`
     and `--cuda-counts PATH`. Counts cover the whole run, not individual phases;
     instrumented timings are not ordinary throughput. Read
     `scripts/diagnose_training_transfers.py` for bounded `recording`/`transfers`
     probes on retained inputs. These probes do not establish whole-update gains.
   - Evaluation: follow [Performance evidence and calibration](../../docs/evaluation.md#performance-evidence-and-calibration)
     for the `manabot.eval.serving_benchmark` command and `BenchmarkPlan`
     requirements, using the predeclared plan and output paths. Retain
     `performance.json` with match/deal/checkpoint identities, rows, journals and
     traces. Compare evaluation time to training at the declared cadence, by size.
     Distinguish cell, allocation and evaluator intervals; they overlap and exclude
     preparation and billing. Label any cost projection and its rate source.
5. Retain behavior/decision digests and numerical checks with the timings. For a
   deterministic training comparison, include raw/EMA weights, optimizer state
   and sampled-game evidence; for evaluation, include matched decisions/traces
   and numerical/action agreement. Identify expected nondeterminism and declared
   tolerances before comparing; finite metrics alone are not learning evidence.
   Changes to precision, policy lag or batch composition require separate learning
   evidence, which this software workflow does not produce or authorize.
6. Update the existing design with the measured bottleneck, proposed change,
   **Delete — do not maintain** list, focused behavior checks and exact comparison
   commands. Keep receipt paths/hashes, included samples, negative findings and
   limits next to each claim. If no useful change is supported, record why and stop
   without manufacturing edits. If required hardware or evidence is unavailable,
   name the missing input and exact command as a deferred check for gate/CI or
   separately authorized execution; do not rent, substitute CPU for CUDA, or claim
   a gain. A counterexample invalidates dependent comparisons and requires revising
   the plan before implementation.
