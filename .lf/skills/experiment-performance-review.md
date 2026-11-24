---
requires: the comparison plan, the optimization diff and its receipts
produces: supported gains, void comparisons and remaining work in `scratch/plan.md`, and the loop decision
---
Judge training and evaluation speed gains against unchanged behavior, then decide whether to keep optimizing.

1. Read the receipts, not their summary. Baseline and candidate share the
   machine (`TrainingRun.host`, not the GPU label: b015d5 found 1.6–1.8x
   variation on one label), build, model size, world, capacities, vocabulary,
   precision and sampling and serving settings. Cross-machine wall time is
   never a software speedup.
2. Compare included post-warmup updates and interleaved repetitions, size by
   size: update time, collection's stepping and encoding, inference calls and
   rows per call, learner time, decisions and games. Keep CUDA counts and
   transfer probes apart from throughput, report evaluation's share of the
   workload from `performance.json`, and don't sum overlapping clocks or call
   elapsed time a bill. Judge packing against the declared objective with
   overlapping samples; no overlap, no packing measurement. Keep negative
   results and spread visible.
3. Check digests, matched traces and affected correctness checks. Missing
   digests or unexplained divergence leave equivalence unproved; narrowed
   choices, lost observations, changed skipping or a broken checkpoint
   contract void the speed result.
4. When precision, policy lag, batch composition or sampling changed, faster
   updates don't show equal learning: name the learning evidence needed and
   leave adoption to the person at demo. Frozen worlds, observation versions
   and recipes don't change to make a comparison pass, and baseline adoption
   stays a reviewed operation.

Iterate with the next bottleneck when one has a credible measured gain.
Advance when none does, with what was ruled out; one size doesn't extrapolate
to the ladder. Unavailable hardware is a named check, never a rental or an
invented result.
