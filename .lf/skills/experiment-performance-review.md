---
requires: the comparison plan, the optimization diff and its run receipts
produces: supported gains, void comparisons and remaining work in `scratch/plan.md`
action_style: procedural
---
Judge training and evaluation speedups against unchanged behavior.

1. Read the receipts directly, not a summary. Baseline and candidate must
   share `TrainingRun.host` (a GPU label alone hides 1.6–1.8x host
   variation), build, model size, world, capacities, vocabulary, precision
   and serving settings. Cross-machine wall time is never a software speedup.
2. Compare interleaved post-warmup updates, size by size: update time,
   collection's stepping and encoding, inference calls and rows per call,
   learner time, decisions and games. Keep CUDA counts and transfer probes
   separate from throughput, report evaluation's share from
   `performance.json`, and don't sum overlapping clocks or call elapsed time
   a bill. Packing requires overlapping samples against matched solo
   ordinals. Keep negative results and run-to-run spread visible.
3. Check digests, matched traces and affected correctness checks. Missing
   digests or unexplained divergence leave equivalence unproved; narrowed
   choices, lost observations, changed skipping or a broken checkpoint
   contract void the speedup.
4. If precision, policy lag, batch composition or sampling changed, faster
   updates don't show equal learning: name the learning runs needed and leave
   adoption for a person to decide. Never alter frozen worlds, observation
   versions or recipes to pass a comparison, and keep baseline adoption a
   reviewed change. Don't edit code here.

When deciding whether to continue:

- **Iterate** with the next bottleneck when a profile shows a credible gain.
- **Advance** when none remains, noting what was ruled out; one model size
  doesn't extrapolate to the ladder. Unavailable hardware is a named check,
  never a rental or invented result.
