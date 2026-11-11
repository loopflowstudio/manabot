---
requires: baseline, training or evaluation optimization diff and bounded comparison evidence
produces: supported gains, invalid comparisons and remaining work in the existing design
---
Judge training and evaluation speed gains against preserved behavior and what the evidence can establish.

1. Read the comparison plan, diff and retained receipts, not just its summary.
   Verify baseline/candidate source, build and workload identities on the same
   machine, including `TrainingRun.host`, GPU, model size, world, input capacities,
   effective vocabulary, precision and sampling/serving settings. GPU labels are
   insufficient: b015d5 found 1.6–1.8x variation on one label. Never use wall-time
   differences across machines as a software speedup.
2. Compare included post-warmup ordinals and interleaved repetitions by size.
   Read the **Update time** Perf series, collection's native stepping/encoding,
   policy inference calls and rows per call, and learner time beside whole-update
   seconds, decisions and games. Separate instrumented CUDA counts and retained
   transfer probes from throughput measurements. Check `performance.json` for
   evaluation matches and report evaluation's time relative to training at the
   declared cadence. Do not sum nested/overlapping clocks, mistake cache reuse for
   a measurement or label elapsed time/steady rates as an inclusive rental bill.
   Keep negative results, failed attempts, sample limits and timing spread visible.
   Judge packing against the declared objective: higher aggregate throughput can
   slow each seed. Require overlapping packed samples and matched solo ordinals;
   a missing overlap is no packing measurement.
3. Check behavior/decision digests, sampled-game or matched-trace evidence and
   affected correctness checks. Training state comparisons include raw/EMA weights
   and optimizer state where deterministic; nondeterministic runs need the
   predeclared numerical/action tolerances. Missing digests or unexplained
   divergence leave equivalence unproved. Narrowed choices, lost observations,
   changed skipping or a broken checkpoint contract invalidate dependent speed
   evidence; repair the contract and remeasure within existing authority.
4. Inspect whether precision, policy lag, batch composition or other numerics or
   sampling changed. Fewer seconds per update, finite losses and a short health
   check do not prove faster or equivalent learning. Name the separate learning
   evidence needed and leave adoption unresolved; this graph neither produces
   that evidence nor launches a scientific campaign. Do not change frozen worlds,
   observation versions or recipes to make the comparison pass.
5. Update existing findings for the next navigation decision: supported gains and
   their scope, invalid/unmeasured claims, worthwhile remaining bottlenecks and a
   justified stop or next implementation. Do not infer a universal GPU ranking,
   extrapolate one size to the ladder or require edits when no useful gain remains.
   Unavailable hardware is a named check with the exact command and required
   device, never an automatic rental or fabricated result. Preserve the existing
   publish/demo/delivery/verify boundaries; software verification does not complete
   a parent scientific experiment. Baseline adoption remains a reviewed operation,
   never automatic regeneration to hide a regression.
