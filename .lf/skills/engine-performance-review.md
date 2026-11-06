---
requires: baseline, optimization diff and bounded comparison evidence
produces: performance findings and remaining bottlenecks in the existing design
---
Judge measured gains against preserved semantics and the remaining useful opportunities.

1. Compare representative workloads on the same hardware/build profile. Confirm
   source/workload identities and behavior digests, not just elapsed time.
2. Inspect debug invariant checks and affected behavior evidence. If semantics
   changed, invalidate the speed comparison and repair the working contract before
   accepting dependent results. Never accept narrower choices or lost observations
   as an optimization.
3. Record supported gains, negative results, worthwhile remaining bottlenecks and
   justified stopping tradeoffs. Do not claim an absolute optimum or treat the
   regression threshold as proof that optimization is complete. Missing capable
   hardware is a named gate/CI check, not fabricated measurement.

Update existing findings for the navigation decision. Preserve performance budget
reference adoption as its existing reviewed operation, not automatic regeneration.
