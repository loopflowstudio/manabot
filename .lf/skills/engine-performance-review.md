---
requires: the baseline, the optimization diff and same-runner comparisons
produces: performance findings in `scratch/plan.md` and the phase's loop decision
---
Judge the measured gains against unchanged behavior, then decide whether optimization is done.

1. Compare the workloads on the same hardware and build profile. Confirm
   source and workload identities and behavior digests, not just elapsed time.
   Read the `bench_states` rows for the large-board states the change touches:
   a cell whose work grows much faster than its size is a finding even when
   the curated decks are no slower.
2. Check debug invariants and affected behavior. Changed semantics void the
   comparison until the contract is fixed; narrower choices or lost
   observations are never an optimization.
3. Record supported gains, negative results and the bottlenecks still worth
   attacking. Missing hardware is a named gate or CI check, never a made-up
   number. Don't change code: the next implement pass does, so the loop
   reviews the final diff. Adopting a new budget reference stays its own
   reviewed operation.

Iterate with the next bottleneck when one has a credible measured gain.
Advance when none does, saying what was ruled out. There is no absolute
optimum, and a passing regression threshold doesn't end the search.
