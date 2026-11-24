---
requires: baseline measurements, the optimization diff and same-machine comparisons
produces: performance findings in `scratch/plan.md`
action_style: procedural
---
Judge measured engine speedups against unchanged behavior.

1. Compare baseline and candidate on the same machine and build profile.
   Confirm source and workload identities and exact behavior digests, not
   just elapsed time. Read the `bench_states` rows for every large-board
   state the change touches: work that grows much faster than board size is a
   finding even when curated decks are no slower.
2. Check debug invariants and affected behavior. Changed semantics void the
   speed comparison until the contract is fixed; narrower choices or lost
   observations are never an optimization.
3. Update `scratch/plan.md` with supported gains, negative results and the
   bottlenecks still worth attacking. Leave unavailable hardware as a named
   command, never a made-up number. Don't edit code here. Adopting a new
   budget reference stays its own reviewed change.

When deciding whether to continue:

- **Iterate** with the next bottleneck when a profile shows a credible gain.
- **Advance** when none remains, saying what was ruled out. Passing the 10%
  regression allowance alone doesn't finish optimization.
