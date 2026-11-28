---
requires: baseline measurements, the optimization diff and same-machine comparisons
produces: performance findings in `scratch/plan.md`
action_style: procedural
---
Judge measured engine speedups against unchanged behavior.

1. Compare base and candidate on the same machine and uninstrumented release
   build (`bench_steps`, `bench_search`), confirming identities and digests.
   On a contended host, require instructions retired and a CPU profile instead
   of deferring speed to CI; matching `bench_states` counters alone doesn't
   clear per-step cost. Flag superlinear `bench_states` growth on touched
   large-board shapes even when curated decks aren't slower.
2. Check debug invariants and affected behavior; changed semantics void the
   speed comparison, and narrower choices or lost observations are never gains.
3. Update `scratch/plan.md` with supported gains, negative results and open
   bottlenecks, without editing code. When scoped by `card-perf`, do not chase
   pre-existing global engine bottlenecks on a card branch—record them as
   `engine-performance` follow-ups once the branch regression and new-card
   stress blowups are resolved.

When deciding whether to continue:

- **Iterate** when a profile shows a credible gain or an unprofiled
  `bench_steps`/`bench_search` regression remains.
- **Advance** when none remains, saying what was ruled out. Passing the 10%
  budget floor alone doesn't finish optimization.
