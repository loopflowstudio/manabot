---
requires: card-perf baseline measurements and the branch-scoped optimization diff
produces: updated performance verification in `scratch/plan.md`
action_style: procedural
---
Verify that the branch-scoped performance fix resolved the regression or new-card scaling blowup without changing behavior or scope.

1. Re-run the exact same-machine uninstrumented release `bench_steps` and
   `bench_search` comparisons (retired instructions via `/usr/bin/time -l` or
   `perf stat`, identical behavior digests) and the touched new-card
   `bench_states` recipes from `card-perf`.
2. Confirm debug invariants and affected behavior tests pass; changed semantics,
   narrowed choices or lost observations void any speed claim.
3. Keep scope tight: do not chase pre-existing global engine bottlenecks on a
   card branch—record those as `engine-performance` follow-ups instead.
4. Update `scratch/plan.md` with the before/after numbers, digests and any
   remaining branch-owned bottleneck, without editing code here.

When deciding whether to continue:

- **Iterate** only if a branch-introduced regression (>~2% retired instructions
  vs. base), superlinear new-card `bench_states` blowup or broken debug
  invariant still remains with a concrete profile-backed fix.
- **Advance** when the branch no longer regresses base, the new cards' stress
  shapes scale cleanly and debug invariants hold.
