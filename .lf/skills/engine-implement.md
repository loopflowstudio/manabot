---
requires: profile-backed optimization slices in `scratch/plan.md` from `engine-measure` or `engine-performance-review`
produces: committed behavior-preserving Rust performance slice with same-machine measurements and debug invariant checks
action_style: procedural
---
Build the next profile-backed Rust performance slice from `scratch/plan.md` without altering game behavior.

1. **One profile-backed slice per session.** Read the next unbuilt performance
   slice and the files it names (finished slices are one line each; their
   detail is in git). Implement only improvements backed by the recorded
   profile—do not rewrite cold paths on intuition. When `card-perf` scoped the
   work on a card-coverage branch, fix only branch-introduced regressions or
   superlinear new-card `bench_states` paths; leave pre-existing global
   hotspots as `engine-performance` follow-ups.
2. **Preserve behavior, choices and invariants strictly.** Keep legal offers,
   observations, replay identities, native skipping and debug assertions
   identical; never narrow choices, drop observations or relax an invariant to
   buy speed. Run the affected debug Rust tests (including layer/payment
   oracles and `bench_states` debug self-tests) after edits.
3. **Measure on the same build CI times.** Compare base and candidate in an
   uninstrumented release build (`bench_steps`, `bench_search` without
   `--features benchmark`), confirming identical behavior digests and checking
   load-invariant retired instructions (`/usr/bin/time -l` or `perf stat`) on
   contended hosts. Re-run touched `bench_states` recipes (`--features
   benchmark`) to verify scaling and exact witness digests. Never induce
   artificial load or run a stress loop.
4. **Keep or revert cleanly.** If a change fails to improve retired
   instructions or touched `bench_states` work (or changes any digest), revert
   the dead code instead of leaving speculative complexity behind, and record
   the negative result in `scratch/plan.md`.
5. **Collapse the plan and commit.** Replace the completed slice in
   `scratch/plan.md` with one line naming its commit and measured delta, and
   commit your edits cleanly before finishing.

When deciding whether to continue:

- **Iterate** while `scratch/plan.md` still has unbuilt profile-backed
  performance slices.
- **Advance** when no unbuilt slices remain so `engine-performance-review` can
  verify the cumulative gain.
