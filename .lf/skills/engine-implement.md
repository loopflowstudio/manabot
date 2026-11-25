---
extends: implement
---
## Engine performance

**One profile-backed slice per session.** Implement only improvements backed by
the recorded profile in `scratch/plan.md`—do not rewrite cold paths on
intuition. When `card-perf` scoped the work on a card-coverage branch, fix only
branch-introduced regressions or superlinear new-card `bench_states` paths;
leave pre-existing global hotspots as `engine-performance` follow-ups.

**Preserve behavior, choices and invariants strictly.** Keep legal offers,
observations, replay identities, native skipping and debug assertions
identical; never narrow choices, drop observations or relax an invariant to buy
speed. Run the affected debug Rust tests (including layer/payment oracles and
`bench_states` debug self-tests) after edits.

**Measure on the same build CI times, and keep or revert cleanly.** Compare
base and candidate in an uninstrumented release build (`bench_steps`,
`bench_search` without `--features benchmark`), confirming identical behavior
digests and checking load-invariant retired instructions (`/usr/bin/time -l` or
`perf stat`) on contended hosts. Re-run touched `bench_states` recipes
(`--features benchmark`) for scaling and exact witness digests. Never induce
artificial load or run a stress loop. Revert changes that fail to improve
retired instructions or touched `bench_states` work (or alter any digest), and
record the negative result in `scratch/plan.md`.
