# MTG-237: Engine performance budget in CI

Design from Jack Heart's concept review of 2026-11-02 (Performance section,
`scratch/engine-foundations.html` in the T6 checkout) and the Task brief.

## Outcome

- `Engine Budget` CI job (`.github/workflows/engine-budget.yml`, called from
  `ci.yml`, selected by the engine, the benchmarks, the baseline and the
  comparator) builds `bench_search` and `bench_steps` in release on a runner
  of its own and fails the PR when a timing is more than 10% slower than
  `managym/perf/baseline.json` or a digest differs.
- `bench_search`: unchanged workload (seeds 1-3, 40 decisions, 16x4 flat
  search), summary now sums totals. `bench_steps`: 512 seeded random-legal
  games of the curated matchup through `Env::reset`/`step` with skip_trivial,
  ~100k steps, one digest over every decision. Decks shared in
  `examples/curated/mod.rs`.
- `scripts/engine_budget.py compare|record` (stdlib only; `record` rewrites
  the baseline with commit, date, run URL and reason). Tests in
  `tests/infra/test_engine_budget.py` include the 20% slowdown fixture.
- Manual dispatch with `measure=<commit>` benchmarks another commit with this
  ref's benchmarks and records it without comparing: the one-time T4/T5/T6
  measurement and the way a baseline is taken on the runner profile.

## Delete — do not maintain

Nothing is replaced. The Clean-machine Play 120 s wait stays a launch proof,
not a latency budget.

## Status

Implemented 2026-11-02. Baseline and the T4/T5/T6 measurements come from CI
dispatches (see questions.md for the measurement record).
