# MTG-237: Engine performance budget in CI

Design from Jack Heart's concept review of 2026-11-02 (Performance section,
`scratch/engine-foundations.html` in the T6 checkout) and the Task brief.

## Outcome

- `Engine Budget` CI job (`.github/workflows/engine-budget.yml`, called from
  `ci.yml`, selected by the engine, the benchmarks, the baseline and the
  comparator) builds `bench_search` and `bench_steps` in release on a runner
  of its own, rebuilds the reference commit of `managym/perf/baseline.json`
  with the same benchmarks on that runner, and fails the PR when the
  candidate is more than 10% slower than the reference or a digest differs
  (runner CPU models vary too much for absolute numbers; see questions.md).
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

## Decisions

- Accepted (Task brief, Jack Heart, 2026-11-02): both workloads budgeted at
  10% against a committed file; digests asserted identical; a 20% slowdown
  fixture; one-time T4/T5/T6 measurement; `uv` only; no induced load.
- Draft, awaiting Jack Heart's review: "committed baseline" is read as a
  committed *reference commit* rebuilt and measured A/B on the same runner,
  not committed absolute numbers. Evidence: one engine spread 6.8k-10.3k
  steps/s across three runner CPU models (far outside 10%); two runs on one
  model agreed within 2.5%. The file's figures are orientation only.
- Draft, awaiting review: the headless workload is native `Env::step`
  random-legal play (512 games, skip_trivial), not the rul-9 Python
  semantic-Command runner (deleted in #343). Its steps/s is its own
  baseline, not comparable with rul-9's 588.5.
- Draft: any rules change reaching the curated decks moves the random-play
  digest and costs one `record` per such PR; accepted as the price of (3).

## Status (reconciled 2026-11-02, second pass)

Exists: workflow, both benchmarks, comparator, tests, baseline file,
AGENTS.md note and `docs/development.md` section with the T4/T5/T6 table.
Reference 4c8324a59 recorded from CI run 41517623999. Full CI validated by
workflow_dispatch run 41527511986 at 07512aa8c (Engine Budget A/B included).
Since then: the script simplification b734238c4, and the compress pass of
2026-11-02 that gave the workflow one `other` checkout (the measured commit
or the reference, chosen once; `CANDIDATE`/`REFERENCE` paths replace
`ENGINE_DIR`/`MEASURE` branching in every later step) and dropped `record`'s
unused `--tolerance` flag (the tolerance is edited in the file). The second
reconciliation completed the documented benchmark `Env` surface
(`is_game_over`, `winner_index`, which `bench_steps` calls). Main has not
moved past the base 4c8324a59 (checked 2026-11-02), so nothing upstream
changes the approach. Tests and actionlint pass locally; CI has not run on
this head because the branch has no PR and branch pushes do not trigger CI.
The PR run covers the compare path; the `measure` path is unexercised since
the rewrite and wants one dispatch before the next re-record (gate/CI).

Remains (publication, owned by the caller): open the PR with the T4/T5/T6
figures in its body; the PR run confirms path selection on the
`pull_request` event, and the first main run after merge confirms the push
path. Jack Heart's review decides the two draft decisions above.

Proposed Wave memory, not written: engine speed now has a CI budget relative
to a committed reference commit; GitHub runner wall time is not comparable
across runs; T4 main measured 6.8k steps/s and 13.8 s worst Search-64 reply
on EPYC 7763. Left in scratch because the A/B reading of "committed
baseline" it rests on is a draft awaiting Jack Heart's review; write it to
`.lf/wave/engine/MEMORY.md` once the review settles it.

Check: `uv run --extra dev pytest tests/infra/test_engine_budget.py tests/infra/test_ci_selection.py -q` → 55 passed (2026-11-02, head 675a2e623 plus the docs edit).
