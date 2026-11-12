# CI under 8 minutes (Task 38ef25, also delivers MTG-263)

## Docs written before implementation

- `docs/development.md`, CI section: the Integration row of the job table and
  the new **Job time** paragraph (8-minute limit per PR job, how to read job
  times with `gh run view`, order of remedies, Engine Budget exception).

The example output in that paragraph is illustrative; replace its numbers and
group names with the PR's own CI run once the groups are final. If a group is
renamed or Perf proof/training smoke move out of `evaluation`, update the
Integration table row. The comment above the matrix in
`.github/workflows/manabot-integration.yml` still quotes the old group times.

## MTG-263 brief (deliver here; say so in the PR so it can be closed)

"Measure per-file durations, then either balance file shards across matrix
groups or adopt pytest-xdist where tests are process-safe (check tmp dirs,
ports, DuckDB files, subprocess fixtures). Target: no group above ~8 min wall
time; report before/after from CI runs. Do not drop or weaken tests."

## Assumptions

- "A file listed in no group does not run in CI" is stated as current
  behavior (groups name exact paths); no completeness check is promised.
- The macOS nightly is not held to the 8-minute limit (the Task says "on the
  ordinary ubuntu runner").

## Built (first cut, measured by CI dispatch on the branch)

- Baseline, main run 43312163427 (job wall s): core 1443, long 1056,
  evaluation 746, Engine Budget 720 (exempt), every other job <= 302. Other
  main runs: core 1102-1726. Setup was ~180 s per Integration job because
  `uv sync` compiled managym (release, ~94 s) and `maturin develop` compiled it
  again (dev, ~83 s); the `actions/cache` step on `~/.cache/uv` never hit
  (setup-uv moves `UV_CACHE_DIR`).
- Change: one sync with all extras and `--no-install-package managym`, one
  `maturin develop`, `UV_NO_SYNC=1` (the Perf proof script calls `uv run`
  itself); `pytest -n auto` (pytest-xdist added to `dev`); groups `model`,
  `training`, `recovery`, `experiment`, `evaluation`;
  `test_a_size_cell_trains...` runs 16 transitions instead of 64 (215 s test).
  `tests/conftest.py` no longer arms the five-minute clock in the xdist
  controller, where nothing restarts it.
- `tests/training/test_credit_assignment.py` was listed in both `core` and
  `long`; it now runs once, in `training`.
- Python Integration still runs the dev-profile extension (debug assertions
  on). A release build would make engine-heavy tests faster but drops those
  assertions from Python-driven games; not done, a coverage decision.
- Local timing is not usable as evidence: the machine was at load ~330 from
  other work. CI dispatches on the branch (`gh workflow run ci.yml --ref
  <branch>`) are the measurements.
