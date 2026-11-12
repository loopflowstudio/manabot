# CI under 8 minutes (Task 38ef25, also delivers MTG-263)

## Docs written before implementation

- `docs/development.md`, CI section: the Integration row of the job table and
  the new **Job time** paragraph (8-minute limit per PR job, how to read job
  times with `gh run view`, order of remedies, Engine Budget exception).

The example output there is from dispatch run 43327574361.

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

## Built (measured by CI dispatches on the branch)

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

## Results (job wall seconds, full `workflow_dispatch` runs on the branch)

| run | result | Integration groups | slowest other job |
|---|---|---|---|
| main 43312163427 | green | core 1443, long 1056, evaluation 746 | Rust Tests 280 |
| 43322780291 (5 groups, `-n auto` = 2 workers) | training failed | evaluation 665, training 650, model 473, recovery 302, experiment 271 | Rust Tests 298 |
| 43324957183 (6 groups, `-n logical` = 4) | training failed | recovery 437, experiment 342, model 305, training 294, evaluation-long 262, evaluation 247 | Rust Tests 406 |
| 43327574361 (final) | green | experiment 404, training 379, evaluation 378, evaluation-long 365, model 316, recovery 311 | Rust Tests 397 |

Engine Budget (exempt) was 720-919 s throughout. Academy Integration went
221 -> 165 s with the single build.

- Run 1 failure: the a384l8 size cell hit the five-minute limit at 16
  transitions; collection is complete-games, so the test now uses 2 streams
  (62 s in run 2, was 215 s).
- Run 2 failure: `test_initial_admission_precedes_collection[True]` raised
  `TimeoutError` on a 30 s `future.result` beside three other workers; its
  waits are hang guards, now 120 s.
- The same group varied between runs 2 and 3 with no change to its files
  (evaluation-long 153 -> 249 s of tests, model 170 -> 170), so the margin
  under 480 s is needed; the limit has one green run behind it, not several.
- No test for the xdist-controller clock change in `tests/conftest.py`: run 1's
  training step ran 533 s under `-n` without the controller ending it.
