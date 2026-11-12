# CI under 8 minutes (Task 38ef25, also delivers MTG-263)

## MTG-263 brief (deliver here; say so in the PR so it can be closed)

"Measure per-file durations, then either balance file shards across matrix
groups or adopt pytest-xdist where tests are process-safe (check tmp dirs,
ports, DuckDB files, subprocess fixtures). Target: no group above ~8 min wall
time; report before/after from CI runs. Do not drop or weaken tests."

## Scope and surviving path

- `docs/development.md`, CI / Job time, owns the per-job budget and remedies;
  its example is dispatch run 43327574361. Engine Budget is exempt. The macOS
  nightly is outside the Task's "on the ordinary ubuntu runner" scope.
- `manabot-integration.yml` selects directories, files and test nodes; it has
  no completeness check. Keep six measured groups and `pytest -n logical`.
- Both Integration workflows sync their extras with
  `--no-install-package managym`, build once with `maturin develop`, then keep
  `UV_NO_SYNC=1` (the Perf proof script also calls `uv run`). The dev-profile
  extension preserves debug assertions in Python-driven games.
- `tests/conftest.py` keeps the five-minute clock in each xdist worker, not
  its controller. The size-cell smoke uses two complete-game streams and one
  update; shrinking its transition floor alone did not bound collection.

## Delete — do not maintain

- Removed the admission test's thread pool, event, file-polling loop and
  `_HANG_SECONDS`. `execution.py` persists the initial checkpoint before its
  admission sleep; the test now inspects that record and releases the gate in
  the injected sleep, advancing its fake clock. Both admission outcomes and
  collection/accounting assertions remain. No supervisor scheduling or
  wall-time wait is needed to test this boundary.
- Removed the duplicate engine build, ineffective uv cache step and duplicate
  `test_credit_assignment.py` selection (formerly in both core and long).
- No remaining deletion targets.

## Baseline observations

- Main run 43312163427: every job outside Integration/Engine Budget <= 302 s.
  Other main runs: core 1102-1726 s. Setup was ~180 s per Integration job because
  `uv sync` compiled managym (release, ~94 s) and `maturin develop` compiled it
  again (dev, ~83 s); the `actions/cache` step on `~/.cache/uv` never hit
  (setup-uv moves `UV_CACHE_DIR`).
- Local timing is not usable as evidence: the machine was at load ~330 from
  other work. CI dispatches on the branch (`gh workflow run ci.yml --ref
  <branch>`) are the measurements.

## Results (job wall seconds, full `workflow_dispatch` runs on the branch)

| run | result | Integration groups | slowest other job |
|---|---|---|---|
| main 43312163427 | green | core 1443, long 1056, evaluation 746 | Rust Tests 280 |
| 43322780291 (5 groups, `-n auto` = 2 workers) | training failed | evaluation 665, training 650, model 473, recovery 302, experiment 271 | Rust Tests 298 |
| 43324957183 (6 groups, `-n logical` = 4) | training failed | recovery 437, experiment 342, model 305, training 294, evaluation-long 262, evaluation 247 | Rust Tests 406 |
| 43327574361 (before compression) | green | experiment 404, training 379, evaluation 378, evaluation-long 365, model 316, recovery 311 | Rust Tests 397 |

Engine Budget (exempt) was 720-919 s throughout. Academy Integration went
221 -> 165 s with the single build.

- Run 1 failure: the a384l8 size cell hit the five-minute limit at 16
  transitions; collection is complete-games, so the test now uses 2 streams
  (62 s in run 2, was 215 s).
- Run 2 failure: `test_initial_admission_precedes_collection[True]` raised
  `TimeoutError` on a 30 s `future.result` beside three other workers. Run 3
  used 120 s guards; compression later removed that threaded test harness.
- The same group varied between runs 2 and 3 with no change to its files
  (evaluation-long 153 -> 249 s of tests, model 170 -> 170), so the margin
  under 480 s is needed; the limit has one green run behind it, not several.
- No test for the xdist-controller clock change in `tests/conftest.py`: run 1's
  training step ran 533 s under `-n` without the controller ending it.

Compression check (2026-11-12): `uv run --no-sync pytest
tests/training/test_active_training.py -n 2 -q` — 5 passed in 99.14 s. Ruff and
`git diff --check` passed. This is correctness evidence, not an Ubuntu CI timing.
