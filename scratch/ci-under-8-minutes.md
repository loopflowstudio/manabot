# CI under 8 minutes (Task 38ef25; includes MTG-263)

## Outcome and constraints

Every ordinary Ubuntu PR job finishes in under 480 seconds, including install
and build, demonstrated on the PR revision. Preserve test coverage and the
300-second single-test limit; never induce load. Engine Budget retains its
same-runner change/reference comparison and is exempt; the macOS nightly is
outside this budget.

Jack Heart's Task comment (2026-11-12): "Read its brief and deliver its outcome
here; say so in the PR so MTG-263 can be closed with it."

MTG-263 brief:

"Measure per-file durations, then either balance file shards across matrix
groups or adopt pytest-xdist where tests are process-safe (check tmp dirs,
ports, DuckDB files, subprocess fixtures). Target: no group above ~8 min wall
time; report before/after from CI runs. Do not drop or weaken tests."

## Implemented

- `docs/development.md`, CI / Job time, owns the per-job budget and remedies;
  its example is dispatch run 43327574361. Engine Budget is exempt. The macOS
  nightly is outside the Task's "on the ordinary ubuntu runner" scope.
- `.github/workflows/manabot-integration.yml` selects directories, files and
  test nodes in six measured groups with `pytest -n logical`.
  `tests/infra/test_ci_selection.py::test_workflow_test_files_exist` checks
  named files exist, not selection completeness.
- Both Integration workflows sync their extras with
  `--no-install-package managym`, build once with `maturin develop`, then keep
  `UV_NO_SYNC=1` (the Perf proof script also calls `uv run`). The dev-profile
  extension preserves debug assertions in Python-driven games.
- `tests/conftest.py` keeps the five-minute clock in each xdist worker, not
  its controller. The size-cell smoke uses two complete-game streams and one
  update; shrinking its transition floor alone did not bound collection.

- `tests/training/test_active_training.py` inspects the initial checkpoint and
  releases admission in the injected sleep, advancing its fake clock.
  `manabot/training/execution.py::execute_regime` persists the checkpoint
  before that wait. Both admission outcomes and collection/accounting
  assertions remain; the thread pool, polling loop and real-time guards are gone.
- Removed the duplicate engine build, ineffective uv cache step and duplicate
  `test_credit_assignment.py` selection (formerly in both core and long).

## Baseline observations

- Main run 43312163427: every job outside Integration/Engine Budget <= 302 s.
  Other main runs: core 1102-1726 s. Setup was ~180 s per Integration job because
  `uv sync` compiled managym (release, ~94 s) and `maturin develop` compiled it
  again (dev, ~83 s); the `actions/cache` step on `~/.cache/uv` never hit
  (setup-uv moves `UV_CACHE_DIR`).
- Local elapsed time does not establish the Ubuntu job budget. Full branch
  CI dispatches (`gh workflow run ci.yml --ref <branch>`) supply measurements.

## Results (job wall seconds, full `workflow_dispatch` runs on the branch)

| run | result | Integration groups | slowest other job |
|---|---|---|---|
| main 43312163427 | green | core 1443, long 1056, evaluation 746 | Rust Tests 280 |
| 43322780291 (5 groups, `-n auto` = 2 workers) | training failed | evaluation 665, training 650, model 473, recovery 302, experiment 271 | Rust Tests 298 |
| 43324957183 (6 groups, `-n logical` = 4) | training failed | recovery 437, experiment 342, model 305, training 294, evaluation-long 262, evaluation 247 | Rust Tests 406 |
| 43327574361 (before compression) | green | experiment 404, training 379, evaluation 378, evaluation-long 365, model 316, recovery 311 | Rust Tests 397 |
| 43332039325 at `f90ae2893` | green | recovery 416, evaluation 411, experiment 394, training 359, evaluation-long 309, model 268 | Rust Tests 255 |

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

## Reconciled with upstream (2026-11-12)

The green run is at `45fe9f866`; current code at `f90ae2893` includes the
admission-test simplification and main through `48e958fc1` (including
`85ed9214e`, CUDA forward-layout defaults). Training learning modules also moved
to `manabot/rl/` and `manabot/training/updates/`. Integration selections have
not changed since the green run, but the code they exercise has.

A configuration-only check on this revision confirmed the size recipe keeps
its saved layout switches off and its CPU override round-trips through
`TrainingJob.model_validate_json`; the CUDA default change does not require a
new fixture override. No training was run for that check. The ordinary CLI
smoke still selects `local`, which supports CPU fallback.

GitHub inspection on 2026-11-12 found only the three dispatch runs above and
no PR for this branch. Run 43327574361 was rechecked: slowest non-exempt job
404 s, Engine Budget 918 s. This supports the approach, not completion on the
current revision.

## Concept review (2026-11-12, at `f90ae2893`)

A contributor meets four things: the job budget (`docs/development.md`, Job
time), a group's selection list, the parallel-worker contract (`TESTING.md`)
and the five-minute test limit. Each has one owner and the three remedies are
ordered. No design change follows.

- Selection is conserved: every path main's Integration workflow named is
  still named, none twice; the six `tests/eval/` files ignored in `evaluation`
  are exactly the six `evaluation-long` runs.
- Groups are balance buckets, not domains (`experiment` also runs
  `tests/verify/`); the step names say what each holds and the guide says to
  add a file where there is room. Renaming them would buy nothing.
- Unchanged from main, and outside this Task: a new test file outside a
  selected directory runs in no Integration group until someone lists it, and
  `test_ci_selection.py` checks only that named paths exist. The budget is
  likewise read from a run, not enforced by a job timeout; with 150-250 s
  variation on unchanged tests, a hard 8-minute timeout would fail healthy
  runs.

## QA (2026-11-12, at `f90ae2893`)

Full dispatch 43332039325 at `f90ae2893` passed. The slowest non-exempt job took 416 s
(recovery) and Engine Budget took 683 s. Recovery's `Install dependencies` step took 145 s;
the other Integration jobs took 17-31 s, and the same step took 33 s in 43327574361. Its tests took
184 s, against 186 s before. One runner's slow download used about 110 s of the
margin. Evaluation, at 411 s, is the tightest group on test time.

Watchdog under xdist, checked locally by lowering `LIMIT_SECONDS` to 5 with a throwaway
`-p` plugin and test (both since removed): five tests totalling 11 s passed with
`-n 2`, so the controller did not end the healthy run. A 30 s test crashed its
worker with `worker 'gw0' crashed while running ...::test_stuck`, printed the
stacks, and the run exited 1. The same tests without `-n` gave the same outcomes.

## Remaining delivery evidence

- If the PR revision differs from `f90ae2893`, run full Ubuntu CI on it again.
- Retain the before/after results and MTG-263 scope in the PR. One historical
  green run provides limited evidence about timing variation; preserve the
  observed 64-second margin rather than treating 480 s as typical performance.
- No product decision or substantial implementation change is identified.
  Publishing, PR validation and Task completion remain outstanding.

## Publication validation (2026-11-12)

`lf sync` merged main through `285cd49ce` cleanly, producing `30cfcb2e6`.
That merge includes the token-definition changes from main, so the prior
green run cannot establish this revision's timing. Full dispatch
43334548912 ran on `30cfcb2e6` before publication. Every ordinary Ubuntu job
passed under 480 s: evaluation 448, Rust 411, evaluation-long 378,
experiment 368, model 321, training 305, protocol 264, recovery 231,
Academy UI 210, Academy Integration 174, clean-machine play 146,
Python unit tests 145, semantic conformance 122. Engine Budget passed in
896 s, and the complete run is green. The observed minimum timing margin
is now 32 s; this is one run,
not a guarantee against runner and download variability.

On the synced checkout, `uv lock --check`, Ruff check/format for the three
changed Python files, and `git diff --check` passed. The focused selection
check (`uv run --no-sync pytest tests/infra/test_ci_selection.py -n 2 -q`)
passed all 53 tests in 3.88 s.
