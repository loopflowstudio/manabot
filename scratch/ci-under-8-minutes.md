# CI under 8 minutes (Task 38ef25; includes MTG-263)

## Outcome and constraints

Every ordinary Ubuntu PR job finishes in under 480 seconds, including install
and build, demonstrated on the PR revision. Preserve test coverage and the
300-second single-test limit; never induce load. Engine Budget retains its
same-runner change/reference comparison and is exempt; the macOS nightly is
outside this budget.

## Fix-forward plan (2026-11-12, after #506)

PR #506 merged at `60664d149`; its exact push run `43341403551` failed
the live experiment/notebook fixture with `training resource wall deadline
exceeded` (129 passed, one failed). All ordinary jobs stayed below 480 s.
The earlier green PR does not supersede this failure. Full evidence is in
`scratch/ci-post-merge-verification.md`.

Bound the fixture's four real training attempts to one update of eight
one-seat learner transitions on each of two streams using `exact-quota`.
Give each training stage/regime a 120-second resource allowance for startup
and parallel scheduling. These are failure guards, not work targets; the
300-second pytest watchdog and job budget stay unchanged. Retain all four
complete live evaluation cohorts, overlap, accounting, notebook edits and
outputs, and resume assertions. Assert each saved training record contains
one update and 16 learner transitions.

**Delete — do not maintain:** remove the fixture's inherited full-game
training closure and its 30/60-second training guards in place. No production
path, shared baseline, frozen evidence, assertion or test selection is deleted.

Remaining: affected file/report validation and Ubuntu CI timing in gate;
publish a fix-forward PR in this Task through
the existing delivery steps, preserving #506 as history.

Sanity checks:

- `lf sync --manual --json` — synced main through `6b400f95f` into this checkout (`9690bf60b`); existing scratch edits retained.
- Focused test before rebuilding — four learners failed at catalog admission (`catalog_tokens.keys: expected array`), identifying the stale local native extension rather than reaching collection.
- `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed in 46.12 s; installed the matching cp313 extension atomically. No Rust source changed; Ubuntu CI must exercise its debug extension.
- `uv run --no-sync pytest tests/training/test_experiment_runner.py::test_tiny_comparison_evaluates_live_and_notebook_refresh_preserves_edits -q --durations=1` — 1 passed in 45.42 s (43.98 s test body), including all four real updates and monitoring attempts, live overlap, notebook execution/edit preservation and resume. Local timing does not certify Ubuntu.
- `uv run --no-sync ruff check --fix tests/training/test_experiment_runner.py` and `uv run --no-sync ruff format tests/training/test_experiment_runner.py` — passed; `git diff --check` passed.

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

Engine Budget (exempt) was 683-919 s in the dispatches above. Academy Integration went
221 -> 165 s with the single build.

- Run 1 failure: the a384l8 size cell hit the five-minute limit at 16
  transitions; collection is complete-games, so the test now uses 2 streams
  (62 s in run 2, was 215 s).
- Run 2 failure: `test_initial_admission_precedes_collection[True]` raised
  `TimeoutError` on a 30 s `future.result` beside three other workers. Run 3
  used 120 s guards; compression later removed that threaded test harness.
- The same group varied between runs 2 and 3 with no change to its files
  (evaluation-long 153 -> 249 s of tests, model 170 -> 170), so the margin
  under 480 s is needed; at that stage the limit had one green run behind it.
- Before the controlled QA check below, run 1 provided evidence for the
  xdist-controller clock change: training ran 533 s under `-n` without the
  controller ending it. There is no permanent watchdog regression test.

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

## Delivery status

PR #506 merged. Its green PR run and original gate evidence below are
historical; the failing main push and fix-forward plan above supersede the
completion decision. The Task remains open. Existing `scratch/pr-body.md`
and `scratch/pr-title.txt` describe #506 and must be rewritten for the new
fixture-only PR. Preserve #506 as delivery history and its MTG-263 scope.

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

## Gate (2026-11-12, at `1fc4e1b24`)

No implementation defect found. Workflow, test and dependency inputs match
PR run [43337105047](https://github.com/loopflowstudio/manabot/actions/runs/43337105047),
so reuse that full run for integration, debug Rust, browser, clean-machine
installation and dependency validation. This pass changes review documentation
only; no second full matrix or training run was needed.

Every job passed. Non-exempt jobs took at most 349 s (recovery), down from
1443 s on main, a 76% reduction in the slowest job. Other Integration groups:
evaluation 346, evaluation-long 324, experiment 320, model 311, training 299 s.
Academy Integration took 186 s; Engine Budget passed in 613 s and remains
exempt. The PR run leaves 131 s of margin, while the prior full dispatch left
32 s. This is observed timing, not a guaranteed ceiling.

Validation plan and results (one command/result per line):

- `gh run view 43337105047 --repo loopflowstudio/manabot --json headSha,event,conclusion,jobs,url` — successful `pull_request` run; head matches `1fc4e1b24`; all ordinary jobs below 480 s.
- `gh run view 43337105047 --repo loopflowstudio/manabot --log` — reused exact-code CI: training 346 passed, recovery 49 passed, experiment 130 passed, evaluation-long 66 passed; changed size cells took 63.58/19.14 s and admitted gate case 19.82 s. Model/evaluation retain conditional skips; no skip was added by this branch.
- `uv run --no-sync pytest tests/infra/test_ci_selection.py -n 2 -q` — 53 passed in 0.37 s; checks the changed workflow selections and routing.
- `uv run --no-sync ruff check --fix tests/conftest.py tests/training/test_active_training.py tests/training/test_training_throughput_sizes.py` — passed, no changes.
- `uv run --no-sync ruff format tests/conftest.py tests/training/test_active_training.py tests/training/test_training_throughput_sizes.py` — all three files unchanged.
- `uv lock --check` — passed, 176 packages resolved; isolated dependency installation already passed in the exact-code CI run.
- `git diff --check` and `git diff origin/main...HEAD --check` — passed.
- `uv run --no-sync python` selection audit against `origin/main` — all 69 original paths/nodes retained; the six evaluation exclusions exactly match `evaluation-long`. Initial literal set equality rejected the six newly explicit files already covered by the old directory selection; correcting that audit assumption established coverage conservation without changing the workflow.

Earlier required-check failures are explained above: complete-game collection
made the large eight-stream smoke exceed 300 s; the final two-stream fixture
keeps its one-update assertion. The old admission test's 30 s future timeout
was host-sensitive; the final injected-sleep fixture removes that timeout and
retains both admission outcomes and active-time accounting.

Updated `scratch/pr-title.txt`, `scratch/pr-body.md` and `scratch/.pr-copy-ref`
for the next delivery step, preserving the existing Task context marker and
MTG-263 scope. The live PR description has not been edited in this gate.
No checks are newly deferred, and no server or watcher was started.
