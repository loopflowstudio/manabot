# CI under 8 minutes (Task 38ef25; includes MTG-263)

## Outcome and constraints

Every ordinary Ubuntu PR job finishes in under 480 seconds, including install
and build, demonstrated on the PR revision. Preserve test coverage and the
300-second single-test limit; never induce load. Engine Budget retains its
same-runner change/reference comparison and is exempt; the macOS nightly is
outside this budget.

## Implemented fix-forward (2026-11-12, after #506)

PR #506 merged at `60664d149`; its exact push run `43341403551` failed
the live experiment/notebook fixture with `training resource wall deadline
exceeded` (129 passed, one failed). All ordinary jobs stayed below 480 s.
The earlier green PR does not supersede this failure. Full evidence is in
`scratch/ci-post-merge-verification.md`.

`tests/training/test_experiment_runner.py::demo_declaration` now bounds each
of four real training attempts to one update of eight one-seat learner
transitions on each of two streams using `exact-quota`. Stage and regime
resource allowances are 120 seconds. The comparison test reads every saved
`TrainingRun` and asserts one stage, one update and 16 learner transitions.
All four live complete-game evaluation cohorts, overlap, accounting, notebook
edits/outputs and resume assertions remain.

`manabot/training/regime.py::TrainSelfPlay` explains why the previous eight
transitions were only a floor under `complete-games`; `exact-quota` pauses at
that many learner rows per stream and requires one learning seat. This is an
orchestration fixture, so complete-game training is unnecessary (the assumption
is retained in `scratch/questions.md`). Production defaults and the shared WDL
baseline are unchanged. The 120-second allowances guard failure, not expected
work duration; they do not establish that four attempts plus monitoring fit
the unchanged 300-second watchdog or 480-second job budget.

The merged-run error came from `manabot/training/execution.py`'s resource
`check()`. Its log does not identify which clock expired or prove how much time
was collection versus scheduling. Bounding collection addresses the unbounded
workload; Ubuntu CI must still establish the repaired fixture's reliability.

Sanity checks:

- `lf sync --manual --json` — synced main through `6b400f95f` into this checkout (`9690bf60b`); existing scratch edits retained.
- Focused test before rebuilding — four learners failed at catalog admission (`catalog_tokens.keys: expected array`), identifying the stale local native extension rather than reaching collection.
- `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed in 46.12 s; installed the matching cp313 extension atomically. No Rust source changed; Ubuntu CI must exercise its debug extension.
- `uv run --no-sync pytest tests/training/test_experiment_runner.py::test_tiny_comparison_evaluates_live_and_notebook_refresh_preserves_edits -q --durations=1` — 1 passed in 45.42 s (43.98 s test body), including all four real updates and monitoring attempts, live overlap, notebook execution/edit preservation and resume. Local timing does not certify Ubuntu.
- `uv run --no-sync ruff check --fix tests/training/test_experiment_runner.py` and `uv run --no-sync ruff format tests/training/test_experiment_runner.py` — passed; `git diff --check` passed.
- Compression: `uv run --no-sync pytest tests/training/test_experiment_runner.py -q --durations=3` — 10 passed in 151.36 s; live comparison 82.03 s, diagnostic exports 66.86 s. Ruff check/format and `git diff --check` passed. All four training receipts still assert one update and 16 learner transitions. No process remains running.

Jack Heart's Task comment (2026-11-12): "Read its brief and deliver its outcome
here; say so in the PR so MTG-263 can be closed with it."

MTG-263 brief:

"Measure per-file durations, then either balance file shards across matrix
groups or adopt pytest-xdist where tests are process-safe (check tmp dirs,
ports, DuckDB files, subprocess fixtures). Target: no group above ~8 min wall
time; report before/after from CI runs. Do not drop or weaken tests."

## Surviving implementation and evidence

PR #506 delivered six measured Integration groups with logical-core pytest
workers, one dependency sync/native debug build per Integration job, the
per-worker 300-second watchdog, injected-time admission tests and a two-stream
real-worker size smoke. `docs/development.md` owns the ordinary Ubuntu job
budget; `TESTING.md` owns process isolation. These are already on main.

| GitHub Actions run | Revision / event | Result and slowest ordinary job |
|---|---|---|
| 43312163427 | main baseline | green; Integration core 1443 s |
| 43327574361 | `45fe9f866`, dispatch | green; experiment 404 s |
| 43332039325 | `f90ae2893`, dispatch | green; recovery 416 s |
| 43334548912 | `30cfcb2e6`, dispatch | green; evaluation 448 s |
| 43337105047 | `1fc4e1b24`, PR | green; recovery 349 s |
| 43341403551 | `60664d149`, merged push | experiment failed at 433 s; all other ordinary jobs passed, max 418 s |

Runs: `https://github.com/loopflowstudio/manabot/actions/runs/<id>`.
The exempt Engine Budget passed in 613 s on the PR and 685 s on the merged
push. The full post-merge failure and timing record is in
`scratch/ci-post-merge-verification.md`; it supersedes the old green gate and
demo decision. No human demo or approval was recorded.

Contrary findings and limits:

- One recovery runner spent 145 s installing dependencies versus 17–31 s on
  its peers; unchanged evaluation-long tests varied from 153 to 249 s. The
  32-second margin in dispatch 43334548912 is not a timing guarantee.
- Earlier attempts failed because complete-game collection was not bounded
  by its transition floor, and a threaded admission fixture had a 30-second
  future timeout. Both causes were repaired before #506; the post-merge
  experiment failure is separate.
- The historical controlled watchdog proof used a temporary five-second
  limit: five tests totaling 11 s passed with two workers; a 30-second test
  printed stacks, crashed its worker and exited 1. Serial behavior matched.
  Temporary fixtures were removed. This remains evidence for unchanged code.
- The merge audit retained all 69 original selected paths/nodes and ran all
  six evaluation exclusions in evaluation-long. New test files outside a
  selected directory still need explicit workflow selection; no new coverage
  guarantee or hard eight-minute job timeout was introduced.

Detailed prior measurements, command results and gate history remain at
`937f4f152:scratch/ci-under-8-minutes.md`. This plan keeps the current repair
and the evidence that still constrains it.

## Concept review (2026-11-12, headless)

No change to implement and no open product choice. The fixture now keeps two
ideas apart that #506's version ran together: how much work an attempt does
(one update, 16 learner rows, asserted from every saved `TrainingRun`) and how
long it may take before it is called hung (120 s). A regression to
`complete-games` fails the row assertion rather than reappearing as a slow run.
`transitions` still means a floor under `complete-games` and a quota under
`exact-quota`; it is a serialized key, explained beside `batch_closure`, and
keeps its spelling.

Recovery direction if the Ubuntu gate fails this fixture again with
`training resource wall deadline exceeded`: do not raise the allowance.
`execution.py::remaining()` takes the minimum of three clocks (the launch
`deadline`, the regime's charged wall and the stage's charged wall) and
`check()` reports none of them, so a second failure would be as undiagnosable
as the first. The next change is then to have the error name the expired clock
with charged and allowed seconds. It is deferred, not part of this PR: the text
appears in `tests/training/test_learning_state.py:332` as a retained failure
payload, so changing it needs that reader checked, and a green Ubuntu run makes
it unnecessary for this Task.

## Remaining delivery

QA established the fixture repair on Ubuntu and passed the companion report
tests, but found the installation gap below. Gate must measure the cache repair
on Ubuntu: record the first miss/save and a subsequent hit/restore, including
Setup uv, Install dependencies, Post Setup uv and total job seconds. Confirm
the warm install avoids Torch/nvidia downloads, all selected tests pass with
debug assertions, and ordinary jobs finish below 480 s. A cold run alone cannot
validate retention; a warm run alone cannot establish initial save overhead.
No hosted measurement of the cache repair is recorded yet.

Publish the fixture and cache fix-forward PR in Task 38ef25, retaining #506 as
history. Existing `scratch/pr-body.md`, `scratch/pr-title.txt` and
`scratch/.pr-copy-ref` describe #506 and must be rewritten at delivery.
PR-event timing and post-merge verification remain necessary; the Task stays
open until post-merge verification succeeds.

## QA of the fix-forward (2026-11-12)

Dispatch [43345024882](https://github.com/loopflowstudio/manabot/actions/runs/43345024882)
at `4db5cd64b`: every job succeeded. The repaired comparison test took 151.70 s on
Ubuntu debug/xdist (176.21 s and failing on the merged push). The experiment job took
**549 s**, over budget: install 181 s (peers 17–18 s; slow CUDA/nvidia wheel download
after a pruned uv-cache hit), build 66 s, tests 284 s. Every other ordinary job ≤ 430 s.
Locally: CI's experiment group 130 passed in 80.87 s; `test_experiment_report.py`
17 passed. Reverting the fixture to `complete-games`/both seats failed the new
assertion (`610 == 16`). The cache choice is resolved below and in `questions.md`;
details in `qa.html`.

## Cache retention repair (2026-11-12)

Keep the existing Torch build and dependency selection. Disable setup-uv's
pruning in `manabot-integration.yml`; every group restores the resulting cache,
and `model` alone saves it because all six groups install the same extras.
This avoids six compressions/uploads on a miss. `docs/development.md` explains
the cold-key limitation and includes restore/save overhead in timing comparisons.

Delete — do not maintain: remove Integration's implicit pruned-cache behavior.
No files, symbols or tests are deletion targets. Keep the bounded four-attempt
fixture, all test selections, native debug build, 300-second watchdog, dependency
groups and lockfile. Keep one cache owner (setup-uv), without a separate
actions/cache step, CPU Torch alternative or per-group cache copies.

External fact checked against setup-uv v6 on 2026-11-12: `prune-cache` defaults
to true and calls `uv cache prune --ci`; false removes the `-pruned` key component,
so the old immutable cache is not reused. The key also includes architecture,
platform, Python and dependency-manifest hash. `save-cache: false` skips saving;
an exact hit skips both pruning and saving. Sources:
[inputs](https://github.com/astral-sh/setup-uv/blob/v6/action.yml),
[restore](https://github.com/astral-sh/setup-uv/blob/v6/src/cache/restore-cache.ts),
[save](https://github.com/astral-sh/setup-uv/blob/v6/src/save-cache.ts).

Sanity checks: `actionlint .github/workflows/manabot-integration.yml` — passed;
`uv run --no-sync pytest tests/infra/test_ci_selection.py::test_workflow_test_files_exist -q`
— 1 passed; `git diff --check` — passed. A read-only comparison to HEAD confirmed
that removing the four cache lines reproduces the prior workflow exactly,
including every install/build/test command. `lf context` is within both limits.
Hosted cold/warm timing is pending at gate; no under-budget claim follows from
static validation.
