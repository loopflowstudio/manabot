# CI under 8 minutes (Task 38ef25; includes MTG-263)

## Outcome and constraints

Every ordinary Ubuntu PR job finishes in under 480 seconds, including install,
build and cache overhead, demonstrated on the PR revision. Preserve coverage
and the 300-second single-test limit; never induce load. Engine Budget retains
its same-runner change/reference comparison and is exempt. The macOS nightly
is outside this budget.

Jack Heart's Task comment (2026-11-12): "Read its brief and deliver its outcome
here; say so in the PR so MTG-263 can be closed with it."

MTG-263 brief:

"Measure per-file durations, then either balance file shards across matrix
groups or adopt pytest-xdist where tests are process-safe (check tmp dirs,
ports, DuckDB files, subprocess fixtures). Target: no group above ~8 min wall
time; report before/after from CI runs. Do not drop or weaken tests."

## Current repair (2026-11-12)

PR #506 merged at `60664d149`. Its exact push run `43341403551` failed the
live experiment/notebook fixture with `training resource wall deadline
exceeded` (129 passed, one failed), although every ordinary job took less than
480 s. The error did not identify which resource clock expired or separate
collection from scheduling. Details: `scratch/ci-post-merge-verification.md`.

`tests/training/test_experiment_runner.py::demo_declaration` now bounds each
of four real training attempts to one update of eight one-seat learner rows
on two streams using `exact-quota`. Every saved `TrainingRun` must contain one
stage, one update and 16 learner transitions. Four live complete-game evaluation
cohorts, overlap, accounting, notebook edits/outputs and resume assertions remain.

`manabot/training/regime.py::TrainSelfPlay.batch_closure` explains the choice:
`complete-games` treats transitions as a floor; `exact-quota` pauses at that
many learner rows and requires one learning seat. This orchestration fixture
needs no complete-game training (`scratch/questions.md`). Production defaults
and the shared WDL baseline stay unchanged. Stage and regime allowances of
120 s guard hangs; they are not work targets.

QA dispatch [43345024882](https://github.com/loopflowstudio/manabot/actions/runs/43345024882)
at `4db5cd64b` passed every job with Ubuntu debug assertions and xdist. The
repaired comparison test took **151.70 s** (176.21 s and failing on the merge).
The experiment job still took **549 s**: installation 181 s versus peers'
17–18 s, build 66 s, tests 284 s. Logs showed slow CUDA/nvidia wheel downloads
after a pruned uv-cache hit. Every other ordinary job was at most 430 s;
Engine Budget passed in 587 s. Full QA evidence: `scratch/qa.html`.

The cache repair in `.github/workflows/manabot-integration.yml` sets
`prune-cache: false`. All groups install the same extras and restore the cache;
`model` alone saves it, avoiding duplicate compression/uploads. This preserves
the existing Torch build; CPU Torch was the alternative. `docs/development.md`
owns the cache explanation and the job-time budget. Hosted cache-repair timing
is still pending.

**Delete — do not maintain:** the inherited complete-game fixture workload,
singleton stage loop and 30/60-second guards are replaced, as is Integration's
implicit cache pruning. No remaining file, symbol or test deletion targets.
Keep setup-uv as the sole dependency-cache owner, all test selections, the
native debug build, the 300-second watchdog and the existing dependency groups
and lockfile.

## Evidence and limits

PR #506 already delivered six measured Integration groups with logical-core
pytest workers, one dependency sync/native debug build per job, the per-worker
watchdog, injected-time admission tests and a two-stream real-worker size smoke.
`TESTING.md` owns process isolation.

| GitHub Actions run | Revision / event | Result and slowest ordinary job |
|---|---|---|
| 43312163427 | main baseline | green; Integration core 1443 s |
| 43327574361 | `45fe9f866`, dispatch | green; experiment 404 s |
| 43332039325 | `f90ae2893`, dispatch | green; recovery 416 s |
| 43334548912 | `30cfcb2e6`, dispatch | green; evaluation 448 s |
| 43337105047 | `1fc4e1b24`, PR | green; recovery 349 s |
| 43341403551 | `60664d149`, merged push | experiment failed at 433 s; other ordinary jobs passed, max 418 s |
| 43345024882 | `4db5cd64b`, dispatch | green; experiment 549 s, over budget |

Runs: `https://github.com/loopflowstudio/manabot/actions/runs/<id>`.
Engine Budget passed in 613 s on #506's PR and 685 s on its merged push.
No human demo or approval was recorded. Historical green results do not
validate the current repair.

- Recovery installation previously took 145 s versus peers' 17–31 s;
  unchanged evaluation-long tests varied from 153 to 249 s. The 32-second
  margin in dispatch 43334548912 is not a timing guarantee.
- Earlier failures came from complete-game collection exceeding its transition
  floor and a threaded admission fixture's 30-second future timeout; both were
  repaired before #506. The post-merge experiment failure was separate.
- Controlled watchdog proof used a temporary five-second limit: five tests
  totaling 11 s passed with two workers; a 30-second test printed stacks,
  crashed its worker and exited 1. Serial behavior matched. Temporary fixtures
  were removed; this remains evidence for unchanged code.
- The merge audit retained all 69 original selected paths/nodes and ran all six
  evaluation exclusions in evaluation-long. New files outside a selected
  directory still require explicit selection. Eight minutes is a measured
  target, not a hard job timeout.
- Before rebuilding locally, four learners failed catalog admission with
  `catalog_tokens.keys: expected array`. The matching cp313 release extension
  was rebuilt in 46.12 s and installed atomically; no Rust source changed.
  The focused comparison then passed in 45.42 s. The full file passed 10 tests
  in 151.36 s (comparison 82.03 s, diagnostics 66.86 s).
- QA's local experiment group passed 130 tests in 80.87 s with the release
  extension; `test_experiment_report.py` passed 17 in 16.14 s. Temporarily
  restoring complete-game/both-seat training failed the new row assertion
  (`610 == 16`) in 59.90 s. The fixture was restored. Local elapsed time does
  not establish Ubuntu timing.

External fact checked against setup-uv v6 on 2026-11-12: `prune-cache` defaults
to true and calls `uv cache prune --ci`. False removes the `-pruned` key
component, separating the new cache from the old immutable cache. Keys include
architecture, platform, Python and dependency-manifest hash. `save-cache: false`
skips saving; exact hits skip both pruning and saving. Sources:
[inputs](https://github.com/astral-sh/setup-uv/blob/v6/action.yml),
[restore](https://github.com/astral-sh/setup-uv/blob/v6/src/cache/restore-cache.ts),
[save](https://github.com/astral-sh/setup-uv/blob/v6/src/save-cache.ts).

Cache implementation checks: `actionlint .github/workflows/manabot-integration.yml`
passed; `uv run --no-sync pytest tests/infra/test_ci_selection.py::test_workflow_test_files_exist -q`
passed (1 test); `git diff --check` passed. Removing the four cache lines
reproduced the prior workflow exactly, including every install/build/test command.

Earlier measurements and gate history: `937f4f152:scratch/ci-under-8-minutes.md`.
Pre-compression investigation and command details:
`5f9a69184:scratch/ci-under-8-minutes.md` and `5f9a69184:scratch/qa.html`.

## Remaining delivery

Gate must record Ubuntu cache miss/save and subsequent hit/restore runs,
including Setup uv, Install dependencies, Post Setup uv and total job seconds.
Confirm warm installs avoid Torch/nvidia downloads, all selected tests pass
with debug assertions, and ordinary jobs finish below 480 s. Neither a cold
run alone nor a warm run alone establishes both costs.

If the fixture's resource deadline fails again, identify the expired clock
and charged/allowed seconds in `manabot/training/execution.py::remaining/check`
before adjusting an allowance. Its three clocks are launch, regime and stage;
`tests/training/test_learning_state.py` retains the existing error text. This
diagnostic change is deferred unless another failure warrants it.

Publish the fixture and cache repair in Task 38ef25, retaining #506 as history.
`scratch/pr-body.md`, `scratch/pr-title.txt` and `scratch/.pr-copy-ref` still
describe #506 and must be rewritten at delivery. PR-event timing and successful
post-merge verification remain necessary before completing the Task.

Compression: kept the direct fixture/cache implementation; removed repeated
plan explanations and stale QA cache guidance. Wave memory's statement that
Ubuntu fixture timing is unproved is superseded by run 43345024882; realign
should update it from this evidence (Wave context is read-only here).

Compression checks: `uv run --no-sync python scripts/check_loopflow_workflows.py`
— passed; `gh run view 43345024882 --json jobs` returned job/step timestamps;
`git diff --check` — passed. No executable behavior changed in this pass, so
tests were not rerun. `lf context` remains under both token and byte targets.
