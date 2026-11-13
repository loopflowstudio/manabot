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
per stream on two streams using `exact-quota`. Every saved `TrainingRun` must contain one
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

## CPU Torch implementation (2026-11-12)

Integration now selects the existing `play-runtime` group with default groups
disabled and uses setup-uv defaults. The retained-wheel repair is removed.
The fixture and all test/build commands remain unchanged. Basis measured on
2026-11-12:

- Each Integration job downloads 3,046 MiB of wheels (job log, run 43345024882).
  About 2.8 GiB is `nvidia-*`, `triton` and `torch==2.15.0+cu132`, which a
  hosted CPU runner cannot use. The slow job spent 2 m 59 s on them; five
  peers took 17–30 s for the same downloads. The tail is download variance.
- Repository Actions caches hold 8.2 GB of GitHub's 10 GB limit in 357 entries
  (`gh api repos/loopflowstudio/manabot/actions/cache/usage`). Rust caches are
  105–127 MB each; today's uv cache is 0–4 MB. A retained wheel cache is at
  least the 3 GB of already-compressed wheels, saved again per ref whenever
  `uv.lock` changes. It cannot fit without evicting the Rust caches that keep
  the 52–80 s extension build from becoming a cold one.
- A retained cache turns a 2 s restore into a multi-GB restore on all six
  groups to protect against a tail on one; the median gets no faster.
- The lockfile already resolves CPU Torch for these extras:
  `uv export --frozen --no-default-groups --group play-runtime --extra dev
  --extra play --extra notebook --extra artifacts --extra reports --extra llm
  --extra evaluation` yields `torch==2.15.0+cpu` on Linux, `torch==2.15.0` on
  macOS, and no `nvidia-*` or `triton`. `baseline.yml` installs this way now.
  `questions.md` assumed CPU Torch "changes the build"; it needs no dependency
  or lock change, only the two group flags on Integration's `uv sync`.

### Delete — do not maintain

- Removed Integration's `prune-cache: false`, matrix `save-cache` and their
  comment; removed the matching wheel-cache ownership and cold/warm protocol
  in `docs/development.md` and `.lf/skills/qa.md`. No exclusive tests or fixtures.
- Preserve the bounded experiment fixture, all selected test nodes, debug
  extension build, per-worker watchdog, dependency definitions and lockfile.
- Preserve QA's hosted-CI commands and within-run install comparison.

### Checks and remaining implementation

CUDA review found no selected test requiring a CUDA-built Torch on a CPU
runner. Actual GPU cases already guard on `torch.cuda.is_available()`;
other cases validate declarations, mock runtime/launch boundaries or inspect
precision contexts. The vLLM fixture uses a local tokenizer and mocked HTTP,
not a local GPU server. The CPU build still exposes the precision controls;
focused precision checks and hosted tests verify that boundary.

Checks: `actionlint .github/workflows/manabot-integration.yml` passed;
locked sync of the exact CI extras passed; focused file-selection, TF32/autocast
and vLLM HTTP-boundary tests passed (13 in 3.58 s); workflow validation passed;
`git diff --check` passed. Initial collection lacked `huggingface_hub`; syncing
the declared CI extras resolved it. Frozen export selects Linux Torch
`2.15.0+cpu`, with no NVIDIA/Triton dependencies. A diff audit confirms the
build/test commands, bounded fixture, dependency definitions and lockfile
are unchanged.

Pending: one full hosted Ubuntu dispatch showing CPU Torch installation
without NVIDIA/Triton downloads and every ordinary job below 480 s.
PR-event and post-merge evidence belong to delivery.

What survives: same Torch version, same lockfile, same test selections, debug
extension build, watchdog. What changes on purpose: Integration stops importing
the cu132 wheel. Python unit and Academy Integration still sync the default
`training-runtime` group, so a hosted job still imports the Linux training
build. Counterexample to check during implement: a selected test that needs a
CUDA-build-only component on a CPU runner; the 17 test files that mention
`cuda`, `torch.compile` or `triton` (`tests/llm/test_llm_vllm.py` among them)
are the places to look. If one exists, keep that group on the default install rather than
restoring cache retention.

Known limit, not changed here: Python unit (308 s on the merged push) and
Academy Integration (146 s) keep the 3 GiB download and its tail. The observed
tail added 163 s, which would put Python unit at about 471 s.

The fixture no longer uses complete-game training or the 30/60-second guards.
Test selections, the native debug build, the 300-second watchdog, dependency
groups and lockfile are unchanged. Wave memory's cache-retention bullet
describes the superseded repair; realign should replace it with the facts above.

## Reconciled with upstream (2026-11-12)

The checkout includes main through `f1b0109ae` (#504, mana-payment engine
optimization), merged here before `b8ed0970b`. A fresh fetch found no main
commits missing from this checkout. That engine change postdates the fixture
QA revision `4db5cd64b`; its timing cannot be inferred from the old run.
The fixture repair remains present after the merge. The next Ubuntu run must exercise both the current engine and the CPU runtime
change; the prior run does not establish their timing.

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

After the CPU Torch change, one hosted Ubuntu run on the current head must show
every ordinary job below 480 s with all selected tests passing under debug
assertions. It also supplies the first timing with #504's engine change.

If the fixture's resource deadline fails again, identify the expired clock
and charged/allowed seconds in `manabot/training/execution.py::remaining/check`
before adjusting an allowance. Its three clocks are launch, regime and stage;
`tests/training/test_learning_state.py` retains the existing error text. This
diagnostic change is deferred unless another failure warrants it.

Publish the fixture and CPU runtime repair in Task 38ef25, retaining #506 as history.
`scratch/pr-body.md`, `scratch/pr-title.txt` and `scratch/.pr-copy-ref` still
describe #506 and must be rewritten at delivery. PR-event timing and successful
post-merge verification remain necessary before completing the Task.

No open product choice: nobody has asked that Integration run the CUDA build,
and the stated outcome counts install and cache overhead against the budget.
