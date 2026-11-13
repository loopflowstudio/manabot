# CI under 8 minutes (Task 38ef25; includes MTG-263)

## Outcome and constraints

Every ordinary Ubuntu PR job finishes below 480 seconds, from job start to
job end including install, build and cache overhead. Preserve coverage and
the 300-second single-test watchdog; never induce load. Engine Budget keeps
its same-runner comparison and is exempt. The macOS nightly is outside the budget.

Jack Heart's Task comment (2026-11-12): "Read its brief and deliver its outcome
here; say so in the PR so MTG-263 can be closed with it."

MTG-263 brief:

"Measure per-file durations, then either balance file shards across matrix
groups or adopt pytest-xdist where tests are process-safe (check tmp dirs,
ports, DuckDB files, subprocess fixtures). Target: no group above ~8 min wall
time; report before/after from CI runs. Do not drop or weaken tests."

## Current implementation

PR #506 delivered six measured Integration groups, logical-core pytest workers,
one dependency sync/native debug build per job, the worker watchdog,
injected-time admission checks and a two-stream real-worker size smoke.
Its merged push failed the separate live experiment/notebook fixture.

The repair has two parts:

- `tests/training/test_experiment_runner.py::demo_declaration` uses one-seat
  `exact-quota` collection: one update, two streams, eight learner rows per
  stream. All four saved `TrainingRun` receipts must contain one stage, one
  update and 16 learner transitions. Four live complete-game monitoring
  cohorts, overlap, accounting, notebook edits/outputs and resume checks remain.
  Stage and regime allowances are 120-second hang guards. Production defaults
  and the shared WDL baseline are unchanged. `TrainSelfPlay.batch_closure` in
  `manabot/training/regime.py` owns the contract: complete-game transitions
  are a floor; exact-quota needs one learning seat.
- Integration and Python unit select the existing CPU Torch
  via `--no-default-groups --group play-runtime`, with setup-uv defaults.
  Tests, selections, native debug build, dependency definitions and lockfile
  are unchanged. Python unit repeats the runtime flags on `uv run` to preserve
  that selection. `docs/development.md` owns the job budget and balancing;
  `TESTING.md` owns parallel isolation; `.lf/skills/qa.md` owns hosted checks.

No selected test needed a CUDA-built Torch on a CPU host: actual GPU cases
already guard on hardware availability; precision controls still exist on
CPU Torch; vLLM tests use a local tokenizer and mocked HTTP. The hosted CPU
run retained the prior model/evaluation pass and skip counts.

Retained-wheel cache overrides and their cold/warm protocol were removed.
Superseded #506 PR copy, demo and verification notes remain in
`e22af480b:scratch/`; their relevant evidence is consolidated below.

## Hosted evidence (observed 2026-11-12)

| Actions run | Revision / event | Result and slowest ordinary job |
|---|---|---|
| 43312163427 | main baseline | green; Integration core 1443 s |
| 43327574361 | `45fe9f866`, dispatch | green; experiment 404 s |
| 43332039325 | `f90ae2893`, dispatch | green; recovery 416 s |
| 43334548912 | `30cfcb2e6`, dispatch | green; evaluation 448 s |
| 43337105047 | `1fc4e1b24`, PR #506 | green; recovery 349 s |
| 43341403551 | `60664d149`, merged push | experiment failed at 433 s; others passed, max 418 s |
| 43345024882 | `4db5cd64b`, fixture repair dispatch | green; experiment 549 s, over budget |
| 43348710097 | `a1efea11b`, CPU Torch dispatch | green; Rust 406 s, Integration recovery 350 s |

Runs: `https://github.com/loopflowstudio/manabot/actions/runs/<id>`.
#506 merged at `60664d1496089dd3161813c77dad36a1928fa15f` on
2026-11-13T00:59:22Z. No merge-queue result or human demo/approval was recorded.
Engine Budget passed in 613 s on that PR and 685 s on its merged push.

The merged experiment group reported 129 passed, one failed in 307.66 s:

```text
training attempt 1 failed: RuntimeError: learner exited 1: TimeoutError: training resource wall deadline exceeded
```

The comparison took 176.21 s; its 30/60-second training guards expired, not
the 300-second watchdog. The log does not identify which resource clock
expired or separate collection from scheduling. The bounded fixture then
passed Ubuntu debug/xdist in 151.70 s in run 43345024882, but installation
cost 181 s (peers 17–18 s), build 66 s and tests 284 s: 549 s total.

That run downloaded 3,046 MiB per Integration job, about 2.8 GiB of CUDA Torch,
NVIDIA and Triton wheels. A pruned uv-cache hit did not avoid those downloads.
At investigation, Actions caches occupied 8.2 GB of a 10 GB limit in 357 entries;
Rust caches were 105–127 MB and uv caches 0–4 MB. Retaining multi-GB wheels
would add restore/save cost and eviction pressure. CPU Torch was already locked.
Earlier recovery installation also took 145 s against peers' 17–31 s;
unchanged evaluation-long tests varied from 153 to 249 s. Margin matters.

[CPU dispatch 43348710097](https://github.com/loopflowstudio/manabot/actions/runs/43348710097)
at `a1efea11bab6669ff9dfdc06f79511f7139555f8` also includes main's #504
mana-payment optimization (`f1b0109ae`). All six groups installed Torch
`2.15.0+cpu` in 5–6 s, with no NVIDIA/Triton log entries and native dev builds.
Model passed 823 with 5 existing skips; training 346; recovery 49; experiment
130; evaluation 288 with 1 existing skip; evaluation-long 66. The repaired
comparison took 138.45 s. Experiment build took 70 s, tests 239 s, install 5 s.

| Job | Total seconds including setup and cache |
|---|---:|
| Rust Tests | 406 |
| Integration recovery | 350 |
| Integration training | 344 |
| Integration experiment | 340 |
| Integration evaluation-long | 293 |
| Integration model | 287 |
| Integration evaluation | 277 |
| Protocol v1 | 249 |
| Academy Integration | 178 |
| Academy UI | 175 |
| Clean-machine Play | 149 |
| Semantic conformance | 119 |
| Python unit | 89 |
| Detect Changes | 5 |
| Tests Result | 2 |
| Engine Budget (exempt) | 571 |

The maximum ordinary job leaves 74 s. This is dispatch evidence, not PR-event
or post-merge proof, and does not guarantee runner timing. At that revision,
Python unit and Academy Integration still used the default CUDA runtime. Adding
the observed 163-second download tail to the merged push's 308-second Python unit job
would put it near 471 s; that is a timing-risk estimate, not an observed job.

## Reconciled revision

After the CPU dispatch, main's `e82a20033` release preparation was merged into
this checkout (`38e3e5557`). It changes package/crate versions from 0.2.2 to
0.2.3, the matching managym requirement, lockfile workspace versions and release
documentation. It changes neither the Torch selection nor engine/training
behavior. The workflow and repaired fixture match the measured revision;
the version and lock changes can still affect build/cache timing, so dispatch
43348710097 does not certify the final PR revision.

GitHub inspection on 2026-11-12 confirmed main at `e82a20033`, dispatch
43348710097 successful at `a1efea11b`, and only merged #506 for this branch.
The CPU installation change, docs and hosted QA guidance requested in the
iteration feedback are implemented. No further implementation mismatch or
open product decision was found. Infrastructure memory now records the CPU
result and removes the superseded cache-retention guidance.

## Final-revision QA (2026-11-12)

[Dispatch 43350950910](https://github.com/loopflowstudio/manabot/actions/runs/43350950910)
at `38e3e5557` (with v0.2.3 merge): green; ordinary max Integration evaluation
399 s, Rust 396 s. Python unit spent 195 s installing: engine build ended
about 105 s in, then CUDA/NVIDIA/Triton downloads ran another ~88 s for 4 s of
tests. `ef0839c9c` gives Python unit the same locked CPU selection on `uv sync`
and its `uv run` (otherwise `uv run` re-syncs the default CUDA group); the
website step is its own project and is unchanged.
[Dispatch 43353030839](https://github.com/loopflowstudio/manabot/actions/runs/43353030839)
at `ef0839c9c`: green; Python unit 104 s (install 89 s, Torch `2.15.0+cpu`,
157 passed/1 skipped and 21 passed, same as before). Ordinary max Rust 406 s,
Integration max experiment 335 s (comparison test 140 s in the prior run).
Engine Budget (exempt) took 901 and 906 s in these two runs, against
571–685 s earlier; its paired benchmark step alone was 715 s. Not investigated.
Academy Integration still installs the default runtime (20 s here).
Still required: PR-event timing and post-merge verification.

## Other retained checks

- Exact locked CI sync, frozen export, actionlint and workflow validation passed.
  Export selected Linux Torch `2.15.0+cpu`, macOS `2.15.0`, no NVIDIA/Triton.
  Focused selection/TF32/autocast/vLLM checks: 13 passed in 3.58 s. Initial
  collection lacked `huggingface_hub`; syncing the declared extras resolved it.
- Before rebuilding the stale local cp313 extension, four learners failed with
  `catalog_tokens.keys: expected array`. After atomic replacement, the focused
  comparison passed in 45.42 s and the full file passed 10 in 151.36 s.
  Local QA's experiment group passed 130 in 80.87 s; report tests 17 in 16.14 s.
  These release-extension Mac timings do not establish Ubuntu performance.
- Temporarily restoring complete-game/both-seat collection failed the receipt
  assertion (`610 == 16`) in 59.90 s. The fixture was restored.
- Controlled watchdog proof: temporary 5-second limit; five tests totaling
  11 s passed with two workers, while a 30-second test printed stacks, crashed
  its worker and exited 1. Serial results matched; temporary files were removed.
- The merge audit retained all 69 original selected paths/nodes and routed all
  six evaluation exclusions to evaluation-long. Selection tests passed 53 in
  0.44 s. New files outside selected directories still need explicit selection.

Detailed investigation and source links: `5f9a69184:scratch/ci-under-8-minutes.md`.
Pre-consolidation evidence, QA and delivery history: `e22af480b:scratch/`.

## Remaining delivery

The fixture/CPU repair is published as PR #508 at `6a453e3f4`, keeping #506
as delivery history. Its description retains MTG-263 scope and before/after
evidence. Obtain PR-event timing on the final revision and
successful post-merge verification before completing the Task: every selected
ordinary Ubuntu job must pass below 480 s including setup and cache overhead.
Retain the six Integration groups' CPU install logs and job totals; Engine
Budget remains exempt. No additional implementation is currently identified.

If the resource deadline fails again, identify the expired launch/regime/stage
clock and charged/allowed seconds in `manabot/training/execution.py::remaining/check`
before changing an allowance. Do not add that diagnostic path speculatively.

Reconciliation reused the existing focused and hosted checks; no tests were
rerun for these context-only corrections. Check: `git diff --check` passed;
`lf context` reports memory and scratch below both configured limits.

## Gate (2026-11-12)

Reviewed the diff against `origin/main`, including the fixture, both workflows,
hosted QA guidance and documentation. No implementation defect found. The local
tree differs from measured `ef0839c9c` only in memory and scratch evidence;
reuse its full hosted matrix and the recorded fixture checks. The changed-path
selector requests all CI jobs because `ci.yml` changed; those run on the PR,
not as a second local full-suite pass. No browser interaction changed.

- `git diff --name-only -z origin/main...HEAD | bash .github/ci/select-tests.sh` — all nine job selectors true.
- `uv run --no-sync pytest tests/infra/test_ci_selection.py -q` — 53 passed in 0.33 s.
- `uv run --no-sync ruff check --fix tests/training/test_experiment_runner.py` — passed, no edits.
- `uv run --no-sync ruff format tests/training/test_experiment_runner.py` — unchanged.
- `actionlint .github/workflows/ci.yml .github/workflows/manabot-integration.yml` — passed.
- `uv run --no-sync python scripts/check_loopflow_workflows.py` — all nine graphs compiled; boundaries valid.
- `uv lock --check` — passed, 176 packages resolved.
- `git diff --check` — passed after the gate's evidence-only edits.
- `gh run view 43353030839 --json conclusion,headSha,jobs` — rechecked green dispatch, ordinary max 406 s; unchanged test/workflow inputs reused.

PR-event run `43355252190` is already in progress on `6a453e3f4`; no additional
run was launched. Successful post-merge verification remains a delivery check.
