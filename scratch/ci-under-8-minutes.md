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
- `.github/workflows/manabot-integration.yml` selects the existing CPU Torch
  via `--no-default-groups --group play-runtime`, with setup-uv defaults.
  Tests, selections, native debug build, dependency definitions and lockfile
  are unchanged. `docs/development.md` owns the job budget and balancing;
  `TESTING.md` owns parallel isolation; `.lf/skills/qa.md` owns hosted checks.

No selected test needed a CUDA-built Torch on a CPU host: actual GPU cases
already guard on hardware availability; precision controls still exist on
CPU Torch; vLLM tests use a local tokenizer and mocked HTTP. The hosted CPU
run retained the prior model/evaluation pass and skip counts.

### Delete — do not maintain

Completed: removed retained-wheel cache overrides and their cold/warm protocol.
This pass removes #506's obsolete PR-copy files, demo and post-merge notes,
consolidating their evidence here; history remains at `e22af480b:scratch/`.
The general QA skill no longer prescribes this fixture's one-off mutation
experiment. No code, test or fixture deletion remains. Keep the bounded
fixture, all selected nodes, debug assertions, watchdog and lockfile.

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
or post-merge proof, and does not guarantee runner timing. Python unit and
Academy Integration still use the default CUDA runtime; the observed download
tail of 163 s would put the merged push's 308-second Python unit job near 471 s.

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

## Remaining delivery and memory reconciliation

Publish the fixture/CPU repair in Task 38ef25, keeping #506 as delivery history.
Generate fresh PR copy; the removed copy described #506. Retain MTG-263 scope
and before/after evidence. Obtain PR-event timing and successful post-merge
verification before completing the Task. No local process remains running.

If the resource deadline fails again, identify the expired launch/regime/stage
clock and charged/allowed seconds in `manabot/training/execution.py::remaining/check`
before changing an allowance. Do not add that diagnostic path speculatively.

Wave memory is read-only in this step. Realign should replace its superseded
cache-retention claim with: CPU Torch and setup-uv defaults passed dispatch
43348710097 including #504; ordinary max 406 s; installs 5–6 s with no
NVIDIA/Triton; PR-event and post-merge evidence remain outstanding.

Compression: runtime code is already direct; no behavior changes or test reruns
are needed. Removed duplicate/stale context and the fixture-specific QA recipe.

Check: `uv run --no-sync python scripts/check_loopflow_workflows.py` — all
nine graphs compiled and validation passed; `git diff --check` passed.
`lf context` reports memory and scratch below both configured limits.
