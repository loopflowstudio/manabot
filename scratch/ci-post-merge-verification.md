# CI post-merge verification — 2026-11-12

Verification decision: **Fix forward.** PR #506 merged, but its exact
main push exposed a failing experiment fixture. The Task stays open. This is
new evidence after the successful PR run, not an absence of progress.
The fixture repair is now implemented and locally checked; current remaining
validation and delivery are in `ci-under-8-minutes.md`. The observations below
record the failed merge, not the repaired revision.

Implementation follow-up: the run subsequently completed with failure. Engine
Budget passed (00:59:34–01:10:59 UTC, 685 s); the experiment fixture is the
remaining failure. `gh run view 43341403551 --json status,conclusion,jobs`
confirmed this on 2026-11-13 UTC. The pending status below records what was
known at the verification decision.

## Revision and claims

`gh pr view 506 --json state,mergeCommit,mergedAt,url` reported MERGED at
`60664d1496089dd3161813c77dad36a1928fa15f`, at 2026-11-13T00:59:22Z.
`git fetch origin main` and `git rev-parse origin/main` resolved that same SHA.

The Task promises ordinary Ubuntu PR jobs below 480 seconds including setup,
without losing coverage or raising the 300-second single-test limit. Engine
Budget is exempt. The PR description also delivers MTG-263's measured balancing
and parallel-test scope. The demo note recorded no human demonstration or approval.

## Separate CI evidence

- PR event: `gh run view 43337105047 --json event,headSha,conclusion,jobs,url`
  reported successful `pull_request` at `1fc4e1b244f856e1a4192cc461c0edeabf0a7115`.
  Its slowest ordinary job was recovery, 349 seconds.
  https://github.com/loopflowstudio/manabot/actions/runs/43337105047
- Merge queue: `gh run list --event merge_group --limit 10 --json databaseId,event,headSha,conclusion,url`
  returned `[]`; no queue result is claimed.
- Post-merge event: `gh run view 43341403551 --json event,headSha,status,conclusion,url,jobs`
  reports `push` at the exact merge SHA. At this decision the overall run is
  **in progress**, with Engine Budget the only remaining job. Experiment has
  already **failed**; every other ordinary job succeeded. This is not a green
  push run regardless of the remaining exempt job's eventual result.
  https://github.com/loopflowstudio/manabot/actions/runs/43341403551

All ordinary jobs finished below 480 seconds: experiment 433 (failed),
evaluation 418, training 368, Rust 313, recovery 310, Python unit 308,
protocol 283, model 273, evaluation-long 253, Academy UI 191, clean-machine
play 170, Academy Integration 146, semantic conformance 112, detection 5.
The slowest successful ordinary job leaves 62 seconds of timing margin.

## Concrete gap

`gh api repos/loopflowstudio/manabot/actions/jobs/131610394198/logs --allow-escape-sequences`
provided the completed experiment job's log while the overall run was pending:

```text
FAILED tests/training/test_experiment_runner.py::test_tiny_comparison_evaluates_live_and_notebook_refresh_preserves_edits
experiment incomplete: None
training attempt 1 failed: RuntimeError: learner exited 1: TimeoutError: training resource wall deadline exceeded
1 failed, 129 passed in 307.66s (0:05:07)
```

The failed test took 176.21 seconds. Four xdist workers ran 130 tests.
This was the fixture's training resource deadline, not the 300-second pytest
watchdog. `demo_declaration()` in the merged test sets 60 seconds for a regime
and 30 seconds for a stage, with two streams, eight transitions and four real
training/monitoring attempts. `manabot/training/execution.py::check()` raises
the observed error when its resource budget expires. The log does not identify
which constituent clock expired; do not claim it does.

Repair requirements established at verification: make this live experiment/notebook fixture reliable in
the parallel group by bounding its workload and choosing an adequate fixture
resource allowance. Preserve four real attempts, live monitoring overlap,
accounting, notebook edits/outputs and resume assertions. Keep the 300-second
single-test limit and eight-minute job target. Validate the focused file and
the Ubuntu group, then ship a fix-forward PR in Task 38ef25 from main, retaining
#506 as delivery history. No rerun was used to dismiss the observed failure.

## Other claims checked on main

A read-only `uv run --no-sync python` audit used `git show` for the merge,
its first parent and PR head. It printed:

```text
All 69 original test paths/nodes retained; all 6 evaluation exclusions run in evaluation-long.
All nine changed workflow, test, dependency and guide files exactly match the PR and local checkout.
```

The nine files are both Integration workflows, the three changed Python test
files, `pyproject.toml`, `uv.lock`, `TESTING.md` and `docs/development.md`.
PR-to-main differences are experiment documentation/skills only. Main retains
the single dependency sync/build, six logical-worker groups, per-worker
300-second watchdog, both admission outcomes and the real-worker one-update
assertion. Earlier controlled watchdog evidence therefore remains applicable.

`uv run --no-sync pytest tests/infra/test_ci_selection.py -n 2 -q` printed
`53 passed in 0.44s`. No local whole-suite or synthetic-load run was started.
The hosted push already performs the actual main build and full selected tests;
this repository does not build an `lf` binary or a Mac app.

The local `gh run watch 43341403551 --exit-status --interval 30` observer was
stopped with SIGINT and reaped (exit 130). The remote CI was not cancelled.
No local server, watcher or other background process remains from this pass.
