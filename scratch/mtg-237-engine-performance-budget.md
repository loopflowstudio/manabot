# MTG-237: Engine performance budget in CI

Design from Jack Heart's Task brief and 2026-11-02 concept review
(Performance section, `scratch/engine-foundations.html` in the T6 checkout).
PR: https://github.com/loopflowstudio/manabot/pull/413.

## Outcome and ownership

`Engine Budget` builds release `bench_search` and `bench_steps` on one CI
runner, then builds the committed reference with the same benchmarks there.
The candidate fails above 10% greater duration or if either run differs from
the committed digests. Recorded absolute timings are orientation only.

- Search: seeds 1–3, 40 decisions each, 16 worlds × 4 rollouts, 2,000-step
  playouts. Budget worst reply and summed total. Hash every ordered score,
  framed by action count, simulation count and chosen index: little-endian
  IEEE-754 bits into BLAKE3, first 64 bits as the existing integer output.
- Throughput: 512 seeded random-legal games through native `Env::reset`/`step`
  with trivial-decision skipping; roughly 100k steps, a per-game 2,000-step
  cap, and one decision digest. Both benchmarks share `examples/curated/mod.rs`.
- `scripts/engine_budget.py compare|record` owns comparison and adoption.
  `Run` is measured evidence; `Baseline` owns reference, tolerance and reason.
  CI owns execution and transport, with no additional lifecycle state.
- `.github/workflows/engine-budget.yml`, called by `ci.yml`, measures the
  durable PR head. Candidate output and `measurement.json` are recorded
  before building the reference; the final always-run upload retains them
  even if the reference cannot build. Explicit Bash propagates pipeline
  failures through `tee`; Tests Result carries the failed job to the PR.
- `.cache/engine-budget` is the candidate directory, with `reference/` below.
  `compare` defaults to those paths. `record --reason` adopts a downloaded
  candidate artifact and preserves the file's tolerance. Reference digest
  drift gets a distinct baseline-file failure row.
- `measure=<commit>` dispatch selects only this job, measuring that commit
  with the dispatched ref's benchmarks and uploading without comparison.

The development guide owns the maintainer procedure and historical T4/T5/T6
measurements. The [demo record](engine-budget-demo.md) owns hosted results,
artifact paths, failure evidence and proof limits.

## Completed simplification

No remaining code deletion targets. Removed the winner/simulation-only digest
in `e5615edcc` and implicit Bash execution in `79c623428`; the full-score
and pipefail paths have regression coverage. The Clean-machine Play
120 s wait remains a launch proof, not an engine latency budget.

Working notes retain current decisions and evidence summaries rather than
completed iteration instructions. Historical plans live at
`4b5e9c0e4:scratch/mtg-237-engine-performance-budget.md` and earlier conceptual
counterexamples at `311218a02:scratch/mtg-237-engine-performance-budget.md`.
Pre-curation uncommitted notes are preserved under
`.cache/engine-budget-demo/pre-compress-4b5e9c0e4/`.

## Decisions and review boundary

Jack Heart's brief accepts a 10% budget for both workloads, deliberate reference
moves, identical digests, a controlled 20% slowdown fixture, one-time T4/T5/T6
measurements, Python through uv, and no induced load.

Headless interpretations still subject to Jack Heart's PR review:

- A committed reference commit is rebuilt A/B on the same runner. The same
  engine measured 6.8k–10.3k steps/s across runner CPU models; recorded wall
  times cannot govern another runner. Two EPYC 7763 runs agreed within 2.5%.
- Native random-legal throughput measures the training environment's engine
  step. It is not comparable with rul-9's retired Python Command runner at
  588.5 steps/s. Candidate-first ordering is retained for artifact recovery;
  change it only with evidence of bias beyond the tolerance.

A moved digest, accepted cost or retained speedup moves the reference using
that candidate's own artifact and a reason. Historical winner-only digests
cannot establish full-score equality. Unexplained failures with matching
engine/digests warrant a rerun, not a tolerance change. No dedicated runner
or engine optimization is selected by this Task.

Concept review at `9452c2115` found no open product choice under these
interpretations. The original cross-run timings, incomplete digest and
false-green `tee` pipeline remain counterexamples; their repairs stand.

Headless concept review on 2026-11-02, after the current reconciliation:
**nothing to change and no open product choice** under the recorded
interpretations. The maintainer's normal path is a PR check with separate
timing and correctness rows. On an unexplained failure, investigate or rerun;
on a deliberate reference move, download the candidate and use
`record --reason`, then review the committed baseline diff. This preserves
the measurement identity and tolerance while replacing the reference and its
measurements. If reference compilation fails, the candidate recording survives
for that same adoption path. `Run`, `Baseline` and CI execution already express
these responsibilities without another approval object or persistent lifecycle.
The retained speedup changes the reference, not this interaction or model.
No implementation direction follows from this review. The then-pending hosted
comparison, artifact verification and PR description correction are now complete
as recorded below. This conclusion grants
neither human acceptance of the interpretations nor shipping approval.

## Current implementation and remaining work

The production baseline now adopts run **41555845634**, candidate
`4b5e9c0e4555627664721e7a4e6c909120cd98ac`, whose native source equals merged
main `51fbacbbf`. Delegated demo judgment keeps the measured speedup and
applies the existing reference-moving rule. On one runner: 30.26% less worst
reply time, 28.36% less total search time and 36.65% less time per step; all
full-score/step digests and counts match. Tolerance remains 0.1.
`record --reason` used `.cache/engine-budget-demo/pass-3/pr-head/`; exact
artifact identity and every measurement were reconciled. Current-reference
docs reflect this adoption and retain the earlier accepted-cost evidence.
No runtime, benchmark or workflow change is needed for this iteration.

Reconciliation on 2026-11-02 confirmed the production baseline exactly matches
the downloaded candidate artifact, including identity and every measurement.
Wave memory now reflects this measured result. Origin main remains `51fbacbbf`;
the engine, benchmarks, comparator, workflow and focused tests are unchanged
since measured candidate `4b5e9c0e4`.

PR #413 now publishes head `fb590b3f9` with adoption, docs and the corrected
description. Run **41558082651** was cancelled when this head replaced
`7cfd0f17e`. Decision pass 4 verified **41558633938**: Engine Budget passed
against adopted reference `4b5e9c0e4`, all digests/counts matched, both summaries
and pipefail were present, and the downloaded artifact named the exact head.
Replayed comparison and temporary adoption passed with unchanged tolerance.
The [demo record](engine-budget-demo.md) contains measurements and exact proof
paths. Delegated judgment is **advance**; no implementation change is needed.
Main integration and Tests Result were still pending at verification; inspect
their final results before delivery. Main-push verification is post-merge. Hosted
reference-build failure has only a controlled local proof; failed-job transport
is proven after hosted comparison failure.

Jack Heart's steer authorizes publication and the interactive demo boundary,
not shipping. Delegated headless judgments are not new human feedback. Merge
and Task completion remain outside the accepted scope. Same-runner and native
throughput interpretations remain subject to PR review; no new product choice
arose from this reconciliation.

Check: `uv run --no-project python` artifact reconciliation, `git diff --check`
and `lf context` pass; unchanged implementation retains 65 focused test passes;
hosted run 41558633938 result/artifact verification passed at demo.

## Gate — 2026-11-02, pass 4

Reviewed the branch against checkout base `51fbacbbf`, including the uncommitted
pass-4 demo notes. No implementation defect found. Clarified the development
guide: matching digests and unchanged engine code warrant investigation and a
rerun, not an automatic conclusion of runner noise or a wider tolerance.

Validation plan by changed path:

- `.github/ci/select-tests.sh`, both CI workflows, `scripts/engine_budget.py`
  and their two infra test modules reach selection, comparison, recording and
  pipeline failure propagation. Their bytes are unchanged from `4b5e9c0e4`;
  retain the 65 focused passes and earlier actionlint/Ruff results.
- `managym/Cargo.toml` and the three benchmark source files reach debug digest
  tests, formatting/Clippy and hosted release measurements. They and all native
  source are unchanged from that head; retain earlier debug/static checks.
  Current-head hosted Rust Tests and Engine Budget both passed.
- The adopted `managym/perf/baseline.json` reaches Engine Budget and the baseline
  fixture assertion. Current-head hosted Python Unit Tests and Engine Budget
  passed; pass-4 artifact reconciliation verifies the adopted reference. Reuse
  that evidence rather than dispatch another measurement.
- Agent guidance, development docs, Wave memory and scratch notes have no code
  reach. This gate edits only the guide and local notes; check whitespace and
  selection, with no browser build or native extension rebuild required.

Checks: `git diff --name-only -z 4b5e9c0e4 | bash .github/ci/select-tests.sh`
selected only Engine Budget (the changed baseline), already passed in hosted
run 41558633938. `git diff --check` passed. No duplicated test/build run or
artificial load was started.

`gh pr view 413 --json headRefOid,baseRefName,state,statusCheckRollup` confirmed
open head `fb590b3f9`, with debug Rust, Python, conformance, Academy, clean Play,
long integration and package/install checks successful. Main integration was
still executing its Two-seat and Etude evaluation step; Tests Result is pending,
not passed or timed out. CI owns those remaining results.

Delivery context: the current PR body still says pass-4 demo verification is
pending. The next publication step must reconcile that wording with the completed
demo evidence and retain the main-integration/Tests Result qualification until
they finish. Main-push verification remains post-merge. Gate does not authorize
merge or Task completion; Jack Heart's publication/demo boundary still applies.
