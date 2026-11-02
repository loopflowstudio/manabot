# MTG-237: Engine performance budget in CI

Design from Jack Heart's concept review of 2026-11-02 (Performance section,
`scratch/engine-foundations.html` in the T6 checkout) and the Task brief.

## Outcome

- `Engine Budget` CI job (`.github/workflows/engine-budget.yml`, called from
  `ci.yml`, selected by the engine, the benchmarks, the baseline and the
  comparator) builds `bench_search` and `bench_steps` in release on a runner
  of its own, rebuilds the reference commit of `managym/perf/baseline.json`
  with the same benchmarks on that runner, and fails the PR when the
  candidate is more than 10% slower than the reference or a digest differs
  (runner CPU models vary too much for absolute numbers; see questions.md).
- `bench_search`: unchanged workload (seeds 1-3, 40 decisions, 16x4 flat
  search), summary now sums totals. `bench_steps`: 512 seeded random-legal
  games of the curated matchup through `Env::reset`/`step` with skip_trivial,
  ~100k steps, one digest over every decision. Decks shared in
  `examples/curated/mod.rs`.
- `scripts/engine_budget.py compare|record` (stdlib only) read run
  directories (`bench_search.jsonl` + `bench_steps.jsonl`): the candidate's
  at `.cache/engine-budget`, the reference's under it at `reference/`, both
  the defaults, so CI's compare step takes no flags. Every job run uploads
  that directory as the `engine-budget` artifact with `measurement.json`
  (candidate commit, ISO date, run URL, runner CPU); `record --reason`
  rewrites the baseline from a downloaded one. Tests in
  `tests/infra/test_engine_budget.py` include the 20% slowdown fixture and
  recording from an artifact.
- Manual dispatch with `measure=<commit>` benchmarks a commit that has no PR
  with this ref's benchmarks and uploads the same artifact without
  comparing: the one-time T4/T5/T6 measurement.

## Delete — do not maintain

Nothing is replaced. The Clean-machine Play 120 s wait stays a launch proof,
not a latency budget.

## Decisions

- Accepted (Task brief, Jack Heart, 2026-11-02): both workloads budgeted at
  10% against a committed file; digests asserted identical; a 20% slowdown
  fixture; one-time T4/T5/T6 measurement; `uv` only; no induced load.
- Implied by the accepted outcome (concept review 2026-11-02); not yet
  confirmed by Jack Heart, whose PR review decides it (no PR exists yet):
  "committed baseline" is read as a
  committed *reference commit* rebuilt and measured A/B on the same runner,
  not committed absolute numbers. Evidence: one engine spread 6.8k-10.3k
  steps/s across three runner CPU models (far outside 10%); two runs on one
  model agreed within 2.5%. The file's figures are orientation only.
- Implied by the accepted purpose (same review), pending the same PR review:
  the headless workload is native `Env::step`
  random-legal play (512 games, skip_trivial), not the rul-9 Python
  semantic-Command runner (deleted in #343). Its steps/s is its own
  baseline, not comparable with rul-9's 588.5.
- Any rules change reaching the curated decks moves the random-play digest
  and costs one recording per such PR; accepted as the price of (3). A kept
  speedup is the third reason to move the reference (concept review).

## Concept reviews (2026-11-02, headless; 5c8b0f2db, c8cbc5421 and the third below)

Source: Jack Heart's Performance section ("the PR fails if worst reply or
total exceeds the recorded baseline by more than 10%; a PR that moves the
baseline records why; a headless steps/s workload alongside it; measure T4
and T5 once"; purpose "a CI budget keeps it gone"; order of work "each lands
with its bench_search delta in the PR body"). Both reviews found a mismatch
settled by the accepted decisions, with no open product choice:

- First (fixed in 45f1b4627): moving the reference needed a second `measure`
  dispatch and a seven-flag command, and keeping a speedup was not a
  documented move. Now *every Engine Budget run is a recording of its
  candidate; a PR adopts one by committing it with a reason*: the artifact's
  `measurement.json`, `record --reason`, three documented reasons (rules
  change reaching the decks, accepted cost, kept speedup; so MTG-236's PRs
  record their head and the gain is budgeted), `measure` confined to commits
  without a PR.
- Second (fixed in 2f3fe99e1): on `pull_request` the default checkout is
  GitHub's ephemeral `refs/pull/N/merge`, so a recording from a PR run would
  name a commit the next run's "Check out the other commit" cannot fetch.
  The job now checks out `github.event.pull_request.head.sha || github.sha`
  (PR head; pushed or dispatched SHA otherwise), so the measured tree and the
  recorded commit are one durable commit. Detect Changes still diffs the
  merge commit. The `measure` and push paths were never affected, which is
  why the T4 recording and the validating dispatch passed.
- Third (after a91e930ef; next implement direction, not yet in code): the
  job builds and runs the reference *before* it runs the candidate, so when
  the reference commit does not compile with this ref's benchmarks (the
  `Env` surface change the docs already name, which they say "must keep the
  reference commit compiling with the new benchmarks or move the reference
  in the same PR") the job stops at "Build the reference benchmarks", the
  candidate is never measured, and the artifact is empty. That PR cannot
  move the reference through its own run; its only way is a `measure`
  dispatch of its head, which the docs reserve for commits without a PR.
  "Every run is a recording of its candidate" implies the opposite order:
  build and run the candidate, write `measurement.json`, then build and run
  the reference and compare. Then any run whose candidate builds is a
  recording, including this one, and the documented sentence holds through
  the ordinary adoption. The reference's output moves from the "Record the
  measurement" summary block to after the reference runs. Order on one VM is
  not a bias the 10% tolerance notices (two runs of one engine agreed within
  2.5%); questions.md records the assumption. Two clarifications of the same
  contract: the three reasons to move the reference are really "a digest
  moved" (rules reaching the decks *or* the workload itself: the benchmarks
  or the curated lists, which `curated/mod.rs` already says change only with
  the baseline), an accepted cost, and a kept speedup; and compare's
  "baseline file" row should not send the reader to "re-record from the
  reference commit", since the adoption in the footer (record this run) is
  the recovery for that failure too, whichever of a workload change or a
  hand-edited file caused it.

Checked paths: PASS table and merge; 12% slower with matching digests (fix,
or record the head and cite the delta from the failing run); a baseline whose
digests no longer describe its reference fails as "baseline file: re-record
from the reference commit" before any timing row; a noisy VM is the
counterexample to 10%, and re-running the job is the recovery when digests
match and the engine did not change. Open for later evidence, not this Task:
the 10% figure and whether a dedicated runner is worth buying.

## Next implement direction (third concept review)

1. `engine-budget.yml`: order the steps candidate build, candidate run,
   measurement (candidate block of the summary), reference build, reference
   run (its summary block), compare, upload. No new inputs or flags.
2. `docs/development.md` and the `AGENTS.md` bullet: the reasons to move the
   reference are a moved digest (a rules change reaching these decks, or a
   change to the benchmarks or curated lists), an accepted cost, or a kept
   speedup; a reference that no longer builds with the change's benchmarks
   is recorded from the PR's own run like any other move.
3. `scripts/engine_budget.py`: the "baseline file" row's failure text names
   the fact ("digests do not describe the reference") and leaves the
   recovery to the footer; keep its test.

## Status (reconciled 2026-11-02, compress after the candidate-ref change)

Exists: workflow, both benchmarks, comparator, tests, baseline file,
AGENTS.md note and `docs/development.md` section with the T4/T5/T6 table.
Reference 4c8324a59 recorded from CI run 41517623999; full CI validated by
workflow_dispatch run 41527511986 at 07512aa8c (Engine Budget A/B included).
Later commits (script simplification, one `other` checkout, the recording
design change, run-directory reading, the PR-head checkout, and this compress
pass: the checkout ref inlined, the measured commit local to the measurement
step, `record` taking the tolerance from the file only) have passed tests and
actionlint locally but not CI: the branch has no PR and branch pushes do not
trigger CI. Main moved to 29660d00c (docs/ARCHITECTURE.md and scratch only),
merged at 49765237c; the reference 4c8324a59 still describes the engine
(checked 2026-11-02).

Remains: the next implement direction above (step order, reasons wording,
one failure text), then publication, owned by the caller: open the PR with
the T4/T5/T6 figures in its body; the PR run confirms path selection on `pull_request`,
the PR-head checkout, the flagless compare step and that the artifact's
`measurement.json` names the PR head; the first main run after merge confirms
the push path; the `measure` path wants one dispatch before the next
historical measurement (gate/CI).

Proposed Wave memory, not written (no step so far maintains Wave memory):
engine speed now has a CI budget relative to a committed reference commit;
GitHub runner wall time is not comparable across runs; T4 main measured 6.8k
steps/s and 13.8 s worst Search-64 reply on EPYC 7763. Write it to
`.lf/wave/engine/MEMORY.md` from the step that does.

Check: `uv run --extra dev pytest tests/infra/test_engine_budget.py tests/infra/test_ci_selection.py -q` and `actionlint .github/workflows/engine-budget.yml .github/workflows/ci.yml` → 56 passed, actionlint clean, ruff clean (2026-11-02, compress).
