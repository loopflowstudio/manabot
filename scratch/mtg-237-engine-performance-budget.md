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
- `scripts/engine_budget.py compare|record` (stdlib only). Every job run
  uploads one `engine-budget` artifact: both outputs plus `measurement.json`
  (candidate commit, ISO date, run URL, runner CPU); `record --reason`
  rewrites the baseline from a downloaded artifact directory (default
  `.cache/engine-budget`). Tests in `tests/infra/test_engine_budget.py`
  include the 20% slowdown fixture and recording from an artifact.
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
- Implied by the accepted outcome (concept review 2026-11-02), confirmed in
  Jack Heart's PR review: "committed baseline" is read as a
  committed *reference commit* rebuilt and measured A/B on the same runner,
  not committed absolute numbers. Evidence: one engine spread 6.8k-10.3k
  steps/s across three runner CPU models (far outside 10%); two runs on one
  model agreed within 2.5%. The file's figures are orientation only.
- Implied by the accepted purpose (same review), confirmed in the PR review:
  the headless workload is native `Env::step`
  random-legal play (512 games, skip_trivial), not the rul-9 Python
  semantic-Command runner (deleted in #343). Its steps/s is its own
  baseline, not comparable with rul-9's 588.5.
- Any rules change reaching the curated decks moves the random-play digest
  and costs one recording per such PR; accepted as the price of (3). A kept
  speedup is the third reason to move the reference (concept review).

## Concept review (2026-11-02, headless, against the accepted design)

Source: Jack Heart's Performance section of 2026-11-02 ("the PR fails if
worst reply or total exceeds the recorded baseline by more than 10%; a PR
that moves the baseline records why; a headless steps/s workload alongside
it; measure T4 and T5 once"), its purpose line "a CI budget keeps it gone",
and its order of work ("derived-state revisions: each lands with its
bench_search delta in the PR body").

**What a developer does, and the concepts it takes.**

| Action | Today | Concepts touched |
|---|---|---|
| Push an engine PR | Engine Budget rebuilds the reference commit beside the change, prints one PASS/FAIL table (digests, then worst/total/steps) | reference commit, tolerance, digests |
| Accept a cost, or a rules change that moved a digest | dispatch `ci.yml` with `measure=<head>`, wait for a second ~20 min job, download its artifact, run `record` with seven flags, commit | measure, record, run URL, runner CPU |
| Keep a speedup | undocumented; the reference stays at T4, so later slowdowns are allowed back up to T4 | — |
| Measure a historical commit | `measure=<commit>` dispatch (T4/T5/T6 done) | measure |
| Change the tolerance | edit `tolerance` in the file | tolerance |

**Settled by the accepted decisions (recorded here, not re-opened).**

- "Committed baseline" is a committed *reference commit* rebuilt and
  measured A/B on the PR's runner. The accepted outcome is a 10% budget on
  GitHub-hosted runners; the four dispatches showed one engine at 6.8k and
  10.3k steps/s on two runner CPUs, so committed absolute numbers cannot
  deliver that outcome, and the order of work wants "its bench_search delta
  in the PR body", which only a same-runner comparison can state. The file
  still is a committed artifact a PR moves deliberately with a reason, as
  the design asks. Not a product choice; Jack Heart's PR review is where it
  is confirmed (brief: "One PR, Jack Heart's review").
- The headless workload is native `Env::step` random-legal play. The
  design's purpose is training throughput; the training environment steps
  the native `Env`, and the rul-9 Python runner was deleted in #343. Its
  figure is its own baseline, not rul-9's 588.5.
- Digests are the workload's identity on both benchmarks, so a rules
  change reaching the curated decks moves the reference (one recording per
  such PR); the compare table shows that PR's timing rows anyway, so its
  speed cost is visible in the same run.

**The consequential mismatch.** Every compare run already measures the
candidate on the runner with exactly the output `record` consumes, yet
moving the reference requires a second `measure` dispatch and a hand-typed
command, and the one case the design states as the budget's purpose
(keeping a speedup: "a CI budget keeps it gone") is not a documented move at
all. Two concepts (measure, record) stand where the contract needs one:
*every Engine Budget run is a recording of its candidate; a PR adopts one by
committing it with a reason.*

**Implement direction (design changed in the concept review; implemented
2026-11-02, see Status).**

- The job writes `measurement.json` beside the two `.jsonl` files in both
  modes (candidate commit, ISO date, run URL, runner description from the
  step summary's CPU line). `record` reads a downloaded artifact directory
  (default `.cache/engine-budget`) and takes only `--reason`; the
  `--commit/--date/--run/--runner/--search/--steps` flags go. Compare's FAIL
  line and `docs/development.md` name the two-command path:
  `gh run download <run-id> --name engine-budget --dir .cache/engine-budget`
  then `uv run scripts/engine_budget.py record --reason "<why>"`.
- "Moving the reference" lists three reasons with one procedure, from the
  PR's own compare run: a rules change reaching the decks, an accepted cost,
  a speedup to keep (MTG-236's PRs record their head so the gain is
  budgeted). The recording commit's CI run compares the same engine with that
  reference and passes; the earlier run keeps the delta the PR body cites.
- `measure=<commit>` stays only for measuring a commit that has no PR (the
  historical T4/T5/T6 use); the docs say so and stop presenting it as the
  re-record path.
- Tests: `record` from an artifact directory; the committed-baseline test
  unchanged. The 20% fixture, digest tests and workflow are unchanged.

**Checked against a normal path, a failure and recovery.** Normal: engine
PR, PASS table, merge. Failure: 12% slower with matching digests; the
developer fixes it, or records the head with the reason and the PR body
cites the delta from the failing run. Recovery: a baseline whose digests no
longer describe its reference fails as "baseline file: re-record from the
reference commit" before any timing row, and the fix is one recording. A
noisy VM is the counterexample to a 10% budget; two EPYC 7763 runs agreed
within 2.5%, and re-running the job is the recovery when digests match and
the engine did not change. The `Env` surface edge (new benchmarks must
compile against the reference, or the PR moves the reference) is documented
and accepted.

**Unresolved product choices.** None that block. The tolerance (10%) and
whether a dedicated runner is ever worth buying are open questions for
later evidence, not decisions this Task needs.

## Status (reconciled 2026-11-02, third pass)

Exists: workflow, both benchmarks, comparator, tests, baseline file,
AGENTS.md note and `docs/development.md` section with the T4/T5/T6 table.
Reference 4c8324a59 recorded from CI run 41517623999. Full CI validated by
workflow_dispatch run 41527511986 at 07512aa8c (Engine Budget A/B included).
Since then: the script simplification b734238c4; the compress pass that gave
the workflow one `other` checkout and dropped `record`'s unused
`--tolerance`; the second reconciliation that completed the documented `Env`
surface; and the implement pass of 2026-11-02 that carried out the concept
review's direction: the job writes `measurement.json` beside the two
`.jsonl` files in both modes and uploads one `engine-budget` artifact (the
`-<commit>` suffix in measure mode is gone), `record` takes `--artifact`
(default `.cache/engine-budget`) and `--reason` only, compare's FAIL line
names the two-command path, and the docs list the three reasons to move the
reference and confine `measure` to commits without a PR. Main has not moved
past the base 4c8324a59 (checked 2026-11-02). Tests and actionlint pass
locally; CI has not run on this head because the branch has no PR and branch
pushes do not trigger CI. The PR run covers the compare path and writes the
first real `measurement.json`; the `measure` path is unexercised since the
rewrite and wants one dispatch before the next historical measurement
(gate/CI).

Remains (publication, owned by the caller): open the PR with the T4/T5/T6
figures in its body; the PR run confirms path selection on the
`pull_request` event and the artifact's `measurement.json`, and the first
main run after merge confirms the push path.

Proposed Wave memory, not written: engine speed now has a CI budget relative
to a committed reference commit; GitHub runner wall time is not comparable
across runs; T4 main measured 6.8k steps/s and 13.8 s worst Search-64 reply
on EPYC 7763. Left in scratch because no step so far maintains Wave memory and the A/B
reading is confirmed in Jack Heart's PR review; write it to
`.lf/wave/engine/MEMORY.md` from the step that does.

Check: `uv run --extra dev pytest tests/infra/test_engine_budget.py tests/infra/test_ci_selection.py -q` and `actionlint .github/workflows/engine-budget.yml .github/workflows/ci.yml` → 56 passed, actionlint clean (2026-11-02, implement pass).
