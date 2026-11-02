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

Replace the winner/simulation-only `scores_digest` fold in `bench_search`
with one ordered full-score digest. Preserve the workload, choice, timings,
integer output shape and simulation/choice evidence. No parallel legacy
digest or compatibility path. The Clean-machine Play 120 s wait stays a
launch proof, not a latency budget.

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

## Concept reviews (2026-11-02, headless)

Source: Jack Heart's Performance section ("the PR fails if worst reply or
total exceeds the recorded baseline by more than 10%; a PR that moves the
baseline records why; a headless steps/s workload alongside it; measure T4
and T5 once"; purpose "a CI budget keeps it gone"; order of work "each lands
with its bench_search delta in the PR body"). Each review found a mismatch
settled by the accepted decisions, with no open product choice:

- First (fixed in 45f1b4627): moving the reference needed a second `measure`
  dispatch and a seven-flag command, and keeping a speedup was not a
  documented move. Now every run is a recording of its candidate
  (`measurement.json`; `record --reason`; three documented reasons;
  `measure` confined to commits without a PR), so MTG-236's PRs record their
  head and the gain is budgeted.
- Second (fixed in 2f3fe99e1): on `pull_request` the default checkout is
  GitHub's ephemeral `refs/pull/N/merge`, which a later run cannot fetch. The
  job checks out `github.event.pull_request.head.sha || github.sha`, so the
  measured tree and the recorded commit are one durable commit. Detect
  Changes still diffs the merge commit; `measure` and push were never
  affected (the T4 recording and the validating dispatch passed).
- Third (fixed in 644eab3b0): the reference was built before the candidate,
  so a reference that no longer compiled with this ref's benchmarks (the
  `Env` surface change the docs name) left an empty artifact and that PR
  could not adopt its own run. Now the candidate is built, run and recorded
  first (its summary block), then the reference (its block), then compare
  and upload. Order on one VM is assumed not to bias the 10% tolerance
  (questions.md). Same commit: the reasons to move the reference name a
  moved digest (rules reaching the decks, or the benchmarks or curated
  lists), an accepted cost or a kept speedup; compare's "baseline file" row
  says "digests do not describe the reference" and leaves recovery to the
  footer.
- Fourth (reviewed at 078d37ef4; implementation in progress):
  the Task's score-preservation check currently protects only the chosen
  action and simulation count. `bench_search` folds
  `result.simulations + best` into `scores_digest`, never `result.scores`.
  With identical simulation counts, scores `[0.75, 0.25]` changing to
  `[0.875, 0.125]` leave the chosen action, game trajectory and digest
  unchanged. A faster but incorrect evaluation can therefore pass. The
  accepted requirement (Jack Heart's Task brief: "Score digests are asserted
  identical ... so a speedup that changes search results fails") settles
  this without a new product choice: preserve every action's score in order
  at every measured decision, together with decision boundaries and the
  existing simulation/choice evidence. Keep the workload, timings, 10%
  tolerance, same-runner comparison and artifact adoption interaction.

  Next implement direction: make the benchmark's digest cover the ordered
  score vector with a deterministic encoding, and prove with a controlled
  fixture that changing a losing score while preserving the winner and
  simulation count changes the digest. Also prove identical results repeat
  and changed digests fail even for a speedup. No engine/API redesign is
  needed: `FlatMcResult.scores` already owns the data; the comparator and
  baseline can keep their integer-digest shape. Describe this scope in the
  existing usage docs. Re-record the reference through a real CI artifact
  with a reason for the benchmark digest change; do not calculate replacement
  baseline digests by hand or reinterpret old measurements. The existing
  T4/T5/T6 timings remain dated evidence from the earlier digest, not proof
  of full-score equality. Label that limitation when retaining their table.
  Normal recovery is the existing download → `record --reason` → commit →
  rerun path. The next PR proof must show that the strengthened digests match
  the recorded reference and that the candidate artifact still survives a
  reference-build failure. Outcome: design clarified; implementation work
  remains; no unresolved product choice.

Checked paths: PASS table and merge; 12% slower with matching digests (fix,
or record the head and cite the delta from the failing run); a baseline whose
digests no longer describe its reference fails on the "baseline file" row
before any timing row; a noisy VM is the counterexample to 10%, and
re-running the job is the recovery when digests match and the engine did not
change. Open for later evidence, not this Task: the 10% figure and whether a
dedicated runner is worth buying.

## Status (reconciled 2026-11-02, fourth-review implement)

Exists: workflow, both benchmarks, comparator, tests, baseline file,
AGENTS.md note and `docs/development.md` section with the T4/T5/T6 table.
Reference 4c8324a59 recorded from CI run 41517623999; full CI validated by
workflow_dispatch run 41527511986 at 07512aa8c (Engine Budget A/B included).
Every commit since has passed tests and actionlint locally but not CI: the
branch has no PR and branch pushes do not trigger CI.
Main moved to 29660d00c (docs/ARCHITECTURE.md and scratch only), merged at
49765237c; the reference 4c8324a59 still describes the engine (checked
2026-11-02).

Implemented: length-framed little-endian score bits, simulation count and
chosen action feed BLAKE3, truncated to the existing u64 output. Controlled
example tests cover repeated results, a changed losing score with the same
winner and simulation count, action order, decision boundaries and retained
simulation/choice evidence. `test = true` includes them in ordinary CI
`cargo test`; benchmark-only edits select the Rust job too. No engine
source or workload changed. Historical timings now
state their original digest limitation.

Remains: the new CI recording, then
publication, owned by the caller: open the PR with the T4/T5/T6
figures in its body; the PR run confirms path selection on `pull_request`,
the PR-head checkout, the candidate-first order with its two summary blocks,
the flagless compare step and that the artifact's `measurement.json` names
the PR head; the first main run after merge confirms the push path; the
`measure` path wants one dispatch before the next historical measurement
(gate/CI).

Proposed Wave memory, not written (no step so far maintains Wave memory):
engine speed now has a CI budget relative to a committed reference commit;
GitHub runner wall time is not comparable across runs; T4 main measured 6.8k
steps/s and 13.8 s worst Search-64 reply on EPYC 7763. Write it to
`.lf/wave/engine/MEMORY.md` from the step that does.

Check: focused debug `cargo test --example bench_search -j 2` → 4 passed;
`uv run --extra dev pytest tests/infra/test_engine_budget.py tests/infra/test_ci_selection.py -q`
→ 57 passed; `cargo fmt --check` clean (2026-11-02). Full affected suites
and PR verification remain with gate/CI.
