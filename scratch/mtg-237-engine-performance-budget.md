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
- Fourth (reviewed at 078d37ef4; fixed in e5615edcc):
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

  Implemented with length-framed little-endian IEEE-754 score bits and
  separate simulation/choice words in BLAKE3 (first 64 bits retain the
  integer output shape). Controlled fixtures preserve repeatability and
  detect changed losing scores, action order, decision boundaries and
  simulation/choice changes. Existing comparator coverage rejects changed
  digests even during speedups. CI artifact 41540798745 records the new
  reference through `record --reason`; old T4/T5/T6 measurements retain
  their documented winner-only digest limitation. PR proof of matching
  strengthened digests and candidate recovery after reference-build failure
  remains with gate/CI. No unresolved product choice.

Checked paths: PASS table and merge; 12% slower with matching digests (fix,
or record the head and cite the delta from the failing run); a baseline whose
digests no longer describe its reference fails on the "baseline file" row
before any timing row; a noisy VM is the counterexample to 10%, and
re-running the job is the recovery when digests match and the engine did not
change. Open for later evidence, not this Task: the 10% figure and whether a
dedicated runner is worth buying.

## Status (reconciled 2026-11-02, fourth-review implement)

Exists: workflow, both benchmarks, comparator, tests, baseline file,
AGENTS.md note and `docs/development.md` section with the historical T4/T5/T6
figures and their digest limitation. Completed workflow work at 078d37ef4
is preserved. Main 29660d00c was merged at 49765237c.

Full-score digest implementation: e5615edcc. The focused example tests are
included in ordinary `cargo test`; benchmark-only edits select Rust too.
No engine source, workload, timing boundary or tolerance changed. The new
reference is e5615edcc7f6a7f1ef7adf83ce1a15f8302ac0b9, adopted using
`record --reason` from successful CI measurement run
https://github.com/loopflowstudio/manabot/actions/runs/41540798745.
Artifact: `.cache/engine-budget-score-digest`. The job recorded 120 decisions,
409.358 s total / 14.258 s worst, and 99,923 steps in 512 games at 6,829.1/s
on AMD EPYC 7763. Those timings are orientation only, not a cross-run delta.
The steps digest is unchanged. The three new full-score digests come from
that artifact, not calculations or reinterpretation of historical results.

Remains, owned by later steps: affected gate suites and publication with the
T4/T5/T6 figures and their limitation in the PR body. The PR run confirms
path selection, PR-head checkout, candidate-first recovery after a reference
build failure, the two summary blocks, flagless same-runner comparison with
the strengthened baseline, and artifact identity. The first main run after
merge confirms the push path. The measurement dispatch path passed in
41540798745. Interactive demo remains its declared boundary; shipping is
not authorized.

Proposed Wave memory, not written (no step so far maintains Wave memory):
engine speed now has a CI budget relative to a committed reference commit;
GitHub runner wall time is not comparable across runs; T4 main measured 6.8k
steps/s and 13.8 s worst Search-64 reply on EPYC 7763. Write it to
`.lf/wave/engine/MEMORY.md` from the step that does.

Check: focused debug `cargo test --example bench_search -j 2` → 4 passed;
`uv run --extra dev pytest tests/infra/test_engine_budget.py tests/infra/test_ci_selection.py -q`
→ 57 passed; `cargo fmt --check` and focused Clippy clean (2026-11-02). Full affected suites
and PR verification remain with gate/CI.
