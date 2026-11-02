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

Completed in e5615edcc: the winner/simulation-only `scores_digest` fold in
`bench_search` is gone. One ordered full-score digest preserves the workload,
choice, timings, integer output shape and simulation/choice evidence. No
legacy path or remaining deletion target. The Clean-machine Play 120 s wait
stays a launch proof, not a latency budget.

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

## Review constraints (2026-11-02, headless)

Jack Heart's Performance section asks for a 10% budget, deliberate baseline
moves, a headless steps/s workload, one-time T4/T5 measurements and each PR's
bench_search delta. The completed reviews enforce these constraints:

- Every job records its candidate with `measurement.json`; a PR adopts its
  own artifact using `record --reason`. Record moved digests, accepted costs
  or kept speedups. Reserve `measure` dispatch for commits without a PR.
- Measure the durable PR head, never GitHub's ephemeral merge commit.
- Build, run and record the candidate before building the reference so an
  incompatible reference still leaves an adoptable artifact. Compare's
  baseline-file row distinguishes reference digest drift from candidate cost.
- Preserve every ordered action score and decision boundary, plus simulation
  count and chosen action. The old fold missed changed losing scores when
  the winner and simulation count stayed fixed, allowing incorrect speedups.
  The Task's correctness requirement settles this without a new product choice.

The full review history and counterexamples remain at
`311218a02:scratch/mtg-237-engine-performance-budget.md` (workflow fixes through
078d37ef4, full-score fix e5615edcc). The surviving digest uses length-framed
little-endian IEEE-754 score bits in BLAKE3, keeping its first 64 bits as the
integer output. Controlled fixtures check repeatability, losing-score changes,
action order, decision boundaries and simulation/choice changes. Comparator
fixtures reject changed digests even during speedups. Historical T4/T5/T6
measurements retain their winner-only digest limitation; the new recording and
remaining proof belong to Status below.

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

Reply timings are sorted in place after summing: their original order is no
longer used, so a cloned vector adds no information. Digest bytes, measured
work and the baseline remain unchanged.

Check (2026-11-02): `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml --example bench_search -j 2`
→ 4 passed; `cargo fmt --manifest-path managym/Cargo.toml --check` and
`git diff --check` clean. Reused implement's 57 comparator/CI-selection passes
and focused Clippy result; full affected suites and PR verification belong
to gate/CI. `lf context` confirms memory and scratch remain under both limits.
