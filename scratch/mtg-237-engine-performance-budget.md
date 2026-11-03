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

Completed in `79c623428`: removed implicit `bash -e` execution of benchmark
and comparison pipelines. Explicit job-level Bash owns failure propagation; the
same commands, summaries, artifacts and 10% baseline remain on that path.

## Decisions

- Accepted (Task brief, Jack Heart, 2026-11-02): both workloads budgeted at
  10% against a committed file; digests asserted identical; a 20% slowdown
  fixture; one-time T4/T5/T6 measurement; `uv` only; no induced load.
- Implied by the accepted outcome (concept review 2026-11-02); not yet
  confirmed by Jack Heart, whose PR review decides it:
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

## Review constraints

- CI measures the durable PR head, never GitHub's ephemeral merge commit.
  Candidate and reference are `Run` values; `compare` produces checks;
  `record` adopts the candidate measurement and reason as the next `Baseline`.
  CI owns execution and artifact identity; the file owns reference and tolerance.
- Build, run and record the candidate before building the reference. Upload
  in the final always-run step. An incompatible reference leaves an adoptable
  candidate; a failed candidate leaves diagnostics, not a complete measurement.
- Explicit Bash propagates benchmark and comparison failures through `tee`;
  Tests Result carries the failed job to the PR. A timing or digest failure
  stays visible until fixed or deliberately adopted, preserving tolerance and
  the source run. Reference digest drift gets a separate baseline-file row.
- Search hashes every ordered action score and decision boundary, simulation
  count and chosen action. Length-framed little-endian IEEE-754 bits feed
  BLAKE3; its first 64 bits preserve the integer output. Fixtures cover changed
  losing scores, action order, decision boundaries, simulations and choice.
- Historical T4/T5/T6 timings retain the winner-only digest limitation. Keep
  them in the development guide and PR body; new full-score recordings do
  not reinterpret them. Re-run unexplained timing failures when digests and
  engine match; changing tolerance or buying a dedicated runner needs later
  evidence, outside this Task.

Concept review found no additional product choice or code rewrite necessary.
The same-runner and native-workload interpretations above still await Jack
Heart's PR review. No shipping approval follows from headless review.

## Retained evidence

Implementation through `078d37ef4`, the full-score repair `e5615edcc` and sync
onto main `a8abf01b1` are preserved at `24cfc0a8b`. Explicit Bash and eight
workflow regression cases are at `79c623428`: passing/slow/changed-score
comparisons, four benchmark pipeline failures, and reference-build recovery.
An implicit-shell copy reproduces the original slowdown and benchmark failures.
The complete preceding plan is at
`88dc8c1ad:scratch/mtg-237-engine-performance-budget.md`; earlier concept-review
counterexamples are at `311218a02:scratch/mtg-237-engine-performance-budget.md`.
The [demo record](engine-budget-demo.md) owns run details and transcripts,
including the original false-green run 41548559113 and the repaired hosted proof.

The prior full-score reference `e5615edcc` came from dispatch 41540798745
(`.cache/engine-budget-score-digest`): 120 decisions, 409.358 s total,
14.258 s worst and 99,923 steps in 512 games at 6,829.1/s on EPYC 7763.
These are historical orientation figures, superseded by the adoption below.
Prior gate evidence (before the workflow repair): 855 debug Rust tests,
90 affected Python tests, actionlint, Rust formatting/Clippy and focused Ruff
passed. Gate owns broader affected suites; unchanged engine code needs no new
local performance measurement.

## Demo pass 2 — baseline reconciliation

The [pass-2 demo evidence](engine-budget-demo.md#decision-pass-2--2026-11-02)
establishes repaired hosted enforcement. Published head
`88dc8c1ad` ran in CI 41552162812: the comparison correctly failed with exit 1
under explicit Bash, and its always-run artifact uploaded and downloaded
successfully. Candidate throughput was 6,104.0/s against reference 6,774.5/s
(10.98% more time per step, exceeding 10%); both search timings and every
digest/count passed. Temporary adoption through the real CLI preserved the
measurement identity and tolerance. All 65 focused tests passed again,
including controlled reference-build recovery. Hosted reference-build failure
itself was not induced.

**Current demo decision: iterate only to reconcile the initial baseline with
already-merged main.** The reference predates `230b2f563` and `a8abf01b1`;
this PR has no native source diff from `a8abf01b1`. Under delegated headless
review authority, accept that synchronized engine as the initial budget's
reference while explicitly recording the measured cost; this is not new
human feedback or shipping approval. The detailed judgment and proof limits
are in the demo note.

Implemented: the real `record --reason` command adopted
`.cache/engine-budget-demo/pass-2/pr-head/` (run 41552162812, head
`88dc8c1ad25cd99cca6026581a569f5c2f63ee1b`) into the production baseline.
Its reason names the merged upstream changes, accepted 10.98% per-step cost,
run link and delegated judgment. The current-reference section in
`docs/development.md` records the same evidence without rewriting historical
T4/T5/T6 or full-score receipts. Tolerance remains 0.1; no engine, benchmark,
workflow or digest changed.

Check: direct artifact reconciliation verified every measurement identity and
benchmark field, unchanged tolerance and all digests/counts, and no native
source diff from merged main `a8abf01b1`; `git diff --check` passed. The retained
`uv run --extra dev pytest tests/infra/test_engine_budget.py tests/infra/test_ci_selection.py -q`
result (65 passed) still applies to unchanged code; no benchmarks were rerun.
`lf context` after compression: scratch and memory within token and byte limits.

Remaining: publication updates the PR's current-baseline paragraph to run
41552162812 and its measured accepted cost, and replaces the stale claim that
hosted enforcement is unverified with the pass-2 failure/upload/download proof.
Republish and inspect the new same-runner check at demo. Hosted reference-build
failure remains locally simulated only, and main-push verification stays
post-merge. No separate measurement dispatch or shipping approval.

Proposed Engine memory reconciliation (read-only during implementation):
replace the stale current-baseline and unverified-hosted-enforcement statements
with the pass-2 evidence and adoption above when the Graph reaches realign.
