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

Demo iteration removes the implicit `bash -e` execution of benchmark and
comparison pipelines. Explicit job-level Bash owns failure propagation; the
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

Concept-review outcome (2026-11-02, after full-score and demo enforcement
repairs at `79c623428`): nothing to change and no open product choice
requiring another implement pass. The
developer's interaction is one PR check and, for a deliberate change, one
artifact adoption with a reason. Its concepts map directly to the existing
APIs: the candidate and reference are `Run` values, `compare` produces the
budget checks, and `record` persists the candidate's measurement and reason
as the next `Baseline`. CI owns execution and artifact identity; the committed
file owns the reference and tolerance. No separate measurement lifecycle or
second baseline command is needed.

The normal path preserves scores and passes within budget. A timing or digest
failure remains visible until fixed or deliberately adopted; recording keeps
the tolerance and leaves the original run as evidence. A reference build
failure retains the candidate recording for the same recovery, with the
controlled proof and hosted evidence distinguished below. The two
counterexamples already driving the design—different runner speeds and changed losing scores with the same
winner—are addressed without changing the accepted workload or hiding failure.
The reference/workload interpretations remain explicit for Jack Heart's PR
review, not newly approved here. Next proof remains the existing gate and
PR-path checks; this review requests no code rewrite or execution restart.

The demo's counterexample adds one invariant to that same interaction: a
failed benchmark or comparison must fail the PR check even when its output
is retained through `tee`. Explicit Bash now owns that propagation for all
run steps; the existing Tests Result aggregate carries the failed job to the
PR. No additional status, command or approval concept is needed. A successful
candidate remains recoverable when only the reference build fails: its
measurement is recorded first, then the final always-run upload retains it.
A failed candidate benchmark instead leaves diagnostic output, not a complete
adoptable measurement. These are consequences of the accepted failure and
recovery contract, not new decisions attributed to Jack Heart. The local
fixtures support this model; they do not replace the pending repaired hosted
run or prove GitHub artifact transport after failure. Usage guidance already
describes the same adoption command; only the retained PR wording correction
is needed at publication. No skill instruction changes follow from this review.

## Current evidence and remaining work — 2026-11-02

Implementation through `078d37ef4`, full-score digest repair `e5615edcc`,
compression and sync are preserved at `24cfc0a8b`, which includes main
`a8abf01b1`. The demo repair and regression coverage are committed at
`79c623428`. Earlier realign/publication details and validation commands are
retained at `24cfc0a8b:scratch/mtg-237-engine-performance-budget.md`.
Reconciliation includes the upstream T5 decisions in Engine memory and the
decklist contract in `AGENTS.md`; neither changes this repair's workload,
digest or budget. Engine memory
now preserves the pipeline-failure lesson and the distinction between local
recovery and hosted artifact transport. No immediate child Wave directories
or child memories exist under `.lf/wave/engine/` in this checkout.

The baseline names `e5615edcc7f6a7f1ef7adf83ce1a15f8302ac0b9`, adopted from
successful dispatch [41540798745](https://github.com/loopflowstudio/manabot/actions/runs/41540798745).
Its downloaded artifact (`.cache/engine-budget-score-digest`) matches every
baseline value and measurement field: 120 decisions, 409.358 s total /
14.258 s worst, 99,923 steps in 512 games at 6,829.1/s on EPYC 7763.
These figures are orientation only. Historical T4/T5/T6 numbers and their
winner-only digest limitation remain in `docs/development.md` and must stay
in the PR body; the new full-score digests do not reinterpret old evidence.

Prior gate evidence: 855 debug Rust tests (including four digest fixtures),
90 affected Python tests, actionlint, Rust formatting/Clippy and focused Ruff
passed. These results predate the demo repair. No engine, benchmark, digest,
tolerance or production baseline changes in this iteration require new
performance measurements locally.

**Demo judgment: iterate.** The [demo record](engine-budget-demo.md) owns the
observed PR runs, measurements, artifact paths and replay transcripts. Run
41548559113 selected and recorded durable head `24cfc0a8b`, wrote both summaries
and uploaded the correct artifact, but reported success while its comparator
failed the throughput limit. Implicit `bash -e` let `tee` hide exit 1; replay
with pipefail exits 1. The earlier passing run and CLI tests do not rebut this
workflow defect. This is delegated headless judgment, not new human approval.

The committed repair sets job-level `defaults.run.shell: bash`, giving every
run step Actions' pipefail behavior. Workflow edits select Python regressions
as well as Engine Budget. Eight cases execute the actual run bodies: pass,
20% slowdown, faster-but-changed digest, four benchmark pipeline failures,
and reference-build recovery. The recovery fixture runs candidate binaries,
records through the actual measurement shell, injects reference Cargo exit 42,
copies the always-upload step's declared path and adopts it with the real
`record --reason` CLI. Artifact contents prove recording happened before the
failure; a separate execution ledger adds no evidence. Removing explicit Bash
from a separate workflow copy made the slowdown and four benchmark cases
reproduce the original defect.

The recovery proof is local: copying represents upload/download, not GitHub
transport. Gate owns affected suites. Publication/demo owns the repaired PR
run and artifact inspection, and correcting the PR wording to **recorded before
reference build, uploaded by the final always-run step**. Assess any remaining
timing failure after enforcement works; do not widen the budget or adopt the
marginal measurement to hide this bug. First main-push verification remains
post-merge. The same-runner reference and native-throughput interpretations
still await Jack Heart's PR review. Shipping is not authorized.

Check: retained `uv run --extra dev pytest tests/infra/test_engine_budget.py tests/infra/test_ci_selection.py -q` → 65 passed, with focused Ruff lint/format passing; documentation-only reconciliation: `git diff --check` passes and `lf context` is within both limits. Gate owns broader affected suites; publication/demo owns hosted proof.
