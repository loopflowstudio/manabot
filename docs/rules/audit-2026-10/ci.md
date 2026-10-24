# CI latency and runner cost — MTG-177 draft

The measurements are read-only observations of two already completed successful
[CI runs](https://github.com/loopflowstudio/manabot/actions/workflows/ci.yml):
[39714587357](https://github.com/loopflowstudio/manabot/actions/runs/39714587357)
and [39711865197](https://github.com/loopflowstudio/manabot/actions/runs/39711865197).
They are nearby revisions, not controlled A/B measurements of this draft.
[ci-evidence.json](ci-evidence.json) retains job/step durations and test summaries.

| Measure | 39714587357 | 39711865197 |
|---|---:|---:|
| Observed first-job-start to last-job-end | 558 s (9.30 min) | 632 s (10.53 min) |
| Sum of all job lifetimes | 45.62 runner-min | 35.28 runner-min |
| Rust Tests job / test step | 175 / 165 s | 161 / 152 s |
| Python integration main job / test step | 418 / 348 s | 428 / 352 s |
| Python integration long job / test step | 308 / 238 s | 308 / 232 s |
| Protocol job / TypeScript step | 547 / 432 s | 189 / 80 s |
| Academy UI job / dependency+Chromium step | 473 / 347 s | 621 / 459 s |
| Clean-machine job / Chromium provision | 504 / 423 s | 101 / 24 s |

These sums include setup/post time; they are not rounded billable invoices or
money costs. Job-span wall time excludes queueing before the first job. The
slowest job varies with installation/provisioning, and a sample of two is not a
flakiness or performance distribution. No intentional repeated-run experiment
was performed.

## Native test sharding

The sampled Rust test output reports roughly 78 seconds of summed binary
execution, versus a 165-second test step including compilation/orchestration.
Largest binary: `search_branch_tests` (12 tests, 42.42 s). `rules_tests` takes
14.21 s (297 tests); engine smoke takes 10.83 s; semantic conformance takes
4.91 s. [The retained summaries](ci-evidence.json) identify first/last tests
rather than guessing binary labels from rule names. CI already uses dev
optimization level 1 while retaining debug assertions and overflow checks.

Splitting the rules binary itself is a low-priority optimization: it is not the
largest suite and compilation/setup dominate much of the job. Parallel binary
execution after one build could reduce native wall time while keeping one
runner, but risks contention and requires evidence of CPU/RSS capacity. It does
not solve the Python/browser critical path in either sample.

A shared-build-then-two-runner design must distribute exact debug artifacts,
keep platform/toolchain/features/profiles identical and prove every test binary
is assigned exactly once. An illustrative optimistic bound is 78 -> 42 seconds
of native execution when isolated by binary, before transfer and setup. That
saves at most about 36 seconds here, while additional runner setup/transfer
**increases** total runner time. It is not a measured saving and does not justify
a matrix rewrite in this draft. Never replace debug coverage with release tests.

## Python groups, builds and caching

The reusable integration workflow already shards `main` and `long`. Main's
982 passes/5 skips took 339.65 seconds in pytest, long's 27 passes took 231.97
seconds (job-step timing includes wrapper overhead). Each group independently
installs dependencies and builds an extension; those two builds take 26+23
seconds in the first sample. Academy integration adds 26 seconds; protocol and
frontend release builds add 46 and 31 seconds. These five visible extension
builds sum to 152 seconds, but are **not** all interchangeable artifacts:
profiles, extras, features, package installation and consumers differ.

A shared Python wheel for compatible consumers is a plausible next experiment.
It requires one Linux cp312 build, pinned features and profile, explicit artifact
identity, offline installation proof and a fail-closed dependency on the build.
Even a successful experiment must count build/upload/download/setup across all
runners; a build dependency may lengthen the critical path. Do not share the
release wheel with debug-assertion obligations or with the deliberately fresh
clean-machine proof.

Rust artifacts already use `Swatinem/rust-cache`; integration also caches uv
downloads keyed by dependency inputs, and selected Node jobs use lockfile-based
npm cache. Review actual hit/miss and download bytes before adding caches or
large cross-job artifacts. The clean-machine proof deliberately disables uv
caching and owns a clean-launch contract; optimizing it into a warm-machine
check would change coverage. Large Chromium provisioning swings are worth a
separate controlled infrastructure investigation, not a reason to relax tests.

To choose further Python groups, first collect per-test durations: **the draft
adds `--durations=20` to both existing integration groups**. It changes neither
selection nor assertions and performs no additional tests. A two-runner split
of the 340-second main suite could ideally save up to 170 wall seconds only
under balanced independent tests; duplicated ~70-second setup would add about
1.2 runner-minutes, plus any collection/transfer/imbalance cost. This arithmetic
is a planning bound, not a prediction. Tests may share mutable resources and
must be reviewed before using xdist or more jobs.

## Tiers and required aggregation

Keep directed rules, admission/contracts, protocol checks and deterministic
bounded conformance on PRs. The current conformance job executes 32 seeded
valid-action cases, at most 512 commands per case; this audit does not increase
it. Expensive expanded cohorts and future adversarial exploration belong in an
explicit scheduled/manual tier after a declared budget, with failures retained.
Do not move current required coverage out of PRs merely to get a faster number.
macOS compatibility already has a nightly workflow.

`Tests Result` runs with `always()` and requires successful change detection.
It requires success for each selected job and allows skipped/success only for
unselected jobs; missing outputs and unexpected skips fail closed. PR revision
cancellation replaces obsolete PR runs while main/manual runs have separate
concurrency groups. These semantics stay intact; no branch-protection setting
was read as authority to edit or changed.

The only draft selection change routes `scripts/audit_rules.py` and this pinned
audit directory to the existing Python unit job. Its offline inventory tests
have no Wizards/network dependency. Existing Rust-source changes still select
all dependent consumers; no existing coverage is removed. Selection and
aggregate tests pass locally. A report-only update in this directory now costs
a Python unit job (sample whole job 36–48 seconds, plus detect/result), an
intentional integrity tradeoff.

## Incremental cost and decision

The new three enabled Rust tests execute 1,356 bounded cases in about 0.02
seconds locally. The new Python integrity tests execute in about 0.07 seconds;
combined with selector/aggregation checks the selected run takes 0.74 seconds.
Neither is a controlled before/after CI delta; scheduling resolution and cold
compilation dwarf those local timings. Local cold builds, the known-failure
probe, provisioning failure and repair are all charged in
[validation.json](validation.json). No additional cloud run was initiated.

**Draft decision:** preserve the current topology, collect duration evidence,
and use tiny directed tests. Consider shared compatible wheels before adding
Python shards; consider one-build native scheduling only after confirming
resource headroom. There is no evidence here to promise zero overhead, lower
billed cost, or a new required-check layout.
