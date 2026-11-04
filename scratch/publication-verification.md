# Publication verification — 5da5f4

The accepted publication scope is the explicit v7 shared-effect repair. Main
`3b122de9078c985f846a0f3c5a3a42ebca3b1182` is the actual base and was integrated
by `lf sync`. Fresh defaults, frozen v1–v6 inputs and scientific protocols remain
unchanged. This step publishes for review; it does not merge or complete the Task.

## Checks and repairs

- Complete debug Rust workspace: 909 passed, 2 ignored, no failures, with
  `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2`.
- Broad Python selection: 454 passed, 1 skipped, 2 stale text-fixture failures.
  This includes all model tests, transport/parity, active training, experiment
  discovery, package boundaries, artifact tracking, and the required current-world
  arena/admission/monitoring selection. The text failures expected the old field
  order and injected legacy effects without supplying the new shared authority.
  Updated fixtures retain exact assertions and cover both current and historical
  captures. Full text/CI-selection/release recheck: 107 passed.
- Added shared-effect transport, source-duration parity and study-summary tests
  to the existing integration CI command; they previously were not selected.
  That exact CI invocation collects 953 tests without collisions. An earlier
  partial collection lacked the llm extra; installing the declared extras fixed it.
- Clippy all targets/features passes with warnings denied. Initial interpreter
  lookup required the absolute pinned interpreter path. Replaced constant-sized
  chunk iteration with typed array chunks as required by Clippy, and formatted
  the two native test files missed by earlier focused checks.
- Rebuilt the native release wheel from the root and atomically replaced the
  cp312 extension. Post-rebuild transport/model checks: 23 passed, including
  native scalar/vector/Python parity, retained 140/270 cases, gradients,
  checkpoint reload and ordinary learner/export.
- Ruff passes for the new modules and changed text tests; `git diff --check`
  passes. All four focused debug native shared-effect tests also pass after the
  Clippy-only edit.

Logs live under `.runs/representation/publish-*`. Existing published engineering
receipts retain their original source hashes; these local checks do not rewrite
them. No new paid compute was launched.

## Review limits

Hosted CI, including same-runner Engine Budget, remains pending until publication.
Publication does not establish performance-budget acceptance or authorize a merge
with failing checks. No v7 CUDA run, replicated learning comparison, strength
claim, universal capacity guarantee, memory implementation or semantic timestamp
coalescing is included. The pre-existing keyword-order defect remains documented
in `scratch/questions.md`. V7 has smaller active effect payloads for the retained
cases but larger padded buffers; do not claim a memory or throughput win.
