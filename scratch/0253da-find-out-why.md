# 0253da — speed up manabot development (implementation plan)

Status: implemented 2026-11-05; awaiting gate and the PR's own CI run for the
hosted after-measurement. Evidence: `scratch/manabot-feedback-loops.md`
(manabot loops), `scratch/loopflow-speed-history.md` (loopflow changes).

## Diagnosis that selected the changes

- CI critical path is Integration `main` (22–27 min): one runner runs two
  serial pytest steps, 593 s (1052 tests) then 506 s (326 tests), after ~2.5 min
  install and extension build. Every engine PR selects it.
- Rust Tests runs binaries serially (`cargo test`): ~130 s of tests behind a
  ~2 min compile on main; on MTG-244 (#427) a 137× debug-only blowup in
  `semantic_conformance_tests` held the job 96–105 min and hid five card
  failures that ran in 18.7 s.
- opt-level 1 (3× faster debug tests) lived only in CI env vars; local agents
  ran opt-level 0 (an audit note records search_state_contract exceeding 5 min
  unoptimized).

## Changes (this PR)

1. `manabot-integration.yml`: groups `core` / `evaluation` / `long` on
   separate runners. Expected critical path ≈ 2.5 + 10 ≈ 13 min (was 22–27).
   Cost: one more ~2.5 min setup per run.
2. Rust Tests: `cargo nextest run --profile ci` (`managym/.config/nextest.toml`:
   no fail-fast, slow at 60 s, terminate at 5 min); JUnit timings uploaded.
3. `[profile.dev] opt-level = 1` in `managym/Cargo.toml`; CI env overrides
   removed (one authority). Debug assertions and overflow checks unchanged.
4. Selector: `managym/.config/nextest.toml` selects Rust only.

Consequence: a branch carrying MTG-244's debug-oracle cost now fails Rust Tests
within ~5 min naming the test, instead of running for 1.5 h. That cost belongs
to MTG-244; this PR does not touch it.

## Measurements

- Local (M-series, load avg 35–46, 4 test threads): nextest 969 passed in
  64.5 s wall; per-test sum 200 s; slowest 38.1 s
  (`engine_tests::random_smoke_milestone1_decklists`). Profile change
  reuses the env-var build exactly (no rebuild).
- CI before: run 41935280348 — Rust Tests 4.3 min, Integration main 21.9 min.
  After: read from this PR's run (Rust Tests job, three Integration groups).

## Follow-up Tasks (filed, pending Linear sync)

- 376d23 consolidate managym integration test binaries.
- 7d3ce5 shard the Integration `core` pytest group.
- 0b6170 diff-scoped verification passes and shared warm Rust builds across
  worktrees (MTG-241's verification wait and stale-measurement decision).

## Delete — do not maintain

- `CARGO_PROFILE_DEV_OPT_LEVEL` env in ci.yml (2), manabot-integration.yml,
  academy-integration.yml — removed.

## Checks

- `tests/infra/test_ci_selection.py` 54 passed; actionlint clean; `cargo fmt
  --check` clean; local nextest ci profile 969 passed. Gate owns the rest.
