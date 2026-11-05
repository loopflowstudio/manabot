# 0253da — speed up manabot development (implementation plan)

Status: implemented 2026-11-05; reconciled 2026-11-05; awaiting gate and the PR's own CI run for the
hosted after-measurement. Evidence: `scratch/manabot-feedback-loops.md`
(manabot loops), `scratch/loopflow-speed-history.md` (loopflow changes).

## Diagnosis that selected the changes

- MTG-241 (two Graphs, ~9.2 h, no PR): implement 277 min over 8 passes, growing
  3–6× per Graph as the sibling was integrated; compress 106; realign 125;
  sync 37. Both Graphs ended on tooling/sync failures (≥101 min). Not yet split
  within implement passes into agent work, test wait and rework.

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

## Follow-up Tasks (filed, unstarted)

- 376d23 consolidate managym integration test binaries.
- 7d3ce5 shard the Integration `core` pytest group.
- 0b6170 diff-scoped verification passes and shared warm Rust builds across
  worktrees (MTG-241's verification wait and stale-measurement decision).

## Delete — do not maintain

- `CARGO_PROFILE_DEV_OPT_LEVEL` env in ci.yml (2), manabot-integration.yml,
  academy-integration.yml — removed.

## Remaining

- Hosted after-measurement (Rust Tests, three Integration groups) from this
  PR's CI run; publication and gate own it.
- MTG-241 within-pass split (agent vs test wait vs rework) from conversation
  logs, if still wanted; the step-level breakdown above answers the coarse question.

## Checks

- `tests/infra/test_ci_selection.py` 54 passed; actionlint clean (re-run after compress step reorder); `cargo fmt
  --check` clean; local nextest ci profile 969 passed. Gate owns the rest.
- Realign 2026-11-05: diff matches plan; only required check is `Tests Result`, so renamed Integration groups need no ruleset change; no doctests (nextest skips them).

## Concept review 2026-11-05 (Graph, no person)

Experience: an engine change gets a green or red verdict in ~13 min instead of
~25, and a pathological test is named within 5 min instead of hiding others for
hours. The diff already gives that experience in a few concepts: one dev profile (opt-level 1,
assertions on) shared by local, CI and `maturin develop`; one nextest `ci`
profile (no fail-fast, 5-min kill, JUnit); three Integration groups balanced by
measured time. No new concept is needed to understand them, and the one-authority profile retires the
env-var override. Stale-name sweep: only frozen evidence
(`docs/rules/audit-2026-10/*.json`) names the old `main` group or env var, and it
stays unchanged.

Considered, not changed: renaming the `long` group (now not the longest) is
cosmetic and churns the job name; making the 5-min kill the default local
profile would turn contention on shared hosts (load 35–46) into false failures.
`ci` stays opt-in locally. nextest's default profile already reports tests slower than 60 s.

Outcome: nothing to change; no open product choice.
