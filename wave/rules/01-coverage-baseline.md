# 01: Coverage Baseline + Rule-Cited Test Harness

## Finish line

We can answer “what rules are implemented and tested” directly from repo artifacts.
Current implemented behavior has CR-cited Rust tests with negative paths.

## Changes

### 1) Coverage artifact

Add `docs/rules_coverage.yaml` with entries:
- `rule`
- `status` (`not_started` | `scaffolded` | `implemented` | `implemented_tested`)
- `code_refs`
- `test_refs`
- `notes`

### 2) CR citation convention

- Implementation comments cite rule references at enforcement points (e.g. `// CR 305.1`).
- Rule tests use CR-cited names/comments (e.g. `cr_305_1_*`).
- Snapshot anchor for this stage: `docs/rules/MagicCompRules-20260227.pdf`.

### 3) Rust-first rule harness + backfill

Create `managym/tests/rules/`:
- `helpers.rs` with a thin `ScenarioBuilder` wrapper on public `Game` APIs
- chapter-focused files (`cr_103_*`, `cr_106_*`, `cr_117_*`, `cr_305_*`, `cr_508_*`, `cr_509_*`, `cr_510_*`, `cr_514_*`, `cr_601_*`, `cr_704_*`)
- happy-path + negative-path coverage for each implemented family

Target: ~30-40 tests.

### 4) Explicit defers

Defer from stage 01:
- JSON trace ingestion format
- `parity_validated` / `documented_deviation` statuses
- Training baseline metrics

## Maintenance contract

1. `tests/rules/helpers.rs` stays thin (no duplicated game logic).
2. Rule-test PRs update `docs/rules_coverage.yaml` in the same diff.
3. CR citations in stage-01 artifacts reference the pinned rules snapshot.

## Done when

- `docs/rules_coverage.yaml` exists and maps current rule families.
- `managym/tests/rules/` exists with CR-cited focused tests (including negatives).
- Existing implemented paths are CR-cited in code.
- `cd managym && cargo test` passes.
