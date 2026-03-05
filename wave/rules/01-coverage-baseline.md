# 01: Coverage Baseline + Rule-Cited Test Harness

## Context

- User: "implemented has stages. we should identify written code and strive for tests for it all"
- User: "i would love for comments to explicitly mention which rules code is implementing and likewise for tests"
- Scope for this wave: two-player only; defer multiplayer.

## Finish line

We can answer “what rules are implemented and tested” from repo artifacts, and
current engine behavior has CR-cited focused Rust tests (including negative
paths where reasonable).

## Changes

### 1. Coverage artifact + status ladder

Add `docs/rules_coverage.yaml` with parent-rule rollups allowed.

Suggested statuses:
- `not_started`
- `scaffolded`
- `implemented`
- `implemented_tested`
- `parity_validated`
- `documented_deviation`

Each entry includes `rule`, `status`, `code_refs`, `test_refs`, `notes`.

### 2. Citation convention

- Code comments cite CR rules directly (`// CR 305.1, 305.2`).
- Rule tests cite CR in name and comments (`cr_305_1_*`).

### 3. Rust-first rule test structure

Create `managym/tests/rules/` with chapter subfolders, fixtures, and helpers for
scenario setup and action selection.

### 4. Backfill current behavior

Map and test currently implemented behavior (priority pass loop, land timing,
combat skeleton, lethal/life/library SBA subset, cleanup damage clearing,
invalid action index handling).

## Done when

- `docs/rules_coverage.yaml` exists and maps currently implemented families.
- New rule tests run green under `cargo test`.
- Existing implemented paths have CR citations in code/tests.
- Verification command: `cargo test -p managym`.
