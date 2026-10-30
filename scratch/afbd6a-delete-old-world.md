# afbd6a — delete old-world card machinery (stacked on #394)

Jack Heart (2026-10-29/30): old worlds run from their recorded git revision; delete
what only serves them. Diff should be almost all deletions. Status: implementing.

## Delete — do not maintain (done)

Python compiler/coverage, `content/semantic/{v1,w7–w11}`, their scripts and
compile-only tests, CI coverage job, conformance old-source digests,
`LEGACY_ORDER`/`TRAILING_ORDER`, old-world conformance corpora (archived) and
Academy fixtures.

## Keep / move

- Current-world checks in temur/sacrifice/elemental/optional-payments tests stay
  (they already run on setups); only compile assertions are removed.
- w17 conformance corpus regenerated (receipt shape loses `matchup_source_digest`).
- Phase matrix: rule-family closure against `v1/coverage.evidence.json` is dropped;
  remaining structural checks stay.
- Frozen corpora/evidence (w12–w15, experiments, audit, tla docs) untouched.

## Discovered while implementing

- `manabot/sim/heuristic.py` read v1/w11 sources: ported to
  `managym.card_definitions_json()` (new read-only binding); the 26 old targeted
  spells, instants and ability classes derive identically (old reader left modal
  bodies opaque; kept). Wider pool adds three spells.
  Identity: `heuristic.py` + `anchors.py`.
- 41 Academy fixtures for w7–w15 had no consumer: deleted.
- `test_learning_regression` fed a default-pack observation to an authored-pack
  policy (passed by CardDefId coincidence): uses an authored case now.
- Five tokens await producers (`TOKENS_AWAITING_PRODUCER` in observability.rs).

## Added scope — Jack Heart steers, 2026-10-30 (implemented)

1. Tokens: all 11 in `cardsets/tokens.rs` via `token(...)`/`creature(...)` helpers.
2. Statics: one `StaticAbility` enum in `CardDefinition.statics`; `sacrifice_cost`/
   `kicker` are `AdditionalCost` variants; accessors (`ward()`, `power_cda()`,
   `kicker()`) read variants. Learning schema unchanged (serializer-derived tokens;
   catalog bytes change only through the new field/variant names).
3. Observability: no-wildcard `static_seen`/`cost_seen`, per-variant token + removal
   tests, tensor tests (every witness row naming a column, hand size, unblockable), real-card
   behavior-test index, token-creator test; `tests/semantic/test_static_ability_tokens.py`.
4. Corpora archived (manifest entries; incl. learn-lesson-w4-v3 + its verify script,
   which only checked the frozen w4 cohort); README lists the four w17 corpora.
   Digest shrink (for PR): not worthwhile. w17 is 3.1 MB; replays already store
   only commands + per-command digest checkpoints (~500 B/command). Dropping
   per-command checkpoints saves ~1.5 MB but loses first-divergence localization.

## Check results

2026-10-30: debug `cargo test` (all suites) and clippy `-D warnings` pass; CLAUDE.md
post-world suite + `tests/semantic` (183) and infra/static-token tests pass. Earlier:
sim/belief/bench-study failures equal base (2c50c8793); `tests/eval` full-suite hang
is pre-existing and order-dependent; training suite left to CI/gate.

Compress 2026-10-30: column test iterates the variant witnesses instead of a duplicate
mutation table; `cargo test --test cards observability`, clippy, test_heuristic pass.

Sync onto T2 (c7979d310), 2026-10-30: T2's new definition fields became
`StaticAbility`/`AdditionalCost` variants (see `questions.md`); w17 corpora,
parity pins and Academy fixtures re-recorded. Debug `cargo test` for the cards,
conformance and semantic-conformance suites passes, as do the post-world suite,
the heuristic and the Academy fixture consumers.
