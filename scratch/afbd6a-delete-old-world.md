# afbd6a — delete old-world card machinery (stacked on #394)

Jack Heart (2026-10-29/30): old worlds run from their recorded git revision; delete
what only serves them. Diff should be almost all deletions. Status: implemented, including the three
2026-10-30 steers; remaining: gate/CI and Jack Heart's review (no merge without it).

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
- Frozen evidence (experiments, audit, tla docs) untouched; w12–w15 and other
  old-world corpora archived byte-for-byte to S3 (steer 3 below), not edited.

## Discovered while implementing

- `manabot/sim/heuristic.py` read v1/w11 sources: ported to
  `managym.card_definitions_json()` (new read-only binding); the 26 old targeted
  spells, instants and ability classes derive identically (old reader left modal
  bodies opaque; kept). Wider pool adds three spells.
  Identity: `heuristic.py` + `anchors.py`.
- 41 Academy fixtures for w7–w15 had no consumer: deleted.
- `test_learning_regression` fed a default-pack observation to an authored-pack
  policy (passed by CardDefId coincidence): uses an authored case now.
- Four tokens await producers (`TOKENS_AWAITING_PRODUCER` in observability.rs:
  Ballistic Boulder, Bear, Dragon, Monk); Soldier gained Fire Nation Attacks from T2.

## Added scope — Jack Heart steers, 2026-10-30 (implemented)

1. Tokens: all 11 in `cardsets/tokens.rs` via `token(...)`/`creature(...)` helpers.
2. Statics: one `StaticAbility` enum in `CardDefinition.statics`; `sacrifice_cost`/
   `kicker` are `AdditionalCost` variants; accessors (`ward()`, `power_cda()`,
   `optional_cost()`) read variants. Learning schema unchanged (serializer-derived tokens;
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

## Concept review, 2026-10-30 (headless, against the three steers)

Accepted model holds: `CardDefinition` is identity, characteristics, ability
lists (`abilities`, `statics`, mana/activated) and the spell's effects, targets
and `additional_costs`; all 11 tokens are `token.*` definitions;
`TOKENS_AWAITING_PRODUCER` retires itself (it asserts zero producers).
`card_definitions_json` is the one structural view of definitions for Python
(the heuristic); it reads the same serialization the learning tokens use, so it
adds no second representation. No open product choice.

Implemented 2026-10-30: `kicker()` is now `optional_cost()` (kicker or
waterbend: the cost a cast may choose to pay, which marks it kicked);
`kicker_waterbend()` is gone; `cast_waterbend_generic` and the Scryfall
conformance Kicker check match `AdditionalCost` variants directly. Wire/digest
names stay (`kicker_cost` column, content-pack JSON `"kicker"`), so no receipt
or fixture bytes move. Debug `cargo test --test cards` (310), clippy
`--all-targets -D warnings`, and `tests/sim/test_heuristic.py` (27) after a
rebuild pass. Remaining: gate/CI; Jack Heart reviews before merge.

Concept review, 2026-10-30 (second pass, after `optional_cost`): nothing to
change. `CardDefinition` holds exactly the steer's field set; every former
one-off field is a `StaticAbility` or `AdditionalCost` variant, read by match
(the remaining accessors are one-variant lookups). No definition-level field
name survives outside frozen evidence; `no_maximum_hand_size` remains only as
the player status the static sets. No open product choice.

## Demo, 2026-10-30 (headless; standing in for Jack Heart)

Ran through the real configured path (`scratch/tools/demo_afbd6a.py`, rebuilt `.so`):
- `managym.card_definitions_json()`: 207 definitions; tokens are exactly the 11
  `token.*` definitions (Ally, Ballistic Boulder, Bear, Clue, Dragon, Food, Human
  Soldier, Monk, Soldier, Spirit, Treasure); `tla.rs` has no `is_token`.
- 16 `StaticAbility` variants and 3 `AdditionalCost` variants (Kicker, Sacrifice,
  Waterbend) each appear on a real card.
- Four w17 games (ur-lessons vs gw-allies, seeds 0–3, random legal actions,
  `skip_trivial` on) ran to terminal (137–435 decisions).
- `content/semantic/` holds only `current/`; `manabot/semantic` has no compiler or
  coverage; `conformance/` is 3.1 MB, the four w17 corpora only. Live references to
  deleted roots remain only as history in `WORLDS.md` and `docs/repository-archive.md`.
- PR #403 CI: every check green at HEAD 525adc80f; mergeable.
Decision: advance. Remaining: Jack Heart's review before merge (task rule).
Page: `scratch/demo.html`.

## Gate, 2026-10-30

Diff review: no debug output, TODOs or live references to deleted roots outside
history (`WORLDS.md`, `docs/repository-archive.md`, Academy pack `source_uri`
provenance, dated chapter reviews). `docs/rules/elemental-{combat,sacrifice,temur}.md`
still name `content/semantic/v1`/`w7`/`w8` paths; they are dated w7–w9 contracts
(frozen), left as written. Reflowed one long line in `optional-payments.md`.
Checks: tracked code unchanged since 525adc80f, whose PR #403 CI is all green
(Rust, Python unit, Integration main/long, Semantic Kernel and Protocol
conformance, Academy UI/Integration, packaging); reused, no local rerun for a
docs-only reflow. Remaining: Jack Heart's review before merge.
