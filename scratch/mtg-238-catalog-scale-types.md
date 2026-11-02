# MTG-238: catalog-scale types

Task: sparse counters, keyword enum, card-set crate, animation as a record,
`CreatureWithNamesakes` as a counted predicate.
Base: main `4c8324a59` (T4). The Task says to start after MTG-225 (T6)
publishes and to stack on the effect-vocabulary Task (MTG-235) if both are
open; neither had a PR when this started (2026-11-02) nor at the realign or
the second implement that day, so this branch sits on main (see Status).

## Design (accepted shape, 2026-11-02; batch carried the same day)

1. **Counters** — `Counters(SmallVec<[(CounterKind, i32); 2]>)`, entries kept
   in `CounterKind` order with positive counts only. Same API (`get`, `change`,
   `iter`, `is_empty`) and the same serialization (map of present kinds in
   kind order), so neither content digests nor state hashes move. No longer
   `Copy`: the engine's by-value copies become clones.
2. **Keywords** — `enum Keyword` (14 variants, all evergreen today) and
   `Keywords` as a `u64` bitset: `has`, `insert`, `union`, `without`,
   `names`, `iter`, `Keywords::of(&[Keyword])` (no `remove`; `without`
   is the only subtraction, as the engine's layer-6 loss needs). Serialized
   as the sequence of keywords present in enum order (the `Counters` shape):
   a definition's tokens carry only its keywords and a new keyword changes no
   other card's encoding. Evergreen keywords keep their card/permanent tensor
   columns through `KeywordData`; the observability guard matches `Keyword`
   exhaustively, so the first non-evergreen keyword must name its fact row.
   Set files write `Keywords::of(&[Keyword::Flying])`.
3. **Animation** — `Permanent.animated` is gone. Earthbend records three
   `ContinuousEffect`s at one timestamp on the land: `BecomeCreature`
   (layer 4), `SetBasePt { 0, 0 }` (7b) and `AddKeywords(haste)` (6),
   `Indefinite`, so they end when the object leaves (CR 400.7) exactly as the
   flag did. Crew's "becomes an artifact creature until end of turn" is the
   same `BecomeCreature` record (a Vehicle is a printed artifact);
   `Game::animate_permanent` is the one earthbend writer. Removing a
   serialized `Permanent` field changes the canonical encoding, so
   `MATCH_STATE_HASH_VERSION` is 12.
4. **Counts as conditions** — `StaticTypeGrant` has an optional `condition`
   (`skip_serializing_if`); War Balloon is `TypeGrant { This, [Creature],
   SourceCountersAtLeast { Fire, 3 } }`. `StaticLandType.while_counters`
   is `condition: Condition` (Zhao: `SourceCountersAtLeast { Conqueror, 1 }`).
   `Condition::ControlsMatching` is `ControlsAtLeast { count, predicate }`.
   `Game::permanent_is_animated` = a `BecomeCreature` record or the
   permanent's own `This`-scoped Creature grant whose condition holds (no
   recorded ability loss); it reads only the card and records, never the
   static index, which asks it while being built. A type grant's condition is
   asked of its source (`source_condition_holds`) and predicate matching asks
   animation only for type-reading predicates, so a counter or name question
   never recurses.
5. **Crates** — `managym/Cargo.toml` is a workspace root (package +
   members). `managym-state` (`managym/state/`) holds the former `src/state/`
   minus `hash.rs`; the engine re-exports it as `managym::state` and keeps
   `state::hash`. `managym-cards` (`managym/cards/`) holds the set modules and
   `catalog()` / `incomplete()`; the engine reaches cards only through those.
   `tests/cards` lives in `managym/cards/tests/` (engine as dev-dependency;
   `tests/support` mounted by path so each primitive stays defined once).
   `cargo test` in `managym/` runs all three through `default-members`.

The content digest moved once, with items 2–4 together (2026-11-02):
semantic-kernel evidence, the authored-match receipt, both parity receipts,
the Academy fixtures and the belief v3 receipt pins were re-recorded with the
same Commands and decision revisions; the learning schema stays 10 (tokens
derive from the definitions).

## Status (implement 2026-11-02, third pass)

On the branch: items 1–5, WORLDS.md / managym README / AGENTS.md /
conformance README records, the re-recorded corpora,
`scripts/refresh_academy_fixtures.py`, the rebuilt extension.

Remaining, all at the sync onto T6. MTG-225 still has no PR; its branch has
merged T5 (MTG-224, PR #405 onto main), which is where the Airship and the
`Keywords` literals below come from, so one sync takes both:

- Phoenix Fleet Airship: T5's `StaticAbility::CreatureWithNamesakes(u32)`
  (its `card.rs`, `statics.rs` reader, `Card::creature_with_namesakes`,
  `Game::creature_with_namesakes` and the one `tla.rs` definition) becomes
  `TypeGrant { This, [Creature], ControlsAtLeast { 8, namesake } }` with
  `CardPredicate.namesake` (shares the source's name; no name enters the
  vocabulary). The field arrives with its reader, not before.
- Convert T5/T6's `Keywords { x: true, ..Default::default() }` literals and
  `.keywords.x` reads to `Keywords::of(&[Keyword::X])` / `.has(Keyword::X)`.
- Carry T5/T6's `src/state/*` and `src/cardsets/tla.rs` edits across the
  directory move (`managym/state/src/`, `managym/cards/src/`). T6 adds a
  `copy: Option<CopyEffect>` field to the hashed card encoding with
  `skip_serializing_if`, so it needs no further hash-version change.

Landing order (Jack Heart's steer, 2026-11-02): the batch is approved; hold
until MTG-225 publishes, then sync and publish. MTG-235 (#412) and MTG-236
(#411) are published onto the same base and both edit `managym/src/state/*`,
so whichever lands first also crosses this branch's crate move at the sync.

Follow-on outside this Task: `PyKeywords` (`python/bindings.rs`) is still a
14-bool struct on the Python surface (`.keywords.flying` in Academy and text
observation); its sparse form is a Python API migration for the first
non-evergreen keyword.

## Delete — do not maintain

Done: `Permanent.animated` and its readers, `managym/src/cardsets/` and
`managym/tests/cards/` (moved), the `Keywords` struct fields and tables,
`StaticAbility::CreatureWithCounters`, `CounterThreshold`,
`Card::creature_with_counters`, `Permanent::creature_by_counters`,
`Modification::BecomeArtifactCreature`, `Game::becomes_artifact_creature`,
`Condition::ControlsMatching`, the struct-form `Serialize for Keywords`.
At the T6 sync: `CreatureWithNamesakes`, `Card::creature_with_namesakes`,
`Game::creature_with_namesakes`.

## Checks

- `cd managym && cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings`
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test` (debug) and `cargo test --release`
- extension rebuild, then `uv run --extra dev pytest tests/env tests/academy tests/belief tests/semantic tests/verify tests/model tests/test_package_boundaries.py -q`

Check result (2026-11-02, second implement): `cargo fmt --check`, clippy
`--all-targets --all-features -D warnings` and debug (`OPT_LEVEL=1`) `cargo
test --no-fail-fast` green across the three crates (484 card tests); extension
rebuilt; corpora re-recorded once (kernel 14 cases / 3396 Commands,
authored-match and parity 142 Commands, public commitment 69 commitments, five
Academy fixtures, belief v3 pins; presentation pins unchanged); the Python
suite above 864 passed, 1 skipped, then `tests/academy tests/verify
tests/semantic` 494 passed after the capture-keyword fix and receipt rewrite.
Release `cargo test` and the arena-admission suite in CLAUDE.md are gate's.
Third pass (2026-11-02): no code change; `cargo fmt --check` and the debug
`definitions` / `observability` card tests (47) pass on the checkpointed tree.
