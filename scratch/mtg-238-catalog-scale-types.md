# MTG-238: catalog-scale types

Task: sparse counters, keyword enum, card-set crate, animation as a record.
Base: main `4c8324a59` (T4). MTG-225 (T6) had not published when this started
(2026-11-02); its `CreatureWithNamesakes` static does not exist on this base,
so that conversion waits for the sync onto T6 (see questions.md).

## Design (accepted shape, 2026-11-02)

1. **Counters** — `Counters(SmallVec<[(CounterKind, i32); 2]>)`, entries kept
   in `CounterKind` order with positive counts only. Same API (`get`, `change`,
   `iter`, `is_empty`) and the same serialization (map of present kinds in
   kind order), so neither content digests nor state hashes move. No longer
   `Copy`: the engine's by-value copies become clones.
2. **Keywords** — `enum Keyword` (14 variants, all evergreen today) and
   `Keywords` as a `u64` bitset: `has`, `insert`, `remove`, `union`,
   `without`, `names`, `iter`, `Keywords::of(&[Keyword])`. Serialization
   stays a struct of every keyword name to its bool in enum order, so card
   definitions (content digest) and the learning tokens (serde walk) are
   byte-identical. Evergreen keywords keep their card/permanent tensor columns
   through `KeywordData`; the observability guard matches `Keyword`
   exhaustively, so the first non-evergreen keyword must name its fact row.
   Set files write `Keywords::of(&[Keyword::Flying])`.
3. **Animation** — `Permanent.animated` is gone. Earthbend records three
   `ContinuousEffect`s at one timestamp on the land: `BecomeCreature` (new
   layer-4 variant, a payload-free sibling of `BecomeArtifactCreature`),
   `SetBasePt { 0, 0 }` (7b) and `AddKeywords(haste)` (6), `Indefinite`, so
   they end when the object leaves (CR 400.7) exactly as the flag did.
   `Game::animate_permanent` is the one writer (tests use it). `is_animated` =
   a `BecomeCreature` record, the card's own counter threshold, or
   `BecomeArtifactCreature`. Compress (2026-11-02) replaced the first draft's
   `AddCardTypes(Vec<CardType>)`: the engine only ever read it as "contains
   Creature", so the payload and its observation/Python plumbing were
   speculative. `BecomeArtifactCreature` stays because folding it in would
   rewrite six crew definitions and move the w17 content digest. Removing a
   serialized `Permanent` field changes the canonical encoding, so
   `MATCH_STATE_HASH_VERSION` becomes 12 and every corpus embedding state
   digests is re-recorded (semantic-kernel, authored-match and both parity
   receipts, Academy fixtures); the content digest is unchanged.
4. **Crates** — `managym/Cargo.toml` becomes a workspace root (package +
   members). `managym-state` (`managym/state/`) holds the former `src/state/`
   minus `hash.rs`; the engine re-exports it as `managym::state` and keeps
   `state::hash`. `managym-cards` (`managym/cards/`) holds the set modules and
   `catalog()` / `incomplete()`; the engine reaches cards only through those.
   `tests/cards` moves to `managym/cards/tests/` (engine as dev-dependency;
   `tests/support` mounted by path so each primitive stays defined once).
   `cargo test` in `managym/` runs all three through `default-members`.

## Delete — do not maintain

- `Permanent.animated`, `Permanent::animated_creature`, the animation branch
  of `Permanent::intrinsic_keywords`, `statics.rs` layer-7a animated base.
- `managym/src/cardsets/` (moved), `managym/tests/cards/` (moved),
  `Keywords` struct fields and the 14-arm `without`/`union`/`names` tables,
  `keyword_columns`'s destructuring, `set_keyword` in observability.

## Checks

- `cd managym && cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings`
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test` (debug) and `cargo test --release`
- extension rebuild, then `uv run --extra dev pytest tests/verify/test_observation_parity.py tests/semantic/test_static_ability_tokens.py -q`

Check result (2026-11-02, after compress): `cargo fmt --check`, clippy
`--all-targets --all-features -D warnings` and debug (`OPT_LEVEL=1`) `cargo
test` green across the three crates (484 card tests); extension rebuilt; the
`BecomeCreature` rename moved every state digest with an earthbent land, so
semantic-kernel evidence, the authored-match receipt, both parity receipts and
the Academy fixtures were re-recorded again (same Commands; presentation pins
unchanged); `pytest tests/env tests/academy tests/belief tests/semantic
tests/verify tests/model tests/test_package_boundaries.py` green (861 passed).
Release `cargo test` and the arena-admission suite in CLAUDE.md are gate's.
