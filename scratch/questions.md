# Assumptions

- Token keys stay `token.<name>`; a second token with the same printed name gets a descriptive key
  (for example a plain 1/1 Spirit vs TLA `token.spirit`). Exact key not chosen; no such token yet.
- `CreateToken`'s Rust field is `token` but serializes as `token_name` to keep learning tokens
  stable. Renaming the wire name would move every token-creating card's learning tokens.
- The content digest move is unavoidable once effects name tokens by key; all w17 corpora were
  re-recorded in this change.
- Shared type named `PrintedCharacteristics`, field `printed`, row `ObjectDefinition` (steer asked
  for a non-colliding name). Back face stays on `CardDefinition`, not on the shared type.
- Set-file literals use an outer `..Default::default()` for the absent back face rather than a
  constructor; docs/engine.md shows it. A `CardDefinition::new(key, id, printed)` would save a
  line per card but drop the field labels.
- `ObjectDefinition.back_face` is the back face's name (what the wire and learning tokens
  read); the back face's characteristics are the next row's.

## MTG-320
- Emblem: room only. No `ObjectKind::Emblem` or command-zone admission until a card creates one
  (nothing would use it). Docs say "none yet".
- The record rename is no longer deferred: steers of 2026-11-12 asked for it in this PR (see below).
- Predicate `token` filters now match `ObjectKind::Token` only; spell copies used to match (they were
  tokens). No defined card filters spells by token status, so no behavior moves.
- Spell copies stop being tokens internally (CR 704.5e); observation `is_token` stays true for them.
- Rename as built (steer dbf13009 asks for `Object`/`ObjectId`): record `Card` -> `Object`, moved to
  `state::game_object`; `CardId` -> `ObjectId`; `CardVec` -> `ObjectVec`; `GameState::cards` ->
  `objects`; `card_to_permanent` -> `object_to_permanent`. The old `ObjectId(u32)` (public match id
  for players, objects, abilities and timestamps) collided, so it is now `PublicId`.
- Locals: `card` -> `object`, except in functions that already bind `object` (an `ObjectRef`), where the
  id is `object_id` (and `cards` -> `object_ids`). Serialized fields keep wire spelling via
  `#[serde(rename = "card")]`; Debug output of these types does change (type and field names).
- Kept `card`: observation encoders/receipt writer (`agent/observation_*`, `effect_execution.rs`,
  observation `CardData`, `StackTargetData.card_id`), Python bindings, card definitions,
  content pack, decklists/setup, sideboard, hidden-card worlds (`possible_worlds.rs`), card-name
  string fields (`PublicCommitment`, `HandSnapshot.cards`).
- Token moment: zone lists hold `ObjectId`s, library and hand included (CR 111.8 leaves a moved token
  in the zone it went to); only decklists/sideboards are card-only. Documented in engine.md.
- Not built here (left for the split): separate `Card`/`CardId` store, new object per zone change,
  the CR 400.7 carry-over record, the performance measurements. Steer allows "land the rename plus
  the typed ids here, and say what is left in the PR body".
