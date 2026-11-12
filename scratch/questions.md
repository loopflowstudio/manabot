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
- Kept the runtime record named `Card` and the table `GameState::cards`; the kind distinguishes objects
  (the target's `Card.is_token` field merged into `kind`).
  A rename (e.g. to an object record) would touch ~500 sites; Jack Heart: "im not sure the
  Card->GameObject rename is totally clean". Left for review; docs list it as a departure.
- Predicate `token` filters now match `ObjectKind::Token` only; spell copies used to match (they were
  tokens). No defined card filters spells by token status, so no behavior moves.
- Spell copies stop being tokens internally (CR 704.5e); observation `is_token` stays true for them.
