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
