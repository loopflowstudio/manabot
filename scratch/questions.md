# Assumptions

- Token keys stay `token.<name>`; a second token with the same printed name gets a descriptive key
  (for example a plain 1/1 Spirit vs TLA `token.spirit`). Exact key not chosen; no such token yet.
- `CreateToken`'s Rust field is `token` but serializes as `token_name` to keep learning tokens
  stable. Renaming the wire name would move every token-creating card's learning tokens.
- The content digest move is unavoidable once effects name tokens by key; all w17 corpora were
  re-recorded in this change.
