# MTG-298: TokenDefinition split (implemented)

Built:
- `managym_state::card::TokenDefinition { key, scryfall_id: Option, characteristics: CardDefinition }`;
  `object_definition()` is the pack row (key, id or "", `is_token: true`). `CardDefinition.is_token`
  stays as object status (token rows and token copies), so observations and copies are unchanged.
- `managym_cards::token_catalog()` (from `tokens::tokens()`); `catalog()` is cards only.
- `ContentPack`: `tokens_by_key` beside `cards_by_name`; `token_id`, `instantiate_token`,
  `register_token`, `token_name` (descriptions). `definition_id(name)` no longer finds tokens.
  Rows stay in name order (ties by key), so every CardDefId is unchanged.
- `Effect::CreateToken { token }` holds the key, serialized as `token_name` so learning tokens keep
  their field hash; `Game::create_token(key, ..)`; internal `proposed_event::TokenSource`.
- Offline `scryfall.rs`: card ids required, token ids optional, uniqueness across both, key shapes.
  `check_scryfall.py` skips rows without an id.

What moved / stayed (measured against a base-revision wheel):
- Learning catalog arrays and `content_pool("world")` rows: byte-identical to base.
- Content digest moved 0798b2c7… -> aa606cea… (CreateToken now serializes keys). Command counts and
  outcomes of all 14 semantic-kernel cases unchanged. Re-recorded: semantic-kernel-w17-v2 evidence,
  authored-match receipt + AUTHORITY_SHA256/TERMINAL_STATE pins, both parity corpora, Academy w17
  fixtures, belief v3 receipts in tests/belief/test_state.py.
- Left as retained evidence: tests/fixtures/effect-execution-v8.json, experiments/protocols/*,
  docs/benchmarks/local-update.html (they carry the old digest; tests pass).

Checks: `cargo nextest run --workspace --profile ci` (debug) 1055 run, 2 failed before re-record/fix,
then both pass; clippy clean; focused pytest (belief, effect execution, academy parity, choice nav,
trigger order, text observation, learning projection, current content) pass.

Remaining for gate: CLAUDE.md world-change pytest list with all extras; Engine Budget in CI
(no gameplay change expected).

Compress pass: no reductions found; the diff has no obsolete path left (token lookup by name is gone, `register` is the one row path, `TokenSource` avoids a second `TokenDefinition`). No rerun needed.
