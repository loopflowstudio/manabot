# MTG-298: TokenDefinition split (implemented, PR #476)

Built (`managym/state/src/card.rs`):
- `PrintedCharacteristics`: shared printed values, no identity or token status. Named "printed"
  because it is the layers' input; `Characteristics` is already the layered view.
- `CardDefinition { key, scryfall_id: &str, printed, back_face: Option<Box<PrintedCharacteristics>> }`;
  `TokenDefinition { key, scryfall_id: Option<&str>, printed }` in `managym/cards/src/tokens.rs`,
  reached through `managym_cards::token_catalog()`; `catalog()` is cards only.
- `ObjectDefinition { key, scryfall_id: Option, printed: Arc<..>, is_token, back_face: Option<String> }`
  is the pack row. Its `Serialize` writes the old `CardDefinition` wire shape field for field, so the
  content digest, learning tokens and match-state hashes read as before. `Card` holds the `Arc` and
  its own `is_token`; copies never carry status. `ContentPack::object_row(card)` gives the row shape
  for copy records (`TokenSource::Copy`, canonical state of token objects).
- `ContentPack.tokens_by_key`; `token_id`, `instantiate_token`, `register_token` (panics on a
  duplicate key), `token_name`. `definition_id(name)` no longer finds tokens. Rows stay in name
  order (ties by key), so every CardDefId is unchanged.
- `Effect::CreateToken { token }` holds the key, serialized as `token_name` (learning-token field
  hash unchanged); `Game::create_token(key, ..)`.
- Shared printed names: `content_pack_tests.rs::tokens_sharing_a_printed_name_stay_distinct_by_key`.
- Offline `scryfall.rs`: card ids required, token ids optional, uniqueness across both, key shapes;
  `check_scryfall.py` skips tokens without an id. CLAUDE.md card-definition rules updated.

What moved (measured against a base-revision wheel):
- Unchanged: learning catalog arrays, `content_pool("world")` rows, row ids/names/keys/token status,
  public definitions, record vocabulary; command counts and outcomes of the 14 semantic-kernel cases.
- Content digest moved (CreateToken serializes keys); after the main sync it is c23a2513….
  Re-recorded: semantic-kernel-w17-v2 evidence, authored-match receipt + AUTHORITY_SHA256/
  TERMINAL_STATE pins, both parity corpora, Academy w17 fixtures, belief v3 receipts.
- Left as retained evidence with the old digest (tests pass): tests/fixtures/effect-execution-v8.json,
  experiments/protocols/*, docs/benchmarks/local-update.html.

Known consequence, out of scope: Academy curated-pack identities (`academy/curated_pack.py`,
`academy/ui/src/lib/curated-pack.ts`, `matchup.reachable_tokens`) key token art by printed name;
two tokens sharing a name would share one identity there. No such pair exists yet; key them by
`token.*` when the first pair lands.

Checks (latest): debug `cargo nextest run --workspace --profile ci` 1170 passed; clippy clean;
belief/academy parity/semantic/env pytest 402 passed; world-change list 65 passed.

Remaining: CI on fbfca586b (all checks pass except `preflight` pending at 2026-11-12 realign);
main has no newer commits. Then review; Engine Budget runs in CI (no gameplay change expected).

## Concept review (2026-11-12)

The model holds up: cards and tokens are separate authored definitions sharing one
`PrintedCharacteristics`; `ObjectDefinition` is the registered row; token status lives on the row
and the game object, never on `printed`; tokens resolve by key, cards by name. No open product
choice.

Next implement direction (one comment, no behavior change):
- `managym/src/flow/zones.rs` `create_token` doc says "Tokens are full cards in
  `GameState::cards`", which contradicts the split. Say tokens are game objects stored in
  `GameState::cards` beside cards.

Considered and left alone:
- `CardDefId` names rows that include tokens and back faces. Renaming it touches every engine
  and binding call site for a name only; not this Task.
- `CardDefinition.scryfall_id` defaults to `""` (needed for `..Default::default()` in set files);
  "required" is enforced by the offline `scryfall.rs` check, which is what the Task asked for.
- Game objects remain `Card` for tokens too; that is the object model, not the definition split.
