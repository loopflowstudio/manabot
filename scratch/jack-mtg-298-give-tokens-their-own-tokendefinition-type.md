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
- Shared printed names: `content_pack_tests.rs::tokens_sharing_a_printed_name_stay_distinct_by_key`;
  `register_token` panics on a duplicate key.
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

Concept review (2026-11-12): no change to implement. Authors now write a card (`CardDefinition`, required id)
or a token (`TokenDefinition`, key + optional id). The engine's pack row stays a `CardDefinition` with
`is_token`; that row is the object definition shared by cards, tokens and token copies, so it keeps that
name. Renaming it would churn every set file without changing behavior. An unprinted token's row carries
`scryfall_id: ""`; its public row and `check_scryfall.py` treat "" as absent.
Known consequence, out of scope: Academy curated-pack identities (`academy/curated_pack.py`,
`academy/ui/src/lib/curated-pack.ts`, `matchup.reachable_tokens`) key token art by printed name. Two
tokens sharing a name would share one identity there. No such pair exists yet (MTG-286's Zeppelin
has a unique name); key them by `token.*` key when the first pair lands.

## Definitions-level split (steer 2026-11-12, built)

Reverses the concept review's pack-row call. Built in `managym/state/src/card.rs`:
- `PrintedCharacteristics`: the shared printed values (no identity, no token status). Named
  "printed" because it is the layers' input; `Characteristics` is already the layered view
  from `Game::characteristics(id)`.
- `CardDefinition { key, scryfall_id: &str, printed, back_face: Option<Box<PrintedCharacteristics>> }`;
  `TokenDefinition { key, scryfall_id: Option<&str>, printed }`.
- `ObjectDefinition { key, scryfall_id: Option, printed: Arc<PrintedCharacteristics>, is_token,
  back_face: Option<String> }`: the pack row. Its `Serialize` writes the old `CardDefinition`
  wire shape field for field (struct name, `is_token` position, back face by name), so the
  content digest, learning tokens and match-state hashes read as before.
- `Card` holds `Arc<PrintedCharacteristics>` and its own `is_token`; copies never carry status.
  `ContentPack::object_row(card)` gives the row shape for records of a copy (pending
  `TokenSource::Copy`, canonical state of token objects).
- Deref from all three definition types to `PrintedCharacteristics` keeps reads unchanged.
- Set files: every set file under `managym/cards/src/` changed mechanically (literal nests
  `printed: PrintedCharacteristics { .. }`, outer `..Default::default()` for no back face);
  `tokens.rs` builds `PrintedCharacteristics`. Test/bench literals moved likewise.
- Frontier frozen-protocol CI failure: gone after syncing main, which reported the frontier
  result (48cb12b54) and deleted its protocol and tests; the world fixtures written for it went
  with them.
- Sync with main (2026-11-12): main's new set files nested mechanically; their tests created
  tokens by printed name (`Powerstone`, `Food`, `Spirit`, `Clue`, `Koma's Coil`), now by key.
  Re-recorded for merged digest c23a2513…: semantic-kernel-w17-v2 (only digests/state hashes
  differ from main), authored-match receipt + AUTHORITY_SHA256/TERMINAL_STATE pins, both parity
  corpora, Academy w17 fixtures, belief v3 receipts.

Checks (one line each):
- Release snapshot at HEAD vs this change: content digest aa606cea, catalog_tokens arrays,
  row ids/names/keys/token status + public definitions, record vocabulary, content pool: identical.
- `cargo nextest run --workspace --profile ci` (debug): 1058 passed. Clippy (python feature) clean.
- pytest: frontier modules 96 passed; world-change list 65 passed; belief/academy parity/
  semantic/env 385 passed.

- After sync: debug nextest 1170 passed; belief/academy/semantic/env pytest 402 passed.

Compress pass 2 (definitions split): no reductions. `impl Serialize for PrintedCharacteristics` is needed
(`TokenSource::Copy.front` serializes it); `Row`, `object_row`, `token_name` each have live callers.

Remaining: CI green on the pushed merge (no auto-merge), then stop at review.
