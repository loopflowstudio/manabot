# MTG-298: tokens as TokenDefinition (PR #476)

## Direction
Jack Heart, 2026-11-12: "tokens are not cards in the magic rules, and i think it
would be nice for that to be true in #476"; chose the definitions-level split
("ok do"). Later: "476 needs $ci-fix" — all required checks green, no
auto-merge; stop at review.

## Done (in code)
- `PrintedCharacteristics`, `CardDefinition` (required `scryfall_id`),
  `TokenDefinition` (optional id), `ObjectDefinition` row carrying `is_token`
  (`managym/state/src/card.rs`); tokens in `managym/cards/src/tokens.rs` via
  `token_catalog()`; `Effect::CreateToken { token }` by key (wire name stays
  `token_name`).
- CardDefIds, learning catalog arrays, record vocabulary, world pool rows
  unchanged; content digest moved (`c23a2513…`) and w17 evidence re-recorded
  (list in PR body). AGENTS.md/CLAUDE.md token paragraph rewritten.
- CI fully green on 6320ad43c.

## Concept review (2026-11-12): next implement direction
Types, catalogs, keys and copy/token-status handling match the direction; one
place still merges the two namespaces.

- **A deck entry names a card, whatever tokens share its name.** The pack keeps
  cards by name and tokens by key, but the world pool flattens both to
  `registry_name`, and `deck_admission::named_definition` accepts several rows
  only when all are unsupported for one reason. A card whose name a token also
  prints (Llanowar Elves beside Llanowar Mentor's token, Kobolds of Kher Keep)
  gives rows `[card Eligible, token Unsupported]` → `None` → "ambiguous", and
  the card is refused in decks. No such pair is defined today (read from
  `tokens()` on 686a4724f: Ally, Ballistic Boulder, Bear, Clue, Dragon, Elf
  Warrior, Food, Human, Human Soldier, Koma's Coil, Marit Lage, Monk,
  Powerstone, Soldier, Spirit, Treasure), so this is latent.
  Change: `named_definition` returns the single `Eligible` row when exactly one
  exists; otherwise the present rule (sole row, or a shared rejection). Two
  eligible rows stay "ambiguous"; a token-only name stays "unsupported" with
  the token reason. Add a `decklist.rs` test beside
  `tokens_sharing_a_printed_name_are_unsupported_not_ambiguous`: a pool with a
  token row renamed to an eligible card's name admits that card. Covers both
  callers (`admit_in_world`, `admit_with_world`). No digest, token or
  observation moves.
- Nothing else to change. `CardDefId` naming a token's row joins the
  out-of-scope renames below. No open product choice.

## Concept review 2 (2026-11-12): next implement direction
The deck-entry change is in and right. One more place still identifies a
definition by printed name, and it is the one that breaks the Task's outcome
("two tokens can share a printed name").

- **The learning catalog binds rows by printed name.**
  `BoundSemanticPack.bind` (`manabot/semantic/learning.py`) joins manifest
  `card_def_id`s to `managym.catalog_tokens()` rows through
  `registry_names`, a dict keyed by printed name, and raises on a repeat.
  The native pack already keeps tokens by key, so a second token of a name
  registers fine and then training cannot bind. Run on 1d5a58732 with the
  `token.ballistic_boulder` row renamed "Ally" in both the catalog names and
  the manifest: `ContentPackBindingError duplicate ContentPack registry name
  'Ally'`. Same failure for a card and a token of one name. Latent: no two
  rows share a name today.
  Change: bind on the definition's identity, not its printed name: the token
  key for a token row, the name for a card, back face or fixture (what
  `tokens_by_key` / `cards_by_name` already use natively). Carry `key` beside
  `registry_name` in `catalog_tokens` and in the manifest definitions
  (`ContentPackDefinitionManifest`, the env binding, `bind_policy_catalog`,
  which reads pool rows that already have `key`); a manifest row without a key
  binds by name as now. The join stays independent of CardDefId allocation
  (`test_semantic_only_ablation_is_independent_of_card_def_id_allocation`
  must still pass), and `binding_hash` hashes only `(card_def_id, row)` pairs,
  so no hash, digest, token or observation moves. Test in
  `tests/semantic/test_learning_projection.py`: two token rows sharing a
  printed name bind to their own rows.
- Read and left alone: `world::unsupported_cards` (eligible names only),
  `possible_worlds` / `belief/encoding` (name sets and id-name pairs over deck
  cards), Academy receipts (id to name), `content_digest` (stable sort over
  rows registered in name-then-key order).
- No open product choice.

## Remaining
- Concept-review change made in e22a18a29 (`named_definition` returns the single
  Eligible row; test `a_card_is_admitted_when_a_token_shares_its_printed_name`).
  Compressed in ff1f16a66 to one match over eligible rows (with none, all rows
  are Unsupported, so only the shared reason is compared). Debug
  `cargo test --lib -- cards::decklist cards::deck_admission`: 12 passed.
- All pushed; HEAD 24a69d953 (main merge, no `managym/` file). CI runs on
  686a4724f and e22a18a29 were cancelled by later pushes (Rust Tests had passed
  on 686a4724f). CI run 43305686368 on 24a69d953 in progress when read; PR
  body Checks names it. Remaining: that run green.
- Out of scope (own follow-up Task): renaming runtime `state::card::Card` and
  `CardDefId`;
  keying Academy curated token art by `token.*`.
- Concept review 2 change made: `BoundSemanticPack.bind` joins by key, else
  name (`catalog_tokens()["keys"]`, manifest `key`, `bind_policy_catalog`). A
  keyless manifest row binds the keyless row of its name, else the only row of
  that name; a shared name without a key raises. Cards bind by key too (same
  identity as the pool row), not only tokens.
- Contrary to "nothing moves": the env manifest gained `key`, and retained
  evidence embeds that manifest. Re-recorded (only `key` lines added; content
  digest, binding_hash, state digests, tokens and observations unchanged):
  `academy/fixtures/{learn,precision,precision-mana}-demo-w17.json`, the
  authored-match receipt (SHA now `1b0568bd…`, pinned as `AUTHORITY_SHA256`;
  `TERMINAL_STATE` unchanged) and both parity receipts. A checkpoint world
  binding stores the manifest, so one written earlier on this branch differs.
- Checks: debug `cargo test --lib -- content_pack agent::env cards::pool` 6
  passed; pytest 47 passed (learning projection, current content, policy
  catalog, projection env) and 107 passed (authority, both parities, learn,
  decision surface, world, mirror admission, belief, temur, static tokens).
  `ruff check` reports one import-order error in untouched
  `manabot/semantic/text_observation.py`.
