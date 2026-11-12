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

## Remaining
- Concept-review change made in e22a18a29 (`named_definition` returns the single
  Eligible row; test `a_card_is_admitted_when_a_token_shares_its_printed_name`).
  Compressed in ff1f16a66 to one match over eligible rows (with none, all rows
  are Unsupported, so only the shared reason is compared). Debug
  `cargo test --lib -- cards::decklist cards::deck_admission`: 12 passed.
- CI run 43300525427 on 686a4724f: Rust Tests and Engine Budget passed; two
  Integration jobs (core, evaluation) still pending when read. PR body Checks
  refreshed. CI on e22a18a29: Rust Tests and Engine Budget passed; Integration
  core pending when read. ff1f16a66 not yet pushed.
- Out of scope (own follow-up Task): renaming runtime `state::card::Card` and
  `CardDefId`;
  keying Academy curated token art by `token.*`.
