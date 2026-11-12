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

## Status (2026-11-12)
Both concept-review changes are in code:
- Deck entry: e22a18a29, compressed in ff1f16a66 (test
  `a_card_is_admitted_when_a_token_shares_its_printed_name`). Debug
  `cargo test --lib -- cards::decklist cards::deck_admission`: 12 passed.
- Key binding: 9c2c6db03, compressed in 7b5cb7e72 to one identity map (key, or
  name for keyless back faces and fixtures; the fallback for manifests saved
  without keys was unreachable and is gone). pytest `tests/semantic` + learn
  setup + policy catalog: 233 passed.
- Contrary to "nothing moves": the env manifest gained `key`, and retained
  evidence embeds it, so fixtures and receipts were re-recorded with only
  `key` lines added (digests, binding_hash, tokens, observations unchanged).
  A checkpoint written earlier on this branch has a different world binding.
- `ruff check` reports one import-order error in untouched
  `manabot/semantic/text_observation.py`.

HEAD 15e50b205 merges main a343904ee (#474, training files only; no engine,
evidence or catalog file), level with origin/main. Remaining: CI on
15e50b205 green (in progress when read). PR mergeable.

Out of scope (own follow-up Task): renaming runtime `state::card::Card` and
`CardDefId`; keying Academy curated token art by `token.*`.

## Docs for the key binding (2026-11-12)
Changed: `content/semantic/README.md` (Learning projection: join by key,
`catalog_tokens()["keys"]`), `docs/engine.md` (Card
definition IDs: join on `key`), `WORLDS.md` (MTG-298 paragraph under w17),
`AGENTS.md` (identity is the key, in the token paragraph). The
`catalog_tokens()` example was read from the installed engine. No open product
choice.

## Merge of main e38518093 (2026-11-12)
PR #476 was CONFLICTING with main's typed mana restrictions (`ManaRestriction`,
`SpendingContext`, new `spm`/`woe` sets), so no checks ran on 9c2c6db03.
- Code: kept this branch's shapes with main's `ManaRestriction` calls; `spm.rs`
  and `woe.rs` nest under `PrintedCharacteristics`; `SETS` is 62;
  `SpellSnapshot` holds `Arc<PrintedCharacteristics>` (what
  `copiable_definition` returns here).
- Evidence re-recorded on the merged content (procedure in
  `conformance/README.md`): `semantic-kernel-w17-v2`, authored-match receipt
  (SHA `28ed11cc…`, terminal state `ebd8947a…`, both pinned), both parity
  receipts, the five Academy fixtures (same Commands and revisions) and the
  four belief v3 pins.

## Concept review 3 (2026-11-12): nothing to change
Read on f45aa4d71: every identity join is by key or CardDefId (pack
registration, `token_id`, `CreateToken`, learning-catalog binding), and deck
entry picks the one eligible card of a name. Remaining name reads are display
or card-only (name vocabulary, `unsupported_cards`, belief id-name pairs,
receipt id-to-name, capacity-failure text). Academy `curated_pack` identities
are still keyed by printed name with a card/token kind, so a matchup cannot
hold a card and a token of one name; that is the out-of-scope follow-up above.
No open product choice.

## Gate (2026-11-12, HEAD 4bc48211e)
Diff review: no debug code, TODOs or shims; nothing to polish. Rust is
unchanged since the debug decklist run except main's merged commits.
- `cargo fmt --all --check`: clean.
- `pytest tests/semantic/test_learning_projection.py tests/semantic/test_current_content.py tests/belief/test_state.py`: 43 passed.
- `ruff check` on changed Python: 4 findings (E402 in `managym/__init__.pyi`,
  import order in two tests), identical on origin/main; ruff is not a CI gate.
- CI on 4bc48211e: every required check passed (Tests Result green; Rust Tests 5m18s, Engine Budget 15m, Integration core 28m51s); publish/github-release skipped as on any PR.
