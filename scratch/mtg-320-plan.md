# MTG-320: runtime object kinds by zone

Design: `docs/engine.md#objects-and-zones` (CR term -> type table, zones, token moment, "What is
built"); `managym/README.md` links it. Assumptions and naming exclusions: `questions.md`.

## Built (HEAD 2a3eae3b0, synced with main a343904ee and MTG-298 15e50b205)
- Kind: `ObjectKind { Card, Token, SpellCopy }`, private `kind` on the runtime record, set at
  construction (`with_kind` for token copies and spell copies); `resolve_into_token()` in
  `commit_zone_move` when a spell copy enters the battlefield (CR 111.12). Runtime readers use the
  kind: SBA (CR 704.5d-e), predicate/layer `token` filter, exile cast grants, "card" selection,
  return-to-battlefield guard, main's `PutIntoGraveyard` (`moved.is_card()`). `become_copy_of`
  leaves kind alone. Remaining `is_token` reads are catalog rows, deck/sideboard admission, the
  observation column (`!is_card()`) and hash wire bytes.
- Rename (steers d57fc17e, 0779d3e9, e3b2a142, dbf13009): record `Card` -> `Object` in
  `state::game_object`, `CardId` -> `ObjectId`, `CardVec` -> `ObjectVec`, `GameState::objects`,
  `object_to_permanent`, old u32 `ObjectId` -> `PublicId`; object-level helpers `remove_object`,
  `object_zones`, `matches_object[_with_power]`, `object_power`, `object_mana_value`. Serde keeps
  wire names. Incoming MTG-298 tests ported in 2a3eae3b0.
- Byte identity: observation `is_token` = `!is_card()`; hash keeps `is_token: true` for non-card
  objects. `semantic-kernel-w17-v2` conformance reproduced after the rename (one serde-attribute
  drift on `ObjectLki.counters` found by diffing traces and fixed).
- Test: Azula instant copy is `SpellCopy` and ceases off the stack.

## Checks seen (before the MTG-298 15e50b205 merge and 2a3eae3b0)
- Debug `cargo nextest --workspace --profile ci`: 1166 passed; 4 timed out at load ~250-380 and
  pass alone (conformance 508 s; base revision 345 s at the same load). Clippy `-D warnings` clean.
- pytest env/belief/semantic/verify/academy + world-change list: 907 passed, 1 failed
  (`test_v10_vector_buffers_reach_the_model`, "policy forward requires an observation from
  admit_observation"; no Python differs from main here, so inherited), 1 deselected (5-min limit).
- After the rename's compress: `cargo build --tests --workspace` ok.
- QA (release extension): 60 random games, 150 Azula games, 4 Academy games; no token or copy seen
  off battlefield/stack at a decision except Treasures mid mana choice (CR 704.3, correct).

## Concept review (2026-11-12, HEAD 9a323463f): next implement direction
The kind, the rename and the zone decision match the steers. What is left is names and doc wording
that still say "card" for something that may be a token or copy, the thing steer d57fc17e was about
("320 still needs a lot of renames"). No behavior, wire name, observation or evidence moves.

1. Helpers that move or match any object take object names:
   `Game::move_card_with_entry` -> `move_object_with_entry` (`flow/zones.rs`; a dying token goes
   through it), `capture_card_move` -> `capture_object_move` (`flow/card_capture.rs`),
   `subject_matches_moved_card` -> `subject_matches_moved_object` (`flow/triggers.rs`; its parameter
   is already `moved_object`), and in `assert_stack_consistent` `card_counts`, `zone_spell_cards`,
   `stack_spell_cards` -> `object_counts`, `zone_spell_objects`, `stack_spell_objects` (the stack
   holds copies). Stale `move_card` mentions: comment at `flow/scenario.rs:107`, expect message at
   `tests/rules/cr_603_triggers.rs:21`, and the typed-accessor list in `AGENTS.md`/`CLAUDE.md`
   (line 438; the accessor is `move_object`).
   Left as `card` on purpose (hand, library, graveyard or exile selection, where the rules text says
   card): `priority_candidate_cards`, `priority_action_for_card`, `draw_cards`, `library_top_card`,
   `revealed_cards`, `exile_granted_cards`, `foretold_cards`, `Target::GraveyardCard`.
2. Doc comments on object paths: `state/src/zone.rs:48,110-111` ("holds `card`", "Move `card`",
   "the card's owner" -> object; owner rule CR 108.3 applies to tokens through CR 111.2),
   `game_object.rs:155` ("The card's most recent departed battlefield object"),
   `stack_object.rs:127,135` ("a spell by its card", "The spell's card").
3. `docs/engine.md#objects`: the two paragraphs after the table ("Every zone change makes a new
   object ...", "code that means 'which object' ... goes stale when the object moves") state the
   unbuilt model as present fact; only the table's Card row and the later "What is built" say
   otherwise. Today `ObjectId` is the row that survives a zone change and `ObjectRef` is the
   appearance, the reverse of what those paragraphs say about `ObjectId`. Mark them planned in place
   (one lead sentence, or move them under a "Planned split" heading after "What is built") so a
   reader of the table and the first paragraph gets the built meaning of `ObjectId` first.
4. PR #491 body (already listed under Remaining): also state that library and hand lists hold
   `ObjectId`s, not card ids. Steer dbf13009 says "Card-only zones (library, hand, decklists) hold
   card ids"; the built decision differs because a bounced or tucked token is in that zone until
   state-based actions (CR 111.7-111.8, 704.5d) and observations must stay byte-identical (QA saw
   Treasures in a graveyard at a mana-choice decision). Say so plainly for the reviewer.

No open product choice for this PR. For the later split Task, not here: whether library and hand
become card-id lists (a token would then cease to exist inside the move, which changes what is
observable between the move and state-based actions) or stay object lists as now.

## Remaining
- CI on 2a3eae3b0 (run 43325427802) was pending when read: Rust Tests (debug), Engine Budget
  (digest should not move), conformance, integration. It is the first full check after the
  15e50b205 merge.
- PR #491 body is stale: it says the record keeps the name `Card`, lists the two main readers as
  unmoved, and gives old check counts. Rewrite at publish: rename built, readers moved, current
  checks, and "left for later" below.
- Stack parent PR #476 (MTG-298) is open; it lands first.
- Left for later (steer allows "land the rename plus the typed ids here, and say what is left in
  the PR body"): separate `Card`/`CardId` store, new object per zone change, the CR 400.7
  carry-over record, measuring their layer/`TriggerIndex` invalidation cost.
