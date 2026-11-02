# T6 (MTG-225): Sagas, faces, face-aware copies and turn scheduling — plan

Status (2026-11-02): implementing. Brief: Task MTG-225 (16 cards, allocation
`planned_pr: T6`); direction in [t6-direction.md](t6-direction.md). One
cohesive PR, then stop for Jack Heart's review.

Base: T6 started on main `4c8324a59` (T3 + T4) while T5 #405 synced onto
main, then merged T5's synced head `836042c07` (2026-11-02, four small
conflicts), so it is stacked on T5 with T5's copy effects available.

## Model

- **Faces (CR 712).** A transforming double-faced card is one definition
  function whose `CardDefinition.back_face: Option<Box<CardDefinition>>`
  holds the back face. The content pack registers the back face as its own
  row right after its front, so a transformed permanent's observation row,
  learning tokens and text name the back face. Back-face rows are not deck
  cards (pool reason, deck admission). The front's tokens carry a definition
  reference to the back row (like `token_name`).
- **Face state on the card.** `Card` keeps its front row while the back face
  is up (`Card::transform` swaps `definition_id`/`definition`); leaving the
  battlefield turns it front face up (CR 712.8a, 400.7). Transforming keeps
  the same permanent (counters, damage, attacking; CR 712.9). "Return it
  transformed" enters with the back face up (`ZoneEntry.transformed`,
  CR 712.14). A card with no back face (a token copy) ignores transform
  instructions (CR 701.28c).
- **Sagas (CR 714).** `CounterKind::Lore`. Chapter abilities are
  `TriggerCondition::Chapter { n }` triggered abilities ("I, II" = two
  abilities). A Saga enters with one lore counter (714.3a); after the draw
  step the active player adds one to each Saga they control (714.3b, turn-based
  action as the precombat main phase begins). Adding lore counters triggers
  each chapter whose number was crossed (714.2b). SBA 714.4: sacrifice a Saga
  with lore ≥ its final chapter unless one of its chapter abilities has
  triggered and not left the stack.
- **Turn scheduling.** Extra turns are a queue on `TurnState` (CR 500.7,
  most recent first); extra combat phases are an inserted combat phase
  (CR 500.8) with an optional attack restriction (Bumi).

## Cards

| Card | State |
| --- | --- |
| Leaves from the Vine | supported; directed tests |
| The Cave of Two Lovers | supported; directed tests |
| The Legend of Kuruk // Avatar Kuruk | supported; transform reentry, Spirits, extra turn |
| The Legend of Kyoshi // Avatar Kyoshi | supported; directed test |
| The Legend of Roku // Avatar Roku | supported; directed test |
| The Legend of Yangchen // Avatar Yangchen | |
| The Rise of Sozin // Fire Lord Sozin | |
| Aang, Swift Savior // Aang and La | supported; airbends spells, transforms in place |
| Aang, at the Crossroads // Aang, Destined Savior | |
| Avatar Aang // Aang, Master of Elements | |
| Bender's Waterskin | supported |
| Bumi, Unleashed | |
| Koh, the Face Stealer | |
| Ember Island Production | supported; copies the face that's up, transformed |
| Fire Lord Azula | |
| Foggy Swamp Visions | supported |

## Delete — do not maintain

- `Effect::ExileTopCardPlayThisTurn`: replaced by `ExileTopCardsToPlay {
  count, until: PlayWindow }`. Done.

## Checks

- 2026-11-02: debug `cargo test --tests` green except Scryfall fixture (fixed) and
  semantic-kernel w17 evidence (content digest moved; re-record at gate).
- 2026-11-02 (after T5 merge, extension rebuilt): debug lib/cards/conformance/
  rules targets green; pytest tests/semantic tests/env + parity 276 passed;
  experience protocol 14 passed. Academy UI checks not run (no node_modules).
