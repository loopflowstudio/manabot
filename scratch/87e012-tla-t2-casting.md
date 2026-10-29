# T2 — TLA casting, costs, zone permissions and private zones (Task 87e012)

Stacked on T1 (892c47, PR #394 at `be2b7586b`). Task directive lists the 51 cards
and obligations; allocation in `docs/rules/tla/allocation.jsonl` (`planned_pr: T2`).
Every merge needs Jack Heart's review. Card support = defined in the set file
(Jack Heart, 2026-10-29); T1 is removing the qualification gate in its own
worktree, so new T2 cards read "unsupported" in this base until T1 lands.

## Decisions (reversible operator choices: questions.md)

- 2026-10-29, Jack Heart: card tests live in `managym/tests/cards/<set>.rs`
  (one `tests/cards.rs` entry); moving existing per-card rules tests is T1's.
- Activated abilities carry an effect sequence (`ActivatedAbilityDefinition.effects`).
- Printed flashback is `Cost.flashback` → `CardDefinition.flashback`.
- Earthbend takes a `Count`; printed counts stay `EffectValue::Fixed` (IR unchanged).
- Learning schema 8 grows append-only (as T1's `1e0d3c998`); no new opcode yet.
- Card-zone activation: `ActivationSource` (permanent or card) in
  `Payment::Activation`/`PendingChoice::ActivatedTarget`; hand/graveyard
  abilities are `Action::ActivateCardAbility { card, ability_index }` (type
  `PriorityActivateAbility`, card focus), indexing printed abilities in program
  key order. Landcycling = opcode 24 with `destination: hand`; graveyard
  self-return = `return_to_hand` targeting `source`; Team Avatar uses
  `attacks_alone` + role `triggering_permanent`.
- Exhaust/once-each-turn: `Cost.exhaust`/`once_each_turn` → ability flags;
  `Permanent::{activation_used, record_activation}` own both limits
  (`exhausted` for the object's life, `activated_this_turn` cleared at cleanup),
  recorded at mana activation or stack push. Predicate `with_counter`.
- Restricted mana is keyed by `ManaUse` (state/mana.rs): subtype bit masks over
  `SPEND_SUBTYPES` (Lesson, Shrine, Ally) for spells and ability sources.
  `Mana.restricted` = sorted boxed classes; `ManaCost.purpose` is set from the
  cast spell's subtypes or, in `price_activation`, the source's; narrowest
  usable class spends first. Opcode 56 `add_lesson_mana` takes optional
  `spend_only` (omitted = Lesson spells). `cost_reduction_per` (Cost
  `affinity`) reduces generic activation cost.
- Casting outside hand: `flow/cast_permission.rs` owns `GameState.cast_permissions`
  — exact-`ObjectRef` exile grants (stale once the card leaves exile) and
  `HandOnly` restrictions that override every non-hand permission until the
  restricter's next untap. `Game::zone_cast_cost` is the one non-hand cost
  authority (graveyard flashback, exile grants). Hash omits the field when empty.
- Airbend = opcodes 66 `airbend {target}`, 67 `airbend_all {except, predicate}`
  (inside a target clause; spares the chosen target), 68
  `opponents_cast_only_from_hand`. `Exile {target: "source"}` exiles the resolving
  spell; finalize leaves a spell that already left the stack. Targets take
  `another` ("other target", trigger source excluded); predicate `attacking`.
- Schema 8 also gained opcodes 70 `sacrifice_source`, 71
  `put_into_library_second_or_bottom`, 72 `search_lands_opponent_splits`
  (decisions `SearchDistinctLands` → `OpponentSplitsRevealed`), 73
  `reveal_hand_discard_chosen` (public `known_hand`; `DiscardFromRevealedHand`
  offers one card per definition); fields `up_to`, `milled_any`+`then`,
  `affinity_zone`, `opponent_picks`, `discard_card`; predicate `any_of`.
- Targeted waterbend activations carry their target in
  `PendingChoice::Waterbend`; a {T} ability can't waterbend-tap its source.
- "Discard a card" cost: `PendingChoice::DiscardPayment`, before mana.
- Conditional flash lowers to `CardDefinition.flash_if`;
  `Game::has_instant_timing` owns timing.
- `determinize` pins revealed cards in their owner's library.
- Exile grants carry `cost: Option` (None = the card's mana cost) and `this_turn`
  (dropped at cleanup); lands with a grant are played from exile
  (`may_play_from_zone` gates actions and `play_land`). Opcode 69
  `exile_top_card_play_this_turn` (Zuko).

## Done (slices 1–2 `648edee6b` → T1 merge `f7b59aa99`; 3 `afff1e1ed`; 4a `0d124a66e`; 4b `dea25eda8`, `d84c5f590`; T1 review `c223ea581`; slice 5/6 `43ef1210f`…)

44 of 51 T2 cards are defined (2026-10-28, against `allocation.jsonl`); 7
remain (Next slices). The six T1 cards were reviewed and left `tla::INCOMPLETE`
in `c223ea581`. Each T2-defined card has a witness in `tests/cards/tla.rs` and
a Scryfall fixture entry (`scratch/add_fixture.py <name>` appends one);
per-slice card lists are in the commit messages above.

## Base: T1

Merged onto PR #394's head `86ebe9ccf`. Moving the per-card `tests/rules/`
files into `tests/cards/` stays T1's (Jack Heart, 2026-10-29); re-sync if T1
lands that move.

## Next slices (one PR)

Remaining 7 T2 cards, each needing a casting-pipeline mechanism not built yet:

- United Front: {X} spells — announce X while casting (stack object keeps it,
  mana value includes it), `EffectValue::X`; then X Ally tokens and a counter
  on each creature you control (`PutCountersOnEachMatching` exists).
- Spirit Water Revival: optional additional waterbend {6} while casting (spell
  waterbend payment), "if the additional cost was paid" branch, shuffle
  graveyard into library, draw seven, no maximum hand size, self-exile.
- Yue, the Moon Spirit and Solstice Revelations: cast a spell *during
  resolution* without paying its mana cost (decide: nested cast decision vs. a
  granted one-shot free cast). Solstice also exiles until a nonland, compares
  mana value to Mountains you control, falls back to hand, and has flashback.
- Sozin's Comet: foretell special action (pay {2}, exile face down from hand;
  observation hides the face from the opponent) + later-turn cast for {2}{R};
  "each creature you control gains firebending 5 until end of turn".
- Hama: target opponent mills three, exile up to one noncreature nonland card
  from their graveyard, source-linked own-turn grant paid by waterbend {X}
  (X = its mana value).
- Azula: opponent-chosen exiles (nontoken creature, then nonland graveyard
  card), source-linked own-turn grant with flash and any-type mana.

## Check

2026-10-28 compress (discard-cost payment, discard-draw arms): debug `cargo test --test cards --test rules_tests --lib` passes; clippy and fmt clean. Earlier full debug run (`fb624ef92`): all pass except the pre-existing semantic-conformance pair; `pytest tests/semantic/` fails only T1's frozen-schema assertions (expect 3/7/4, bind 8). World-change Python suite and Academy check deferred to gate.
