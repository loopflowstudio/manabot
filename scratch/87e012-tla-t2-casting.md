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
- Learning schema 8 grows append-only (as T1's `1e0d3c998`).
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
- Schema 8 also gained opcodes 70–73 (`sacrifice_source`,
  `put_into_library_second_or_bottom`, `search_lands_opponent_splits`,
  `reveal_hand_discard_chosen` with public `known_hand`), fields through
  `foretell` (101) and predicate `any_of`.
- Targeted waterbend activations carry their target in
  `PendingChoice::Waterbend`; a {T} ability can't waterbend-tap its source.
  "Discard a card" cost: `PendingChoice::DiscardPayment`, before mana.
- Conditional flash → `CardDefinition.flash_if`; `Game::has_instant_timing`.
- Cast-time announcements are one `CastChoices { kicked, x, free }` in
  `PendingChoice`/`Payment::Spell`; `can_cast_with`/`can_pay_cast` own
  castability for priority and free-cast offers.
- Exile grants carry `cost` (None = mana cost), `this_turn`, `source`,
  `own_turn_only`, `waterbend`, `flash`, `any_mana`, `after_turn`, `foretold`;
  granted lands play from exile (`may_play_from_zone`). Opcode 69 (Zuko).
- Foretell: `Cost.foretell` → `CardDefinition.foretell`; `Action::Foretell`
  (type `PriorityActivateAbility`, card focus) pays {2} by automatic mana and
  exiles face down off the stack. The `foretold` grant is the face-down
  authority (`Game::is_foretold`). `ZoneEntry` (was `BattlefieldEntry`)
  carries `face_down` onto `CardMoved`: observation events and policy history
  name no card, known-hand facts clear. Observations list it only for its
  owner (`CardData.face_down`); Academy's `redact_observation` drops an
  opponent's face-down exile (protocol `LegacyCardView.face_down`, schemas
  regenerated); `determinize` resamples it from unseen cards with foretell.
- Control: each player's battlefield lists permanents they control
  (`ZoneManager` holder = owner, or controller on the battlefield; undo
  records it). A resolving permanent spell enters under its controller
  (`ZoneEntry.controller`, CR 110.2); cast triggers read `spell_caster`;
  observations pair battlefield cards with their controller.

## Done (`648edee6b`…`2673c95cd`; foretell `1144d0f32`; control `431a31f65`)

All 51 T2 cards are defined and none is in `tla::INCOMPLETE` for T2
behavior (2026-10-29, against `allocation.jsonl`). Each has a witness in
`tests/cards/tla.rs` and a Scryfall fixture entry (`scratch/add_fixture.py`).

## Base: T1

Merged onto PR #394's head `86ebe9ccf`. Moving the per-card `tests/rules/`
files into `tests/cards/` stays T1's (Jack Heart, 2026-10-29).

## Remaining

- Gate: world-change Python suite, Academy check (UI shows own foretold cards
  as "foretold, face down" and an opponent's as a count), `--features python`
  clippy.
- Known gaps (questions.md): possible-worlds hidden-hand constraints ignore
  foretold cards; granted land plays of another player's card would enter
  under its owner (no TLA producer).

## Check

2026-10-29 at `431a31f65`: debug `cargo test --test cards --test rules_tests
--lib` (+ search/hash/scenario/protocol-schema tests) pass; clippy and fmt
clean. `pytest tests/semantic/` fails only T1's three frozen-schema
assertions (expect 3/7/4, bind 8); `tests/academy/test_play_modes.py` passes.
