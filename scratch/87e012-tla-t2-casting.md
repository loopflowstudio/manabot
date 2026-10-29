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
- Exile grants carry `cost: Option` (None = the card's mana cost) and `this_turn`
  (dropped at cleanup); lands with a grant are played from exile
  (`may_play_from_zone` gates actions and `play_land`). Opcode 69
  `exile_top_card_play_this_turn` (Zuko).

## Done (slices 1–2 at `648edee6b`, merged onto T1 at `f7b59aa99`; slice 3 `afff1e1ed`)

T1 defined 7 (Badgermole Cub, Dai Li Agents, Haru, Toph, Hermitic Herbalist,
Iroh, Grand Lotus; witnesses in `tests/rules/`). T2 defined 17, each witnessed
in `tests/cards/tla.rs` with a Scryfall fixture entry: Earthbending Lesson,
Sandbenders' Storm, Earth Village Ruffians, Geyser Leaper, Foggy Swamp
Vinebender, Fire Nation Attacks, Rockalanche; slice 1: Canyon Crawler,
Mongoose Lizard, Saber-Tooth Moose-Lion, Giant Koi, Rabaroo Troop, Team Avatar,
Merchant of Many Hats; slice 2: Hog-Monkey, Rough Rhino Cavalry, Rebellious
Captives; slice 3: Barrels of Blasting Jelly, Jasmine Dragon Tea Shop, White
Lotus Hideout, Waterbending Scroll; slice 4a: Aang, the Last Airbender,
Airbending Lesson, Airbender's Reversal, Avatar's Wrath; slice 4b: Zuko, Exiled
Prince. 33 of 51 defined; 18 remain.

## Base: T1

Merged onto PR #394's head `86ebe9ccf` (qualification gate removed,
`cards/native.rs` gone, natives in set files). T1 did not create `tests/cards/`;
this branch's `tests/cards.rs` + `tests/cards/{support,tla}.rs` stand. Moving the
existing per-card `tests/rules/` files (tla_cards, tla_content, temur/elemental/
stage*/sacrifice/interaction) into `tests/cards/` stays T1's (Jack Heart's
2026-10-29 steer); re-sync if T1 lands that move.

## Next slices (one PR)

Slices 1–3 and 4a (airbend) are done; see Decisions.

4b. Remaining exact-zone grants, each needing a mechanism not built yet:
   - Yue, Solstice Revelations: cast a spell *during resolution* without paying
     its mana cost — the cast pipeline must run while a resolution is suspended
     (decide: nested cast decision vs. a granted one-shot free cast).
   - Sozin's Comet: foretell special action (pay {2}, exile face down from hand;
     exile observation must hide the face from the opponent) + later-turn grant;
     also "gains firebending 5 until end of turn" for each creature.
   - Hama: mill + chosen graveyard exile + grant whose cost is waterbend {X}
     while Hama stays controlled (source-linked, own turn).
   - Azula: opponent-chosen exiles + source-linked own-turn grant with flash
     and any-type mana.
5. Private search/choice (Aang's Journey, Elemental Teachings, Seismic Sense,
   Lost Days, Dai Li Indoctrination, Sparring Dummy), additional costs
   (Spirit Water Revival, Professor Zei, Sokka, United Front X).
6. Not yet placed in a slice: Aang's Iceberg (linked exile, source-departure race),
   North Pole Patrol (waterbend + tap cost exclusivity), Serpent of the Pass
   (conditional flash, generic reduction keeping mana value), Zuko's Conviction
   (kicker-selected graveyard return).

## Check

2026-10-28 (slice 4b Zuko): debug `cargo test --tests --no-fail-fast` passes except the pre-existing semantic-conformance pair; `tests/cards` 35 pass; clippy/fmt clean; extension rebuilt; 54 learning/content/boundary/policy-catalog pytest pass. Earlier
2026-10-28 (slice 4a): debug `cargo test --tests --no-fail-fast` passes except the pre-existing semantic-conformance pair; `tests/cards` 34 pass; clippy/fmt clean; extension rebuilt; learning/current-content/package-boundary/optional-payments/policy-catalog pytest pass (stale `schema_version == 7` assertion fixed to 8). Earlier
2026-10-28 (slice 3 + compress, realign): debug `cargo test --tests --no-fail-fast` passes except the pre-existing semantic-conformance pair (world/content-digest drift); `tests/cards` 29 pass; clippy/fmt clean; cp312 extension rebuilt on the final Rust and learning/current-content/package-boundary pytest 60 pass; world-change Python suite and Academy check deferred to gate.
