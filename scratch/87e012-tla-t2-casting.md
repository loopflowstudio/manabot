# T2 — TLA casting, costs, zone permissions and private zones (Task 87e012)

Stacked on T1 (892c47, PR #394 at `be2b7586b`). Task directive lists the 51 cards
and obligations; allocation in `docs/rules/tla/allocation.jsonl` (`planned_pr: T2`).
Every merge needs Jack Heart's review. Card support = defined in the set file
(Jack Heart, 2026-10-29); T1 is removing the qualification gate in its own
worktree, so new T2 cards read "unsupported" in this base until T1 lands.

## Decisions

- 2026-10-29, Jack Heart: card tests live in `managym/tests/cards/<set>.rs`
  (one `tests/cards.rs` entry). T2 tests go in `tests/cards/tla.rs`; moving the
  existing per-card rules tests is T1's (reconcile when syncing).
- Activated abilities carry an effect sequence (`ActivatedAbilityDefinition.effects`),
  not one effect: "draw, then discard" and later multi-clause activations.
- Printed flashback is `Cost.flashback` on the spell program (like kicker),
  lowered to `CardDefinition.flashback`. When Iroh's grant also applies, the
  grant's cost is used (never costlier in TLA); see questions.md.
- Earthbend takes a `Count` (printed integer or `Quantity`); runtime
  `EffectValue::Fixed` carries printed counts, so fixed earthbend IR and its
  conformance trace are unchanged.
- Learning schema 8 grows append-only, as T1 did in `1e0d3c998`: subtypes
  Plant/Rogue, field `flashback`. No new opcodes yet (schema 8 caps at 65);
  the first new opcode needs schema 9 — see questions.md.

- 2026-10-29: activations name an `ActivationSource` (permanent or card) in
  `Payment::Activation` and `PendingChoice::ActivatedTarget`. Hand/graveyard
  abilities are offered as `Action::ActivateCardAbility { card, ability_index }`
  with action type `PriorityActivateAbility` and the card as focus (as for
  casting), so the observation/action shape is unchanged. `ability_index`
  indexes the card's printed `activated_abilities` (programs compile in key
  order). `ActivatedAbilityDefinition.zone`/`discard_source` are omitted at
  their defaults, so existing definition digests hold.
- Landcycling reuses opcode 24 (`search_basic_land`) with `destination: hand`:
  any card of the subtype, revealed into hand (known-hand count), then shuffle.
  "Return this card from your graveyard" lowers `return_to_hand` with target
  `source` to `ReturnSourceToHand` (exact graveyard object only). Team Avatar
  uses trigger event `attacks_alone` and the builtin role
  `triggering_permanent` with `buff_target_value` (toughness now a `Count`).
  No new opcode, so no schema 9: schema 8 grows append-only (subtypes
  Kangaroo/Mongoose/Rabbit/Spider, field `discard_source`, zone `your_hand`,
  trigger event `attacks_alone`, role `triggering_permanent`).
- 2026-10-29: exhaust is `Cost.exhaust` → `ActivatedAbilityDefinition.exhaust`;
  the permanent records activated exhaust indices (`Permanent.exhausted`,
  omitted when empty), marked when the ability reaches the stack; a new
  object starts fresh. Predicate `with_counter` ("with a +1/+1 counter on
  it") matches permanents only. Schema 8 appends field `exhaust`, predicate
  `with_counter`, subtype Monkey; authoring gained `Event::CombatBegins`.

## Done (uncommitted in this worktree)

T1 defined 7 (Badgermole Cub, Dai Li Agents, Haru, Toph, Hermitic Herbalist,
Iroh, Grand Lotus; witnesses in `tests/rules/`). T2 defined 14, each witnessed
in `tests/cards/tla.rs` with a Scryfall fixture entry: Earthbending Lesson,
Sandbenders' Storm, Earth Village Ruffians, Geyser Leaper, Foggy Swamp
Vinebender, Fire Nation Attacks, Rockalanche; slice 1: Canyon Crawler,
Mongoose Lizard, Saber-Tooth Moose-Lion, Giant Koi, Rabaroo Troop, Team Avatar,
Merchant of Many Hats; slice 2: Hog-Monkey, Rough Rhino Cavalry, Rebellious
Captives. 24 of 51 defined; 27 remain.

## Syncing onto T1

As of 2026-10-28 PR #394 is still at `be2b7586b`; T1's worktree holds
uncommitted edits that delete `cards/native.rs`, `inventory.json` and
`qualifications.json`, and it has not yet created `tests/cards/`.
On sync: carry T2's `effects: vec![..]` edit in `native.rs` to wherever T1
moves the legacy slots; keep this branch's `tests/cards.rs` layout, or adopt
T1's if it lands differently, and leave moving the existing per-card
`tests/rules/` files (tla_cards, tla_content, temur/elemental/stage*) to T1
unless T1 lands without it.

## Next slices (one PR)

1. Done: hand/graveyard activation (see Decisions).
2. Done: exhaust.
3. Restricted/filtered mana (Barrels, Jasmine Dragon Tea Shop, White Lotus Hideout):
   generalize Herbalist's Lesson-only restriction to Ally/Shrine spell-or-ability
   restrictions; activation cost reduction (Waterbending Scroll). The Lesson
   restriction is hard-wired as `Mana.lesson` slots, `ManaCost.spend_on_lesson`
   and the planner in `flow/mana.rs` (~L250, ~L1050); generalizing means keying
   restricted slots by a spend predicate (spell subtype, or ability of an Ally
   source), which reaches the pool model and its observation. Barrels also needs
   "activate only once each turn" and a sacrifice-source damage ability.
4. Airbend (exile + cast-for-{2} permission): Aang, Airbending Lesson,
   Airbender's Reversal, Avatar's Wrath; exact-zone grants (Azula, Zuko, Hama,
   Yue, Solstice Revelations, Sozin's Comet foretell).
5. Private search/choice (Aang's Journey, Elemental Teachings, Seismic Sense,
   Lost Days, Dai Li Indoctrination, Sparring Dummy), additional costs
   (Spirit Water Revival, Professor Zei, Sokka, United Front X).
6. Not yet placed in a slice: Aang's Iceberg (linked exile, source-departure race),
   North Pole Patrol (waterbend + tap cost exclusivity), Serpent of the Pass
   (conditional flash, generic reduction keeping mana value), Zuko's Conviction
   (kicker-selected graveyard return).

## Check

2026-10-29 (after slice 2, re-run after compress): debug `cargo test --tests --no-fail-fast` passes except the pre-existing semantic-conformance pair; clippy clean; `tests/cards` 23 pass; cp312 extension rebuilt and binds every new card under schema 8; learning/current-content/package-boundary pytest pass (`test_optional_payments` keeps its pre-existing schema-7 assertion); world-change Python suite and Academy check deferred to gate.
