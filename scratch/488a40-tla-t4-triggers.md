# T4 — triggers, history and stack behavior (Task 488a40)

Brief: the Task directive (75 cards, obligations in `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T4"`). Stacked on T3 (6875f0, PR #399, which merged T2's #398).
Jack Heart's 2026-10-30 launch steer: cards are engine `CardDefinition` values in
their first-printing set file; new behavior is a named rules-type variant with a
directed test in `tests/cards/tla.rs`; every new mechanic reaches the policy
tensor, learning tokens and LLM text, with the exhaustive guards in
`tests/cards/observability.rs`; a card is supported when defined unless listed
in `tla::INCOMPLETE`. Every merge needs Jack Heart's review.

Oracle text: Scryfall `/cards/collection` for the 75 names (set `tla`); not
kept in scratch (budget). Dragonfly Swarm is already defined, in `INCOMPLETE`.

## Approach

Slices of cards grouped by the engine variant they need, smallest new state
first. Each slice: variants (with token/observability guard entries), cards,
one directed test per obligation. Cards outside finished slices stay
undefined (so unsupported), never half-defined.

World: T4 extends the open world in place, like T3 (assumption in
`scratch/questions.md`). T3's main sync renamed it w17, so T4's evidence is
w17 too. Adding definitions changes the content digest, so w17 digest-bound
evidence is re-recorded at gate with `scratch/tools/regen_*.py` and
`semantic_conformance record`.

## Slices

1. **Fixed/valued effects, no new state** (done 2026-10-30): Azula, On the Hunt;
   Vindictive Warden; Firebending Student; Sun Warriors; The Spirit Oasis;
   Northern Air Temple; Southern Air Temple; Crescent Island Temple; Kyoshi
   Island Plaza; Master Pakku; Rocky Rebuke; Pirate Peddlers; Zhao, Ruthless
   Admiral. New (names as consolidated in slice 4): `EffectValue` counts for
   counters-on-each, mill and basic search, `AddManaValue.until_end_of_combat`,
   `EffectValue::SourcePower`, `TriggerCondition::Sacrificed.another`.
2. **Ward against abilities** (done 2026-10-30): Dragonfly Swarm (out of
   `INCOMPLETE`), Benevolent River Spirit. `Target::StackAbility(ObjectId)`;
   `GameEvent::PermanentTargeted { by, by_controller }` emitted for spells,
   activated and triggered abilities; `counter_ability`. The Unagi's
   waterbend ward moves to slice 5 with its draw trigger.
3. **Cast and attack triggers without new state** (done 2026-10-30): Mai,
   Scornful Striker; Sokka, Lateral Strategist; The Earth King; Bitter Work;
   How to Start a Riot; Bumi Bash. New: `TriggerCondition::PlayerCastsSpell`
   (caster as the trigger's player), `TriggerCondition::AttacksWithAnother`,
   `AffectedSpec::EachTargetPlayerControls`, `EffectValue::AttackingMatching`,
   `CardPredicate::nonbasic`.
4. **Turn history** (done 2026-10-30): Fire Nation Raider, Fire Nation
   Engineer, Cruel Administrator, Sold Out, Obsessive Pursuit, Earth Rumble
   Wrestlers. New: `PlayerTurnHistory` (observed in the player row),
   `Permanent.dealt_damage_this_turn` (permanent column, LKI),
   `Condition::{AttackedThisTurn, LandEnteredThisTurn,
   SacrificedThisTurnAtLeast, AnyOf}`, `EffectValue::PermanentsSacrificedThisTurn`,
   `Condition::TargetDealtDamageThisTurn` (names after the concept review), conditional
   `EntersWithPlusOneCounters`; the design review's effect consolidation
   (`DrawCards { count: EffectValue }`, `LoseLife`/`DealDamageToPlayer` with
   `PlayerSelector`, `DrainLife { amount: EffectValue }`, trigger player).
5. **New event triggers.** Done (2026-10-30): Waterbender Ascension,
   Invasion Tactics, Appa, Steadfast Guardian, Fire Lord Zuko, Wan Shi Tong
   (`CombatDamageToPlayers`, `CreatureDealsCombatDamageToPlayer` ×creatures,
   `OneOrMoreDeal…`, `TriggerCondition::FromZone { zone, condition }`,
   `OpponentSearchesLibrary`/`LibrarySearched`, `Permanent.cast_x` → the
   trigger's X (CR 107.3m), `EffectValue::Half`, `TargetRequirement::ANY_NUMBER`,
   `CounterKind::Quest`, `Condition::SourceCountersAtLeast`).
   Left: Earth Kingdom General and Planetarium share one missing mechanic,
   "you may [act]. Do this only once each turn": the trigger fires every
   time, the choice is offered only while unused this turn, and only taking
   it spends the use (declining keeps it). Proposed shape: an optional
   choice keyed by the exact source object and ability index in
   `TurnState` (the frame learns its ability index), with EKG's
   `CountersPlaced { permanent, kind, count }` event ("that much" via the
   trigger's number) and Planetarium's `Scried` event plus a
   `FreeCastOffer::LibraryTop`. `OnceEachTurn` (fires once) is the wrong
   shape for both.
6. Reflexive and delayed triggers, placement-time modes, copies/retargeting,
   trigger multiplicity (Katara the Fearless), LKI counter transfer, control
   change, Destined Confrontation's simultaneous choice.

## Design review (Jack Heart, 2026-10-30, accepted)

Applied in slice 4 (full text: the Task's 2026-10-30 design-review comment):
one effect vocabulary with `PlayerSelector`, `Permanent.dealt_damage_this_turn`,
one `PlayerTurnHistory`, `EntersWithPlusOneCounters { count, condition }`.

## Delete — do not maintain

Done in slice 4 (no parallel path remains): `DrawCardsValue`, `DrainLifeValue`,
`TriggeringPlayerLosesLife`, `DealDamageToController`,
`DealDamageToEachOpponent`, the trigger effect scan for the caster,
`TurnState::cards_drawn_this_turn`. Done with the concept review:
`StaticCondition`, `IfCondition`, `IfKicked`, `IfGraveyardAtLeast`,
`IfTargetMatches`, `IfTargetDealtDamageThisTurn`, and the ActiveIf-plus-If
intervening-if pairs.

## Upstream (2026-10-30)

Synced: main `10eeb1f55` is an ancestor (through T3's sync, merge
`98ec26e5b`), and T3's head `4c4dd0217` is merged (`a315216b2`). T4's ward is
`StaticAbility::Ward`, required waterbend `AdditionalCost::RequiredWaterbend`,
T4's tokens are in `tokens.rs`, Bear and Monk are off
`TOKENS_AWAITING_PRODUCER`. T3 (#399) is still open; each T3 head is merged in
before T4's own gate.

## Remaining

- 43 of 75 cards (32 defined). Slice 5: Earth Kingdom General, Planetarium
  of Wan Shi Tong (above), and The Unagi of Kyoshi Island (deferred from
  slice 2): ward—waterbend {4} is not a `Ward(ManaCost)`, and "an opponent
  draws their second card" needs an opponent form of T1's
  `YouDrawNthCardThisTurn` (Tiger-Seal).
  Slice 6: the other 40. Firebender
  Ascension's ETB, Earth Kingdom General's ETB, Invasion Tactics' ETB and
  Raven Eagle's draw clause are expressible today but wait for their other
  clauses (no half-defined cards). New turn facts go in `PlayerTurnHistory`.
- Gate: Python suites bound to the w17 content digest (`authored-match-w17-v1`,
  `authored-match-parity-w17-v1`, `public-commitment-parity-w17-v1`, choice navigation, decision surface,
  learn evidence/setup, trigger order, belief canonical receipt; T3's tools
  `scratch/tools/regen_*.py`); Academy vitest for `GameBoard` turn history
  (node_modules absent in this checkout).

Compress (2026-10-30): turn-history writes go through
`Game::update_turn_history`; `CombatDamageToPlayers { creatures }` (the
defender is the non-active player); `trigger_multiplicity` counts connecting
creatures; the leaving spell is removed from the stack once.

## Concept review (Jack Heart, 2026-10-30, accepted 1a 2a 3a — done)

One `Condition` (resolution-only `Kicked`, `TargetMatches`,
`TargetDealtDamageThisTurn`, rejected outside `Effect::If` by
`tests/cards/definitions.rs`); one `Effect::If { condition, then, otherwise }`;
`Ability::Triggered { intervening_if }` (CR 603.4). `ActiveIf` stays for Fire
Nation Cadets and Tolls of War. "If you do" stays the optional-payment shape.
T3-sync details: `scratch/questions.md` (T4 and T3-sync sections).

Check (2026-10-30, merged tree `a315216b2`): debug `cargo test --no-fail-fast` all targets, 800 passed, 0 failed, after re-recording `semantic-kernel-w17-v1` (14 cases / 3396 commands; state hashes only, from `CombatDamageToPlayers { creatures }`). Other w17 digest-bound Python evidence stays for gate.
