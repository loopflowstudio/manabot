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
5. **New event triggers** (done 2026-10-30): Waterbender Ascension,
   Invasion Tactics, Appa, Steadfast Guardian, Fire Lord Zuko, Wan Shi Tong
   (`CombatDamageToPlayers`, `CreatureDealsCombatDamageToPlayer` ×creatures,
   `OneOrMoreDeal…`, `TriggerCondition::FromZone`, `OpponentSearchesLibrary`,
   `Permanent.cast_x`, `EffectValue::Half`, `TargetRequirement::ANY_NUMBER`,
   `CounterKind::Quest`); The Unagi (`Ward(PaymentCost)`, so ward may be a
   waterbend; `CounterUnlessPays { cost: PaymentCost }`;
   `OpponentDrawsNthCardThisTurn`); Earth Kingdom General (`Game::put_counters`
   → `GameEvent::CountersPut`, `YouPutCountersOnCreature`, trigger X = count;
   `Effect::MayOnceEachTurn`, keyed by exact source and `EffectFrame.ability`
   in the turn's once-each-turn uses, spent only on accepting); Planetarium
   (`GameEvent::Scried`/`YouScry`, `Effect::CastLibraryTopFree` with
   `FreeCastOffer::LibraryTop { card }`, spent only by casting).
6. **The rest, in commit-sized groups.** Done (2026-10-30): Beetle-Headed
   Merchants, Vengeful Villagers (`Effect::MaySacrifice`, optional
   `Decision::Sacrifice`); Katara, Bending Prodigy, Guru Pathik, Long Feng
   (source questions `Condition::SourceTapped`/`SourceCountersAtLeast`,
   allowed in intervening-ifs via `Game::check_intervening_if`); Ty Lee,
   Artful Acrobat (`Effect::WhenYouDo` reflexive trigger,
   `Modification::CantBlock`); Suki, Courageous Rescuer
   (`TriggerCondition::LeavesTheBattlefield`); Momo, Appa, Loyal Sky Bison
   (`Effect::TriggerModes`, mode chosen at placement, stage `trigger_mode`);
   Beifong's Bounty Hunters, Hei Bai (`ObjectLki.power`/`counters`,
   `EffectValue::TriggeringPower`, `Effect::PutSourceCountersOnTarget`);
   Airbender and Earthbender Ascension (`Effect::ExileAndReturn`).
   Left: delayed triggers (Fatal Fissure, Fire Navy Trebuchet, Ruinous
   Waterbending, Jeong Jeong), copies/retargeting (Firebender Ascension,
   Jeong Jeong, Redirect Lightning), control change (Jet's Brainwashing, Iroh,
   Zuko, Conflicted), up-to-X and unchosen modes (Bumi, Zuko), trigger
   multiplicity (Katara, the Fearless), a one-or-more-flying-attackers
   trigger and a reflexive trigger only when the discarded card is nonland
   (Teo, Spirited Glider), granted prowess (Sokka, Tenacious
   Tactician), any-graveyard targets (Raven Eagle, Boiling Rock Rioter),
   a spell's reflexive trigger (Earth Rumble), "if it had counters" (Buzzard-Wasp
   Colony), finality counters (Wolfbat), LKI-bounded targets (Sandbender
   Scavengers), simultaneous choices (Destined Confrontation), opponent's
   choice (Combustion Man), opponent's search (Price of Freedom), discard as
   a trigger's option and a spell-matching rider (Toph), untap restriction
   while controlled (Ty Lee, Chi Blocker), hexproof/indestructible loss
   (The Fire Nation Drill), nonbasic replacement and type overwrite (Zhao,
   the Moon Slayer), Aura death LKI (Avatar Destiny).

## Design review (Jack Heart, 2026-10-30, accepted)

Applied in slice 4 (full text: the Task's 2026-10-30 design-review comment):
one effect vocabulary with `PlayerSelector`, `Permanent.dealt_damage_this_turn`,
one `PlayerTurnHistory`, `EntersWithPlusOneCounters { count, condition }`.

Follow-up (Jack Heart, 2026-10-30, applied): `Game::counter_ability` emits
`GameEvent::AbilityCountered` (EventType 13, policy history `Countered`,
decision context); `Effect::CounterSpell` counters spells only, and a declined
`CounterUnlessPays` counters the spell or ability it names
(`PaymentProgress.counters_on_decline`). `CardPredicate.nonbasic` stays.

## Delete — do not maintain

Done in slice 4 (no parallel path remains): `DrawCardsValue`, `DrainLifeValue`,
`TriggeringPlayerLosesLife`, `DealDamageToController`,
`DealDamageToEachOpponent`, the trigger effect scan for the caster,
`TurnState::cards_drawn_this_turn`. Done with the concept review:
`StaticCondition`, `IfCondition`, `IfKicked`, `IfGraveyardAtLeast`,
`IfTargetMatches`, `IfTargetDealtDamageThisTurn`, and the ActiveIf-plus-If
intervening-if pairs. Done in slices 5–6: `change_plus_one_counters_on_object`
(placements go through `Game::put_counters`), `CounterSpell`'s ability arm and
ward's `CounterSpell` decline branch.

## Upstream (2026-10-30)

Synced: origin/main `6f48f5ca0` is an ancestor (T3's merge `09e6babe5`,
after T3's sync `98ec26e5b`), and T3's head `4c4dd0217` is merged
(`a315216b2`); both rechecked at `2f73cd16f`. T4's ward is
`StaticAbility::Ward`, required waterbend `AdditionalCost::RequiredWaterbend`,
T4's tokens are in `tokens.rs`, Bear and Monk are off
`TOKENS_AWAITING_PRODUCER`. T3 (#399) is still open; each T3 head is merged in
before T4's own gate.

## Remaining

- 10 of 75 cards (65 defined): Boiling Rock Rioter; Firebender Ascension;
  Jeong Jeong, the Deserter; Jet's Brainwashing; Redirect Lightning; Zhao, the
  Moon Slayer; Avatar Destiny; Bumi, King of Three Trials; Iroh, Tea Master;
  Zuko, Conflicted. Slice 7 (2026-10-30) defined the other 17 in four
  commits (`fc3e5ae74`..`5d870e814`): granted triggers and extra occurrences,
  graveyard triggers and finality, general delayed triggers
  (`Effect::CreateDelayedTrigger`; earthbend's return is one), `ForPlayer`,
  keep-creatures, while-you-control durations, keyword loss, "this way"
  discards, source-power-bounded targets.
- Gate: Python suites bound to the w17 content digest (`authored-match-w17-v1`,
  `authored-match-parity-w17-v1`, `public-commitment-parity-w17-v1`, choice
  navigation, decision surface, learn evidence/setup, trigger order, belief
  canonical receipt; T3's tools `scratch/tools/regen_*.py`); Academy vitest
  for `GameBoard` turn history, `permanent-notes` (`CantBlock`) and
  `decision-labels` (`trigger_mode`) (node_modules absent in this checkout).

Compress (2026-10-30): turn-history writes go through
`Game::update_turn_history`; `CombatDamageToPlayers { creatures }` (the
defender is the non-active player); `trigger_multiplicity` counts connecting
creatures; the leaving spell is removed from the stack once. Second compress: `EffectFrame.ability` is set without
the resolution counter's `usize::MAX` sentinel; `OnceEachTurnKey` names the
once-each-turn key and `Game::spend_once_each_turn` is its only writer.

## Concept review (Jack Heart, 2026-10-30, accepted 1a 2a 3a — done)

One `Condition` (resolution-only `Kicked`, `TargetMatches`,
`TargetDealtDamageThisTurn`, rejected outside `Effect::If` by
`tests/cards/definitions.rs`); one `Effect::If { condition, then, otherwise }`;
`Ability::Triggered { intervening_if }` (CR 603.4). `ActiveIf` stays for Fire
Nation Cadets and Tolls of War. "If you do" stays the optional-payment shape.
T3-sync details: `scratch/questions.md` (T4 and T3-sync sections).

Check (2026-10-30, `2f73cd16f`): debug `cargo test --no-fail-fast` all targets, 820 passed, 0 failed; clippy `-D warnings` and fmt clean; `semantic-kernel-w17-v1` re-recorded at `888f5d359` (14 cases / 3396 commands), Scryfall fixture refreshed for the 16 new cards. Other w17 digest-bound Python evidence stays for gate.

Approval (Jack Heart, 2026-10-30): T4 approved in full — all 75 cards with the
review decisions above — publish and ship without another review. When the
running Graph stops (e.g. at `demo`), the operator runs `lf ship 488a40`.
