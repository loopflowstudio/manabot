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
   `StaticCondition::{AttackedThisTurn, LandEnteredThisTurn,
   SacrificedThisTurnAtLeast, AnyOf}`, `EffectValue::PermanentsSacrificedThisTurn`,
   `Effect::IfTargetDealtDamageThisTurn`, conditional
   `EntersWithPlusOneCounters`; the design review's effect consolidation
   (`DrawCards { count: EffectValue }`, `LoseLife`/`DealDamageToPlayer` with
   `PlayerSelector`, `DrainLife { amount: EffectValue }`, trigger player).
5. New event triggers: combat damage to player (Waterbender Ascension, Invasion Tactics),
   counters placed (Earth Kingdom General), opponent searches (Wan Shi Tong),
   cast from exile / enter from exile (Appa SG, Fire Lord Zuko), scry/surveil.
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
`TurnState::cards_drawn_this_turn`.

## Upstream (observed 2026-10-30)

T3 (#399) has not merged. Main `10eeb1f55` (Task afbd6a, Jack Heart) reshaped
w17's `CardDefinition`: one-off fields became `StaticAbility` variants in
`statics` and `AdditionalCost::{Sacrifice, Kicker, Waterbend}`; tokens moved to
`cardsets/tokens.rs`; the Python compiler and pre-w17 corpora were deleted or
archived; `TOKENS_AWAITING_PRODUCER` was added to `tests/cards/observability.rs`.
The T3/T4 sync onto it must: express T4's `ward` (Dragonfly Swarm, Benevolent
River Spirit) and additional waterbend as those variants; move T4's tokens
(Ally, Bear, Clue, Monk, Soldier, …) to `tokens.rs`; take Bear (The Earth King)
and Monk (Crescent Island Temple) off `TOKENS_AWAITING_PRODUCER`; extend the new
`static_seen`/`cost_seen` guards; re-record w17 evidence after it.

## Remaining

- Slices 5–6 (48 more cards): see the slice list above; Firebender
  Ascension's ETB, Earth Kingdom General's ETB, Invasion Tactics' ETB and
  Raven Eagle's draw clause are expressible today but wait for their other
  clauses (no half-defined cards). New turn facts go in `PlayerTurnHistory`.
- Sync onto main `10eeb1f55` (above), after or with T3's.
- Gate: Python suites bound to the w17 content digest (`authored-match-w17-v1`,
  `authored-match-parity-w17-v1`, `public-commitment-parity-w17-v1`, choice navigation, decision surface,
  learn evidence/setup, trigger order, belief canonical receipt; T3's tools
  `scratch/tools/regen_*.py`); Academy vitest for `GameBoard` turn history
  (node_modules absent in this checkout).

Compress (2026-10-30): turn-history writes go through `Game::update_turn_history`
(journals only on change); one `attack_and_settle` card-test helper.

Check (2026-10-30, after compress `3bbf7dce2`): debug `cargo test` all targets, fmt, clippy `-D warnings`, observation parity, text observation, experience protocol, current content and attachment join pass; schemas regenerated, extension rebuilt.

## Concept review (2026-10-30, proposed — awaiting Jack Heart)

Card author's action: transcribe "if …" from Oracle text. Today it has six
spellings: `StaticCondition` (statics, `IfCondition`), `IfKicked`,
`IfGraveyardAtLeast` (duplicates `StaticCondition::GraveyardAtLeast`),
`IfTargetMatches`, `IfTargetDealtDamageThisTurn`, and an intervening-if
(CR 603.4) written twice by hand: `TriggerCondition::ActiveIf` for the fire
check plus a matching `If*` effect for the resolution re-check (TLA examples
at `cardsets/tla.rs` ~5663 and ~7792; the two copies can silently disagree).

Proposal:
1. One `Condition` vocabulary (rename `StaticCondition`), asked of "you",
   plus resolution-only questions (`Kicked`, `TargetMatches(predicate)`,
   `TargetDealtDamageThisTurn`) that are invalid in a static context.
2. One `Effect::If { condition, then, otherwise }`; delete `IfKicked`,
   `IfGraveyardAtLeast`, `IfTargetMatches`, `IfTargetDealtDamageThisTurn`.
3. Intervening-if becomes `Ability::Triggered { intervening_if:
   Option<Condition>, .. }`; the engine checks it when the trigger would be
   put on the stack and again on resolution (CR 603.4). `ActiveIf` stays only
   for conditionally granted triggers ("has X as long as Y").
Consequences: tokens/observability guard entries collapse to one If and one
Condition match; `Condition::evaluate(ctx)` takes a context that has a frame
only at resolution; a resolution-only question in a static is rejected by a
definition check test. Counterexample to test: "if you do" (reflexive,
slice 6) is not a game-state condition — it stays the existing optional
payment/reflexive shape, not `Condition`.
Open: 1 (do it), 2 (scope: now in T4 vs. separate Task), 3 (where
resolution-only conditions live: one enum + check vs. a separate type).

T3 sync (2026-10-30, onto T3 `636f90bf8`): T3 replaced temporary permanent
modifiers with continuous-effect records and renamed the world w17. T4's
pump, keyword and target-player effects are `ApplyContinuous` records (new
`AffectedSpec::EachTargetPlayerControls`); `ModifyTargetPlayerCreatures`,
`ModifyUntilEot` and `GrantKeywordsToTarget` are gone. v5 widths 67 / 55 / 38
plus T3's fact rows; `semantic-kernel-w17-v1` re-recorded (14 cases / 3396
commands) and protocol schemas regenerated.
