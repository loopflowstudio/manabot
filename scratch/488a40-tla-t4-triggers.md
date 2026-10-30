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

World: T4 extends the open w18 in place, like T3 (assumption in
`scratch/questions.md`). Adding definitions changes the content digest, so
w18 digest-bound evidence is re-recorded at gate with T2's
`scratch/tools/regen_*_w18.py` and `semantic_conformance record`.

## Slices

1. **Fixed/valued effects, no new state** (done 2026-10-30): Azula, On the Hunt;
   Vindictive Warden; Firebending Student; Sun Warriors; The Spirit Oasis;
   Northern Air Temple; Southern Air Temple; Crescent Island Temple; Kyoshi
   Island Plaza; Master Pakku; Rocky Rebuke; Pirate Peddlers; Zhao, Ruthless
   Admiral. New: `Effect::LoseLife`, `Effect::DealDamageToEachOpponent`,
   `Effect::DrawCardsValue`, valued counts for counters-on-each, mill and
   basic search, `AddManaValue.until_end_of_combat`, `EffectValue::SourcePower`,
   `TriggerCondition::Sacrificed.another`.
2. **Ward against abilities** (done 2026-10-30): Dragonfly Swarm (out of
   `INCOMPLETE`), Benevolent River Spirit. `Target::StackAbility(ObjectId)`;
   `GameEvent::PermanentTargeted { by, by_controller }` emitted for spells,
   activated and triggered abilities; `counter_ability`. The Unagi's
   waterbend ward moves to slice 5 with its draw trigger.
3. **Cast and attack triggers without new state** (done 2026-10-30): Mai,
   Scornful Striker; Sokka, Lateral Strategist; The Earth King; Bitter Work;
   How to Start a Riot; Bumi Bash. New: `TriggerCondition::PlayerCastsSpell`
   (caster as context, only for abilities that read it),
   `TriggerCondition::AttacksWithAnother`, `Effect::TriggeringPlayerLosesLife`,
   `Effect::ModifyTargetPlayerCreatures`, `EffectValue::AttackingMatching`,
   `CardPredicate::nonbasic`.
4. Turn history: attacked this turn (raid: Fire Nation Raider/Engineer, Cruel
   Administrator), sacrificed count, damage dealt this turn (Sold Out), land
   entered this turn. Needs observed state (player row columns).
5. New event triggers: combat damage to player (Waterbender Ascension, Invasion Tactics),
   counters placed (Earth Kingdom General), opponent searches (Wan Shi Tong),
   cast from exile / enter from exile (Appa SG, Fire Lord Zuko), scry/surveil.
6. Reflexive and delayed triggers, placement-time modes, copies/retargeting,
   trigger multiplicity (Katara the Fearless), LKI counter transfer, control
   change, Destined Confrontation's simultaneous choice.

## Delete — do not maintain

Nothing replaced yet.

## Remaining

- Slice 4 (next): turn history as observed state — per player "attacked
  this turn", permanents sacrificed this turn, a land entered this turn, and
  per permanent "dealt damage this turn" — through `Observation`, native and
  Python v5 encoders (widen v5 player/permanent rows in place, update
  CLAUDE.md widths and WORLDS.md), learning tokens, LLM text and Academy notes.
  Cards: Fire Nation Raider, Fire Nation Engineer, Cruel Administrator, Sold
  Out, Obsessive Pursuit, Earth Rumble Wrestlers.
- Slices 5–6 (48 more cards): see the slice list above; Firebender
  Ascension's ETB, Earth Kingdom General's ETB, Invasion Tactics' ETB and
  Raven Eagle's draw clause are expressible today but wait for their other
  clauses (no half-defined cards).
- Gate: re-record w18 evidence after the last card (`semantic-kernel-w18-v1`
  is current). Python failures after slice 1 are all digest-bound:
  authored-match receipt and parity, public-commitment parity, choice
  navigation, decision surface, learn evidence/setup, trigger order, belief
  canonical receipt (T3's tools: `scratch/tools/regen_*_w18.py`).

Check (2026-10-30, after slice 3): debug `cargo test` all targets pass
(cards 405; `semantic-kernel-w18-v1` re-recorded, 14 cases / 3396 commands —
ward now answers abilities; heavy benchmark fixture 505 → 509 events), fmt and
clippy `-D warnings` clean, extension rebuilt. `tests/academy tests/belief`
(slice 1): 330 pass, 23 digest-bound fail (gate).
