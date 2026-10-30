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
   waterbend ward moves to slice 4 with its draw trigger.
3. Turn history: attacked this turn (raid: Fire Nation Raider/Engineer, Cruel
   Administrator), sacrificed count, damage dealt this turn (Sold Out), land
   entered this turn. Needs observed state (player row columns).
4. New event triggers: any player casts (Mai), attacks with another (Sokka),
   combat damage to player (Waterbender Ascension, Invasion Tactics),
   counters placed (Earth Kingdom General), opponent searches (Wan Shi Tong),
   cast from exile / enter from exile (Appa SG, Fire Lord Zuko), scry/surveil.
5. Reflexive and delayed triggers, placement-time modes, copies/retargeting,
   trigger multiplicity (Katara the Fearless), LKI counter transfer, control
   change, Destined Confrontation's simultaneous choice.

## Delete — do not maintain

Nothing replaced yet.

## Remaining

- Slices 3–5 (60 cards).
- Gate: re-record w18 evidence after the last card (`semantic-kernel-w18-v1`
  is current at slice 1). Python failures after slice 1 are all digest-bound:
  authored-match receipt and parity, public-commitment parity, choice
  navigation, decision surface, learn evidence/setup, trigger order, belief
  canonical receipt (T3's tools: `scratch/tools/regen_*_w18.py`).

Check (slice 2): debug `cargo test` all targets pass after re-recording
`semantic-kernel-w18-v1` (14 cases, now 3396 commands: ward answers
abilities) and the heavy benchmark fixture's event count (505 → 509);
fmt and clippy `-D warnings` clean. Slice 1: `tests/academy tests/belief`
330 pass, 23 digest-bound fail (gate).
