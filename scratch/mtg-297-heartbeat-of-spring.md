# MTG-297: Heartbeat of Spring, Animal Attendant, Toph, Uncle's Musings

Design: `scratch/tle-production-design.md` with the review decisions in
`scratch/heartbeat-design.md`.

## Scope

Steer 41eb2776 (2026-11-12): "Do the production half now. Build spending
(pool classes, receipts, Animal Attendant, Toph, Uncle's Musings) on MTG-242's
ManaRestriction/SpendingContext and proposed restricted-mana fact row; don't
rewrite pool storage in parallel. If spending must wait for MTG-242 to reach
main, finish production and stop there."

MTG-242 (PR #496) merged 2026-11-12T21:37Z; main is merged into this branch
(824753626) and the spending half is built on its types.

## Built: production

- `TriggeredManaAbility { who: Actor, predicate, output }`,
  `TriggeredManaOutput::{Fixed(Mana), OneProducedType}` (`state/card.rs`).
  Badgermole Cub is `who: You, Fixed(G)`; Heartbeat of Spring
  (`cards/src/chk.rs`) is `who: AnyPlayer`, land, `OneProducedType`.
- Tapping no longer fires bonuses. `flow/mana.rs`: tapping records
  `TappedForMana`; trigger sources are read from the board when the output is
  produced (`production_bonuses`, CR 106.12a) and resolved after it
  (`resolve_mana_bonuses`); a choice among produced types suspends as a
  `ManaChoice` whose `ManaProduction` names its source and the tapped
  permanent. Deleted `tap_mana_bonus`, `tap_mana_bonus_from`,
  `tap_bonus_sources`, `fire_triggered_mana_abilities`.
- Planner folds every bonus combination into the output alternatives.

## Built: spending

- `state/mana.rs`: the pool's class key is `ManaClass { restriction, effect }`
  (was the restriction alone); `Mana.classes` replaces `Mana.restricted`.
  `SpendEffect::CreatureEntersWithPlusOneCounter { without_subtype }` is
  Animal Attendant's rider. Combat and regular pools stay distinct.
- `Mana::pay` and `Game::spend_mana` return `ManaSpent { amounts,
  entry_counters }`. Automatic order within a color: restricted classes and
  classes whose effect applies, then ordinary mana, then classes whose effect
  would be wasted.
- The receipt travels `commit_payment` → `FrameCompletion::Paid.spent` →
  `SpellOnStack.spent` → the resolution frame (`EffectFrame.spent`) → the
  entering permanent (`Permanent.cast_spent`) → its own entering trigger
  (`PendingTrigger.spent`, `TriggeredAbilityOnStack.spent`). A permanent
  spell's `entry_counters` become +1/+1 counters in its `ZoneEntry`.
- `EffectValue::ManaSpentToCast` (Toph) and `ColorsSpentToCast` (converge).
- Cards in `cards/src/tle.rs`: Animal Attendant, Toph, Greatest Earthbender,
  Uncle's Musings; Scryfall fixture entries added for these and Heartbeat.
  `scripts/refresh_card_fixture.py` currently drops the double-faced cards
  already in the fixture, so only the four new entries were added.
- Observation, with no new buffer, column or fact kind:
  - `PlayerData.mana` counts unrestricted class mana;
    `PlayerData.spend_effect_mana` (description, amounts, token program) is
    an `EffectProgram` (kind 11) on the player under the id `PooledMana`,
    native and Python in parity; the text observation names the rider.
  - The receipt is an execution occurrence (`mana spent`) of the stack frame
    or resolution frame, only when a rule reads it
    (`effect_execution::receipt_matters`: the definition mentions the
    spent-mana values, or a spend effect applied).
  - Vocabulary: `PooledMana`, `ManaSpent`, `amounts`, `entry_counters`,
    `mana spent` registered in `record_vocabulary.rs`/`EXECUTION_TEXT`.

## Remaining

- **Spending choices.** The design replaces automatic first-remainder payment
  with a choice whenever receipts or remaining pools differ. Not built:
  payment is still automatic (`Mana::payment_remainders`, the planner's fixed
  sources). Consequences today: converge counts whatever colors automatic
  tapping and the colorless-first generic order spend, and a player cannot
  hold Attendant mana back from a qualifying creature. This is the design's
  second spending deletion ("unconditional first-remainder selection") and a
  new decision surface (action offers, Academy, replay); it is the next
  implement direction or its own Task.
- Witnesses still to write: a countered then reanimated creature gets no
  counter or X; mana spent on a filter's own cost is not in the outer
  receipt; combat-pool Attendant mana.
- Academy's player view shows Attendant mana as ordinary mana without its
  rider (`experience.rs` has no spend-effect field).

## For the PR description

- CR over design: bonuses trigger when the mana ability produces mana
  (CR 106.12a), sources read at production; the design's proposed v10
  `mana_choices[A,4]` block and fact kinds after 14 were not needed.
- Spending choices are not in this change (above).

## Checks

- Production half (before the merge): cards 694, rules_tests 169, engine lib
  197 passed in debug; corpus first 1,000 decks 998 admit (the two failures
  list Lita).
- After the merge and spending (debug): cards 707, engine lib 197,
  rules_tests 170, match_state_hash 11, search_state_contract 6,
  canonical_replay 6, experience_protocol 5, conformance 2, content_pack 3,
  state 15 passed. Python: academy parity/authority/fixture, semantic text and
  mana input, env observation and layout, belief state, authored matchup
  pass after re-recording w17 evidence (authored-match receipt 66d51605…,
  terminal state db1a0bbb…, belief v3 receipts).
