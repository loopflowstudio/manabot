# MTG-418 · Event history and contextual queries

## Outcome and cohort

Implement one native way to ask what happened, to whom, how often, and with what result. Start with Fumigate + Exsanguinate (instruction results), Cat Collector + Venom Connoisseur (different clocks), and Heartfire Immolator + Fiendish Panda (exact sources). Each pair exercises a different distinction. Then author the rest of the accepted cohort through the same rules types.

At checkout revision `0b1cc9cfef9c1a98430c825b9764a49237182696`, the [FDN ledger](../.lf/wave/coverage/research/standard-2026-11/sets/fdn.jsonl) contains **47** primary `event_history` and **34** primary `context_queries` rows. Use their Oracle IDs, obligations and secondary dependencies as the worklist. Heartfire Immolator and Fiendish Panda are additional named witnesses classified under `object_lifetime`. Among cost-primary rows, 17 cite either dependency, and the Task also names Incinerating Blast, classified under `choice_continuations`. Check the remaining cost obligations for newly expressible cards as well. A dependency tag alone does not settle delivery.

Read each definition through `managym_cards::catalog()`, its printed text and its first-printing module. A source scan for quoted names found none of the 81 primary rows in `managym/cards/src` at this revision. Group authoring by the query each card uses. Count cards by Oracle ID once, register created objects through `token_catalog()`, and update delivered rows in the existing ledger with their actual module. Keep the whole worklist through review, including a specific missing interaction or dependency for any unfinished row. The adjacent casting-permission, battlefield-form and zero-dependency composition batches keep their own implementations. Use existing permissions, copies, continuous effects and costs where available.

## Implemented slice · Fumigate and Exsanguinate

At `3c3d2cd851` (compressed in `ea10a4aac3`) the FDN ledger has two delivered context-query rows, `kld.fumigate` and
`som.exsanguinate`. Both resolve through ordinary native spell instructions.
Destruction already owns the exact objects it moved. The batch binding now
identifies destruction after normalization. An unfiltered count reads that
saved list even if a child moved one of its members again. Life changes return
the applied signed difference. The loss instruction totals its own committed
members into the resolution frame. Suspended choices retain that value in hash
v18 and the native continuation projection used for text and v11 policy input.

Directed debug checks from this slice:

- `cards kld::fumigate_counts_destroyed_creatures_even_when_their_destination_changes`:
  cast Fumigate twice. One Gray Ogre went to the graveyard, the other to exile
  under a finality counter, and indestructible Marit Lage survived. The first
  cast emitted one gain from 20 to 22. With only Marit Lage, the second emitted
  no gain. The Noxious Gearhulk optional-destruction fixture also passed.
- `cards som::exsanguinate_gains_the_life_actually_lost`: announce X=3,
  ending at 17/23 without protection and 20/20 after the opponent resolved
  Teferi's Protection. Announce X=0 and finish at 20/20.
- `--lib destruction_count_survives_a_child_returning_one_of_the_destroyed_creatures`:
  choose a destroyed Ogre to return to hand. Replaying that bound selection
  reproduced the state witness, the duplicate was rejected, and the original
  instruction still counted both destroyed creatures.
- `cards observability::life_lost_this_way_survives_a_private_choice_and_its_child_scope`:
  a constructed ability loses three life before offering discard or decline.
  The capture and v11 continuation encode that three. Decline gives 23/17,
  discard with a one-life child instruction gives 24/16, and the root stays
  at 20/17. The exhaustive effect-token and suspended Clone fixtures passed.
- `--lib life_change_result_uses_the_applied_difference_at_the_storage_limit`
  and `object_counts_describe_the_filter_and_resolution_result` passed.
- `runtime_only_programs_have_catalog_tokens` and
  `every_registered_real_card_matches_scryfall` passed. The rebuilt CPython
  3.13 arm64 wheel admitted both spell keys and exposed both batch bindings.
  `tests/verify/test_effect_execution_input.py::test_effect_execution_pairs_reach_actual_model_and_text`
  passed against that extension.

Compression (`ea10a4aac3`: `event_batch` takes its `EventBinding`, which has
no default; `commit_proposed_event` returns the member's `EventResult`) passed 15
engine unit tests and 18 card fixtures. These included both spells, the
suspended discard's encoded label and amount, replacement order, linked returns,
exact-source power, and the pinned v18 suspended Clone witness. Both tests in
`tests/verify/test_effect_execution_input.py` passed with the rebuilt CPython
3.13 arm64 release extension.

## Implemented slice · Cat Collector and Venom Connoisseur

Four of the 81 primary FDN ledger rows have definitions. Cat Collector's Food grants three life in one gain. A second Collector entering after that gain makes Food, but neither Collector makes a second Cat. The queued-gain fixture commits gains of three and two before matching. Their ordinals are one and two and they create one Cat. The turn transition resets both players' counts. Seat zero's next first gain on seat one's turn creates no Cat, and its first gain on its own next turn queues the trigger again.

Venom Connoisseur's second resolution grants team deathtouch. Countering a trigger leaves the count untouched, including after the original source returned. With an old trigger below a new one the stack shows prior counts zero and one in top-first order. Resolving both finishes at two for the original and one for the successor. Each priority Command in the card fixture replays identically in a fork and a spent Command is rejected. Two printed abilities on the same source each resolve for the first time. South Pole Voyager's returning incarnation gains life without drawing, then draws on its next resolution.

The catalog adds `fdn.cat_collector`, `snc.venom_connoisseur` and `token.cat` (the 1/1 white Cat). The Oracle snapshot check, token reference check, experience schema and exhaustive field-to-column check passed. The text/tensor fixture recovers ability index one with count two and a separate player gain count of two.

Hash v19 includes the gain ordinal and exact resolution keys. The stack retains the captured public source, and the stack and live permanent report counts for their respective abilities. The policy encodes two new v11 fact kinds and one appended stack column. The v9 and v11 pinned pilots reproduced their recorded games; the optional frontier checkpoint was unavailable.

Next slice: Fiendish Panda with Heartfire Immolator for exact-source last-known power, then Lunar Convocation for simultaneous gain and loss history. The remaining 77 primary rows, additional anchors, integrated replay/privacy and Engine Budget measurements are open. The catalog and hash
changed. Once the cohort stabilizes, integrated validation needs new current
conformance receipts, including the semantic kernel, authored-match and Academy
parity witnesses listed in `conformance/README.md`. The v17 historical witnesses
do not match the new digest.

## Native implementation

1. **Bind actual outcomes at the instruction.** In `managym/src/flow/proposed_event.rs`, extend the ordered per-member `EventResult` with the committed quantity or operation outcome needed by the instruction. Preserve original operation identity through normalization of destruction into a zone move. Distinguish a failed destruction, successful destruction with a modified destination, and an unrelated move. Read life change from the applied old/new totals. In `decision.rs::bind_event_results`, bind the parent's results only once after replacement choices for the whole batch. Keep those results across a suspended choice, and restore the parent scope after child effects. Reuse `EffectFrame`, its moved-object bindings and its payment receipt.
2. **Count at the event that establishes the fact.** `flow/event.rs::GameEvent`, `zones.rs::emit`, and the draw, combat, cast and resolution commit sites supply the facts. Extend `flow/turn.rs::PlayerTurnHistory` and the exact-source counts with only the values and relations required by the cohort. Retain the event-time ordinal or threshold crossing when a trigger will be matched later. Track occurrences and amounts separately. Attribute departures using pre-move controller and characteristics. Key damage relations by exact source and recipient. Reset turn facts for both players at the turn boundary, including extra turns. An extra combat uses the same turn. A game-scoped query retains its fact across turns and starts empty in a new game. The presentation ledger can be drained and optional policy memory can be disabled while these rules still work.
3. **Use one vocabulary at each query site.** Extend `managym/state/src/ability.rs` (`EffectValue`, `Condition`, selectors and target clauses) and `predicate.rs::CardPredicate` with composable subjects, filters and aggregates. Bind controller, exact source, trigger object/player, announced target, paid contribution or scoped result in the calling context. Share characteristic lookup and comparison across `flow/resolution.rs`, `statics.rs`, `trigger.rs` and `triggers.rs`. Use live layered characteristics or the exact departed object's last-known information (LKI) as the instruction requires. Target offers evaluate at announcement and recheck on resolution. A snapshot or a committed result retains its original value. Lunar Insight exercises distinct mana values over controlled nonlands, including zero and duplicates.
4. **Read actual payment.** `managym/state/src/payment.rs::PaymentReceipt` already owns mana units, colors, life and contributing objects. Read the receipt of the spell or ability identified by the question. Expend accumulates committed mana spent casting spells by that player. Mana floated, ability costs, rolled-back spends and uncast copies contribute zero. Unravel compares the target spell's own spend and stack mana value (including X) even after the counter instruction. Preserve manual source/pip/unit alternatives, automatic restricted-mana priority, native chooser and ordinary forced-action skipping.
5. **Project the rules state once per viewer.** Include decision-relevant facts and pending result bindings in native observations, exact forks/search and the canonical state witness (`src/state/hash.rs`, with its hash-version contract). Project observable scalars/relations through `agent/observation*.rs`, `src/python/*bindings.rs`, named v11 buffers or typed fact rows, `manabot/semantic/text_observation.py`, and Academy's native decision projection. Extend `src/cards/tokens.rs`, `agent/ability_description.rs` and `record_vocabulary.rs` for new authored expressions and record kinds. Keep saved learning-token meanings, column prefixes, v9/v10 and frozen checkpoints intact. Use public object joins for visible relations. Keep hidden identities and raw storage IDs out of both text and tensors.

### Delete when the shared reader takes over

- Replace the slot-only key in `TurnState::ability_resolutions_this_turn`, `resolve_triggered_ability`'s resolution counting path (in `flow/resolution.rs`) and the matching `CanonicalTurn` projection with the exact ability-source incarnation and ability identity. A card returning to the same slot has a new count. Preserve separate meanings for a trigger firing, an optional effect being taken and an ability resolving. South Pole Voyager's second resolution still draws exactly once.
- Consolidate the repeated power/LKI lookups in `Game::effect_value`, `power_toughness_or_lki` and `within_source_bound` as callers move. SourcePower already handles departure here. Preserve Combustion Man's existing reentry behavior and Sandbender Scavengers' graveyard-target bound. Replace the bespoke bound in `TargetSpec::GraveyardCardYouOwnWithinSourcePower` when the shared clause can express its full behavior. Preserve authored token bytes for existing definitions, including compatibility mappings if representation changes.
- Preserve `PaymentReceipt`, ordered `EventResult` and the shared event ledger. New aggregates can extend those owners directly.

## Directed interactions for integrated validation

These are expected outcomes for constructed games. Record the actual Commands, offered alternatives, fixture paths and results alongside them during implementation.

| Interaction | Observable check |
|---|---|
| Fumigate with two destructible creatures and indestructible Predator Ooze | Two creatures destroyed, two life gained in one event. With only the Ooze, gain zero. Separately use Exsanguinate for X=3 against an opponent under Teferi's Protection. Life lost and gained are both zero, versus three of each in the unprotected control. |
| Cat Collector's Food, Fiendish Panda and BLB Lunar Convocation | A Food activation gains three, grows Panda once and creates one Cat on the first gain of the controller's turn. A later gain cannot create another Cat, including when the Collector entered after the first gain. Gain and lose equal amounts in one turn and preserve both facts for Convocation. Check each player's next turn. [FDN release notes](https://magic.wizards.com/en/news/feature/foundations-release-notes) |
| Heartfire Immolator and Fiendish Panda | Sacrifice a prowess-boosted Immolator and use its last power for damage after reentry. Grow Panda before it dies. Offer every other eligible non-Bear creature card up to that last power, exclude Bears and the card just above the bound, and reject a target that left and returned after announcement. |
| Venom Connoisseur, Predator Ooze and Steel Hellkite | Counter the first Connoisseur trigger and resolve two others. The first resolution grants deathtouch to that Connoisseur and the second grants it to the team. Blink it and check the first resolution of the new incarnation. For Ooze compare a damaged creature dying later that turn with its blinked successor dying. Hellkite's X ability filters nonlands by mana value and by controller dealt combat damage by that exact Hellkite this turn. A second copy and noncombat damage provide negative controls. |
| Lunar Insight, Felling Blow and Drake Hatcher | Vary duplicate and distinct nonland mana values while holding permanent count fixed. Felling Blow reads power after adding its counter. Hatcher uses actual combat damage to a player for incubation counters. Exercise its three-counter removal and resulting Drake token. |
| High-Society Hunter, Cackling Prowler and Incinerating Blast | Hunter can sacrifice another creature or decline. It draws for each other nontoken creature dying on either side, including simultaneous deaths with Hunter. A sacrificed creature token can turn on Prowler at its controller's end step without a Hunter death draw. Blast deals damage before the optional discard. Offer every legal hand card and decline, then draw only after a paid discard. Remove its only target before resolution and verify that no discard is offered. |
| BLB Bakersbane Duo and EOE Unravel | Start at three mana spent on spells. A discounted spell crossing four triggers the Duo once even if Unravel counters it. Compare full price, a mana ability activation, a countered cast, and a Duo arriving after four. Draw with Unravel when the target cost less mana than its stack mana value. [BLB rulings](https://magic.wizards.com/en/news/feature/bloomburrow-release-notes), [EOE rulings](https://magic.wizards.com/en/news/feature/edge-of-eternities-release-notes) |
| EOE Plasma Bolt (void) and TMT Michelangelo, Game Master (disappear) | Your sacrificed Food qualifies for both. Your departing land qualifies only for disappear. An opponent's departing nonland qualifies only for void. A spell cast for its warp cost qualifies for void with no departure. Test the cast-provenance fact at the native commit boundary and record whether the actual warp cast is reachable at the integration revision. Preserve that end-to-end obligation for the casting-permission integration. |

**Replay and privacy fixture.** Fork on Blast's optional discard. Replay the same bound Command in equal roots and compare ordered committed outcomes, receipt, pending decision and witnesses. Choose decline and two distinct discards in siblings. Check the unchanged parent, exact rejections of stale/duplicate Commands and exactly one completion per branch. Permute the opponent's hidden cards while keeping the same public history. Compare fixed-viewer native/Python JSON, tensors, LLM text and Academy/replay frames. For the acting viewer, the private choices remain distinguishable. For public learnability, construct equal visible boards with different earlier gains, damage relations or mana spends and check that the legal consequences and their native facts, encoded values and text differ where relevant.

## Checks and later phases

Run focused debug checks while implementing, selecting the new card fixtures by their actual first-printing module. Existing controls from the repository root include the following commands.

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards combustion_man_uses_departed_source_power_across_a_choice_and_reentry
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards hungry_ghoul_receipt_names_the_sacrificed_creature_after_it_has_left
cargo test --manifest-path managym/Cargo.toml -p managym --lib retained_batch_counter_order_is_affected_players_choice_and_commits_once
cargo test --manifest-path managym/Cargo.toml -p managym --test search_state_contract --test canonical_replay_tests
```

Add directed cases to the per-set debug `cards` target and engine rule tests. Exercise all new row kinds in `managym/cards/tests/cards/observability.rs`. Rebuild and atomically install the matching extension per AGENTS.md, then run:

```sh
uv run --extra dev pytest -q tests/semantic/test_text_observation.py tests/verify/test_effect_execution_input.py tests/verify/test_input_v10.py tests/env/test_named_observation_layout.py tests/model/test_pinned_pilot.py
```

- **Working review** reconciles all 81 primary rows, the two additional FDN anchors, newly expressible cost rows and the cross-set witnesses. Record exact check selections, outcomes, newly authored card/token keys and any narrow reproduction. A known defect belongs with its interaction. Keep native catalog/admission independent of test status.
- **Decomposition** demonstrates that several first-printing modules author these cases with the same subjects, comparisons and result readers. Delete replaced handlers with their caller migrations and keep the directed tests. A failed semantic counterexample returns to the working phase.
- **Performance** compares reference and candidate on one runner for ordinary steps and Search-64, within the existing 10% Engine Budget with digest checks. Also measure bounded constructed boards through the documented `bench_states` layers. Use 4/8/16 Cat Collectors and Foods to vary identical listeners on one gain. Compare a mixed board of Collector, Panda, Vanguard Seraph and Exemplar of Light to exercise different listeners. Use Panda with many distinct graveyard creatures on both sides of the power bound to measure complete target enumeration. Vary creatures surviving Fumigate and creatures receiving Venom's team grant to measure batching, characteristic reads and fork cost. For payment, vary Lathril’s board from ten to twelve other untapped Elves and compare complete next choices after each tap. For Blast, vary distinct legal discards in hand. Keep input size, checkpoint, offer counts, work counters and timings together. [Stress states](../docs/development.md#stress-states)
- **Integrated validation** receives the delivered revision plus committed ledger, source and fixture paths and their observed outcomes. The separate validation work owns experimental decks and training. See the implemented slice above for the current evidence.

## Behavior review at 73516216

**Return to implementation.** Two of 81 primary FDN rows have definitions. The first missing resolution-history interaction already fails on a shipped card.

- **Reproduced in a constructed debug fixture.** In `five_color_setup(&["South Pole Voyager"])`, force Voyager onto seat 0's battlefield and advance to its main phase. Create an Ally token and resolve to priority. Record hand size, move the Voyager to exile and immediately back to the battlefield with `move_card`, then resolve its own Ally-entry trigger. Hand size went from 0 to 1. The returned Voyager should gain life on its first resolution and leave the hand at 0. The ordinary `south_pole_voyager_second_resolution_draws` control passed. The probe was temporary.
- **Cause and required fix.** `resolve_triggered_ability` indexes the turn counter by physical object slot and ability index. Both appearances therefore read the same count. Use the source incarnation captured when the trigger fired and its specific ability identity throughout resolution, the state witness and viewer projection. Keep the blink case as a regression when implementing the fix. Also exercise an old trigger still on the stack and a countered trigger. Cat Collector separately needs the gain's event-time ordinal.
- **Delivered interactions.** The Fumigate, Exsanguinate and suspended life-loss debug fixtures passed again. Destruction binds the committed moves and life loss binds the applied deltas. The optional discard fixture exposed both legal offers, retained the amount 3 in capture text and registered v11 continuation tokens, and preserved the root across both branches. The existing destruction Command fixture compares sibling hashes and checks that repeating the Command leaves the hash unchanged. Its prior passing run is recorded above.
- **Payment controls.** The Ally-only automatic spend, Animal Attendant manual spend, spend-rider receipt and Uncle's Musings mana-made-during-payment fixtures all passed. They exercise competing pooled units, a rider kept or spent for the same pip, and a newly floated color the payer can retain. The payment change here passes committed sacrifices into an unbound event batch. Expend and target-spell spend queries still need their own casting and countering witnesses.
- **Evidence boundary.** Eight existing constructed controls passed in this review. The Python model/text check has prior slice evidence. Cross-view hidden-card permutation, Academy parity and full replay receipts remain open. Finish the outstanding card semantics and their focused fixtures before decomposition, gate conformance and Engine Budget measurements.
