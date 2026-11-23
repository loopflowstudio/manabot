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

## Implemented slice · exact source and legal target

The catalog adds `fdn.fiendish_panda` and `m21.heartfire_immolator`. Panda and Sandbender use the same graveyard mana-value bound. A target requirement now uses the spell or ability's captured source at announcement, retargeting and resolution. The spell and activation menus also read the requirement's “another” restriction. Immolator can target itself at announcement.

- `agent/env/snapshot.rs::panda_snapshot_replays_source_and_target_reentry` restores the pending target choice. One gain of three grew Panda to power four. Only Gray Ogre and Hill Giant were offered, with a second Panda and Serra Angel also in the graveyard. The same Hill Giant Command replayed twice per branch. After Panda returned at power three, Hill Giant entered. After Hill Giant left the graveyard and returned, it stayed there. Both viewer projections, complete decision frames and witnesses matched across repeated restores, and the root stayed intact.
- `flow/target_authority_tests.rs::graveyard_target_rechecks_live_source_power_for_retarget_and_resolution` selected a three-mana creature at power three. Removing the source's counter removed the alternative retarget and left the announced creature in its graveyard.
- `cards/tests/cards/m21.rs::heartfire_immolator_deals_its_power_at_sacrifice_after_returning` cast Lightning Bolt for prowess, activated Immolator targeting Craw Wurm, and returned Immolator before resolution. Craw Wurm took three damage. Sandbender's existing sacrifice-and-reanimation control also passed.

At `73f608af`, retargeting reads the announced clause for each slot directly. Sibling distinctness and combined mana-value bounds stay scoped to that clause. Panda and Immolator close the two additional object-lifetime witnesses. Four of 81 primary FDN rows are delivered.

### Behavior review and pending power

Four directed debug fixtures passed again: Panda snapshot and Command replay, live source-power rechecks during retargeting and resolution, Immolator damage after source reentry, and Sandbender reanimation. Panda’s two viewer frames agreed across restores. This checks deterministic projection; the hidden-hand permutation and stale Command checks remain integrated obligations. The pair uses existing cost and token expressions.

The review at `11f107c0` found “Its pending activation still omits that public value.” The correction at `f9a251a9` reads power through `permanent_power_or_lki` for a pending activation with a direct source-power damage instruction. The M21 fixture checks the value after source reentry. In `tests/verify/test_immolator_source_power.py`, base and prowess activations showed two and three to both viewers in the native observation, JSON round trip, v11 tensor and Academy stack label. The acting viewer’s history-free text showed the same value. Craw Wurm took two and three damage respectively. The fixture at `e2e8318d` leaves one untapped Mountain for either activation.

The observation matches the damage expression to decide when to expose power. If another pending instruction needs that value, move recognition of the expression into the shared query vocabulary before extending the stack projection. The exact-source lookup already owns the value.

Behavior review at `64eaa87c` reproduced both M21 damage outcomes after reentry and both history-free Python projection cases. The Panda engine snapshot test also passed with both target outcomes. The pending-power defect is closed. Payment rules and authored tokens did not change in this correction. Integrated stale-Command and hidden-card permutation checks remain open.

## Implemented slice · Lunar Convocation

`blb.lunar_convocation` reads its controller’s existing gain count and life lost through two intervening-if triggers. The Bat uses `token.bat`, a black 1/1 with flying. Both triggers use the native end-step matcher and resolution check.

The two directed BLB fixtures gained life from Cat Collector’s Food before casting Convocation. From the same saved decision, paying before the end step offered two trigger orders. Both left one Bat and the opponent at 19. Paying after the gain-only trigger was on the stack drew a card and left the opponent at 19 with no Bat. Both restored decision frames and search witnesses matched the root. Three payments after two Foods returned the controller to 20 and still created a Bat. Three payments without a gain triggered neither ability. Neither subsequent end step carried over the history.

The timing fixtures in `managym/cards/tests/cards/blb.rs` establish that existing controller history suffices even when the gain predates the enchantment. Keep the two printed triggers independently orderable. The shared condition reader in `managym/src/flow/statics.rs` tests each counter directly, so Convocation needs no history of its own. The mid-game training snapshot merge at `628fb536` left the BLB definition and fixtures intact.

Behavior review at `5717184a` reran both BLB fixtures and seven shared trigger-order debug cases. Same-source triggers retain distinct ability joins, renumbering trigger occurrences preserves the encoded observation, and the opposing viewer omits the chooser’s ordering frames. The nested-fork, exact-description, stale-Command rejection and runtime catalog-token checks passed. Source tracing found that v11 emits the gain count as a player fact and life lost in the turn-state buffer, and LLM text reads both from native turn history. Academy declares both native fields. A Convocation-specific Academy frame comparison and the integrated hidden-card permutation remain unverified. No payment algorithm changed. The Convocation timing obligation is satisfied. The BLB witness leaves the primary FDN count unchanged.

## Implemented slice · Predator Ooze

The Dark Ascension definition uses the ordinary attack trigger and a death subject that checks exact source–recipient membership. The turn records positive damage once per pair and drops all pairs at the next turn. Death checks the effective creature type captured at departure. Hash v20 records both incarnations. Native observations expose the live recipients of each live permanent, using public permanent IDs. V11 relation kind 11 joins both rows without policy history.

The native snapshot fixture restored after a surviving hit. Hitting Hill Giant again and then killing it put one trigger on the stack. Resolving grew the original Ooze to 2/2, while countering in a sibling left it 1/1. Reentering either permanent or advancing the turn before death produced zero triggers. Two Oozes hitting the same creature produced two triggers. Restoring and playing the siblings preserved the parent witness. With two Oozes and equally marked Giants in sibling games, both viewers’ encoded relations identified the different sources and the same recipient.

In the Dark Ascension combat fixture, attacking grew Ooze to 2/2. Hill Giant blocked and survived two damage. Fumigate then killed the Giant, gained one life and left a 3/3 Ooze. The Kaladesh fixture separately verified gains of two and zero with Ooze surviving both casts. The Clone suspended-entry hash witness was refreshed for the catalog and hash contract.

The CPython 3.13 release wheel was rebuilt and installed. Four focused Python checks passed for LLM text, Academy projection, field coverage, named layout and protocol parity. The native experience schema matched its checked-in bytes. Both bundled pinned pilot replays passed; the optional frontier checkpoint was unavailable.

Behavior review at `31316346` reran the exact-pair snapshot and Dark Ascension combat fixtures after the sorted-vector change. Both passed, including the two-viewer tensor comparison, repeated hit, two sources, both reentries and turn reset. The death trigger has no target or optional choice. V11 raises a capacity error if the relations cannot fit. The text and Academy projection test passed with a constructed viewer payload. All six search-state contract cases, the stale-Command rejection case and the runtime catalog-token control passed. These are shared controls; the integrated hidden-card permutation and replay through every consumer remain open. Predator Ooze meets its working obligation. Ooze closed the fifth primary FDN row.

## Implemented slice · Steel Hellkite

The sixth primary FDN row is `som.steel_hellkite`. Hash v21 binds combat-player hits to the source incarnation. The Scars of Mirrodin fixture attacked with one of two Hellkites. Both offered X=0 from the saved position, and each spent its own activation. The hitting copy destroyed the opponent’s Food while its own Food, an opposing land and Hill Giant survived. At X=4 only the Giant died. On the next turn the same Hellkite activated again and a new opposing Food survived.

The native snapshot fixture compared a combat hit and a noncombat hit by different copies. It submitted the saved activation Command in each restored game, chose zero and changed control of a Food before resolution. After the hitting source left, the restored pending ability still showed the damaged seat to both viewers, including in the v11 stack tensor. Only that activation destroyed the transferred Food. Countering the other copy’s activation left its use spent. The root witness stayed unchanged.

The text-play fixture exposed a missing stage in the renderer for announcing an activated ability’s X. With that stage admitted, the player attacked, saw X=0 and X=1, chose zero, and saw seat 1 on the pending ability in the text, both Academy frames, restored observation JSON and viewer-relative tensors. The rebuilt CPython 3.13 extension passed both bundled pilot replays. The optional frontier checkpoint was unavailable. Academy’s Svelte check reported zero diagnostics.

At `73475291`, Ooze and Hellkite share one sorted damage history. Resolution and observation read the combat-player range for the exact source. The v11 permanent and stack rows use the same viewer-relative seat encoding. This answers the pending-source query even after the permanent row disappears.

Review at `426f0470` found that an untapped Hellkite with no mana source offered X=1 and panicked on submission. The combat fixture missed this because attacking tapped the Hellkite.

At `5a5f87c0`, `flow/play.rs::legal_activation_x_values` reads the mana component built by activation payment, including waterbend eligibility, and reserves the source’s own costs in the announcement model. `som.rs::steel_hellkite_offers_only_payable_x_values` offered [0] with Hellkite alone and [0, 1] with a Mountain. Every branch resolved from the saved X decision, leaving Hellkite untapped and tapping the Mountain only for one. The rebuilt CPython 3.13 extension returned the same complete offers through text play and accepted every response after snapshot restore. The genuine waterbend control (`tla.rs::katara_waterbends_x_to_set_your_creatures_base_pt`) offered [1, 2, 3, 4] and paid two by tapping the Ally and Ogre while keeping its floating mana.

Working review at `fb5cc087` reran both Hellkite fixtures, Katara’s waterbend X fixture, automatic Ally-only spend, Animal Attendant’s two manual spends, spend-rider receipt projection and waterbend versus floating-mana accounting. All seven passed in the debug cards target. The Hellkite payment finding is closed. The full working phase, integrated replay/privacy validation and Engine Budget comparisons remain open.

## Implemented slice · Lunar Insight

The FDN ledger records `fdn.lunar_insight` as the seventh delivered primary row. In `managym/cards/tests/cards/fdn.rs::lunar_insight_counts_distinct_controlled_nonland_values_at_resolution`, each branch starts from the same announced spell. Controlled nonlands with mana values [0, 0, 2], [0, 1, 2] and [] drew two, three and zero cards respectively. The Island and opposing Gray Ogre remained present in every branch. The card fixture and `observability.rs::lunar_insight_catalog_tokens_describe_its_draw_amount` passed in the implementation slice.

Working review at `998e139b` reran both debug cards tests matching `lunar_insight`. Both passed. The empty branch would draw for either the Island or the opposing Ogre if its exclusion failed. The two three-permanent branches distinguish counting values from counting objects. The ordinary DrawCards instruction evaluates the expression once before drawing. The same authored selector supplies the native text description and catalog tokens. There is no intermediate choice or retained aggregate. This fixture restores a pending game clone. Command replay, stale rejection and cross-viewer tensor and text parity remain integrated validation obligations.

The broader query work can proceed one instruction at a time. In `managym/src/flow/resolution.rs`, `DistinctManaValues` reuses permanent selection and the existing mana-value reader. Its set lives only while evaluating the draw count. This card needs no announcement snapshot of the board or persistent turn fact.

## Implemented slice · Felling Blow

The FDN ledger records the native fixture for `fdn.felling_blow`. From one pending spell, Gray Ogre gained a +1/+1 counter and dealt three noncombat damage to Steel Hellkite. Returning the Hellkite from exile preserved the Ogre's counter and prevented damage. Returning the Ogre prevented both effects. Restoring the intact spell reproduced the counter and the attributed hit. Both focused cards tests passed, including the catalog tokens for counter placement and power damage. Ordered target requirements and the existing damage instruction handled all four resolutions.

At `0ad573b3`, the fixture checks the committed damage event’s source, recipient and amount on both intact resolutions and checks for zero hits on either blink. The same two authored instructions supply the catalog tokens. This closes the card-level obligation using the announced target slots and live power. The repeated intact branch restores a game clone. Command replay and viewer parity remain in integrated validation.
Behavior review at `0e739651` reran both Felling Blow tests and confirmed the ledger points to the passing fixture. Three shared debug controls passed for positional damage with an illegal victim, stale Command rejection without mutation, and viewer-visible capture identity. The native target prompts feed structured offers, the tensor encoder checks the configured action capacity, and semantic text pairs each offer with its observation action. This consumer trace is by inspection. The constructed card fixture has one creature on each side and exercises ordinary automatic payment. Integrated checks still cover competing target offers, Command replay and viewer parity. The Felling Blow working phase is satisfied.

## Implemented slice · Drake Hatcher

Each committed combat hit carries its exact source, recipient and damage amount through replacement, trigger placement and resolution. The v11 `triggering_damage` fact joins the amount to its continuation frame. Both viewers see the stacked amount. Counter removal is an announced activation cost, paid before the ability goes on the stack.

The FDN fixture restores simultaneous one- and three-damage triggers after removing the power boost. Each source receives its own incubation counters. The activation is absent at two and offered at three. From the paid fork, resolving creates a 2/2 flying Drake and countering creates none. Both leave zero incubation counters. The native Env snapshot fixture restores the same precombat source for one and three damage and restores each pending trigger after its power changes. The replacement-order control and Invasion Tactics’ one-draw-per-batch control passed. Python text and Academy decision context also exposed the bound amount. The FDN ledger links the card fixture. Hatcher belongs to the additional cost-payment cohort.

The unequal-hit fixture in `managym/cards/tests/cards/fdn.rs` checks both native viewers and their v11 fact rows. The Python fixture in `tests/semantic/test_text_observation.py` checks a one-damage hit in acting-viewer text and both Academy decision contexts. Together these cover the requested public amount. The attempted inline trigger-placement state exceeded the 4096-byte GameState limit. Boxing the placement round kept the limit. Keep the committed hit on the event and its bound amount on the pending trigger. Neither counter placement nor activation needs a turn-wide damage total.

The catalog fixture update at `ef9a8011` supplied reference data. Count delivery from executable definitions and directed fixtures. The merge at `58896721` includes upstream MLX self-play work. Integrated consumer and budget checks remain open on the merged revision.

Behavior review at `a36dcdb2`: both `drake_hatcher` debug card tests and the native snapshot test passed. Three focused text tests passed: Hatcher pending damage, manual mana-source Commands (including stale rejection with unchanged digest), and hidden-hand permutations. Hatcher spends only counters, so the earlier pooled-mana and receipt controls still apply. The Hatcher slice meets its working obligations.

## Implemented slice · High-Society Hunter

The two `high_society_hunter` native card fixtures passed. From one attack trigger, sacrificing a Cat added a counter without drawing and declining kept the Cat. Fumigate destroyed Hunter, a friendly Gray Ogre and an opposing Island copying Gray Ogre. Both viewers saw two Hunter death triggers. Restoring the pending game clone yielded two draws or one after countering a trigger. Only Hunter’s controller saw the drawn cards in hand. The existing Predator Ooze damage/death fixture also passed. The shared departure predicate reads creature status and power from the exact departed permanent. Token status is stable across appearances. Vampire is now a classified creature subtype.

## Next slice · Cackling Prowler

Branch from Hunter’s attack trigger with Prowler in play. Sacrifice the Cat in one branch and decline in the other. With no other deaths, only the sacrifice branch should give Prowler a counter at its controller’s end step. Carry that game into the next turn and verify the fact resets. Target Prowler from the opposing seat and exercise paying and declining ward.

Hunter’s death matcher already reads effective creature status at departure (`managym/src/flow/triggers.rs`). Have the turn-history writer use the same departure fact. The matcher still reads other card types from the moved card. If a remaining predicate needs another effective type at departure, consolidate that lookup before extending it.

## Evidence to carry forward

Eight of 81 primary rows have definitions and directed fixtures. Hunter and Hatcher are cost-primary witnesses. The 73 remaining primary rows, integrated replay/privacy checks and Engine Budget measurements are open. Once the cohort stabilizes, refresh the semantic kernel, authored-match and Academy parity receipts in `conformance/README.md`. The historical v17 witnesses predate the catalog and hash changes.

Earlier payment controls passed for Ally-only automatic spend, Animal Attendant manual spend, spend-rider receipts and mana floated during Uncle's Musings. Expend and Unravel still need the cast and counter interactions below.

Checkpoint construction at `7334e781` uses the saved fact embedding width. The merge at `a41f9b08` brought in live game snapshots. Its existing restore checks live in `tests/env/test_game_snapshot.py`. The Fumigate, Exsanguinate, Collector and Connoisseur results above predate that merge. The Panda fixture exercises the merged snapshot path.

## Remaining native work

1. **Count at the event that establishes the fact.** `flow/event.rs::GameEvent`, `zones.rs::emit`, and the draw, combat, cast and resolution commit sites supply the facts. Extend `flow/turn.rs::PlayerTurnHistory` and the exact-source counts with only the values and relations required by the cohort. Retain the event-time ordinal or threshold crossing when a trigger will be matched later. Track occurrences and amounts separately. Attribute departures using pre-move controller and characteristics. Key damage relations by exact source and recipient. Reset turn facts for both players at the turn boundary, including extra turns. An extra combat uses the same turn. A game-scoped query retains its fact across turns and starts empty in a new game. The presentation ledger can be drained and optional policy memory can be disabled while these rules still work.
2. **Use one vocabulary at each query site.** Extend `managym/state/src/ability.rs` (`EffectValue`, `Condition`, selectors and target clauses) and `predicate.rs::CardPredicate` with composable subjects, filters and aggregates. Bind controller, exact source, trigger object/player, announced target, paid contribution or scoped result in the calling context. Share characteristic lookup and comparison across `flow/resolution.rs`, `statics.rs`, `trigger.rs` and `triggers.rs`. Use live layered characteristics or the exact departed object's last-known information (LKI) as the instruction requires. Target offers evaluate at announcement and recheck on resolution. A snapshot or a committed result retains its original value.
3. **Read actual payment.** `managym/state/src/payment.rs::PaymentReceipt` already owns mana units, colors, life and contributing objects. Read the receipt of the spell or ability identified by the question. Expend accumulates committed mana spent casting spells by that player. Mana floated, ability costs, rolled-back spends and uncast copies contribute zero. Unravel compares the target spell's own spend and stack mana value (including X) even after the counter instruction. Preserve manual source/pip/unit alternatives, automatic restricted-mana priority, native chooser and ordinary forced-action skipping.
4. **Project the rules state once per viewer.** Include decision-relevant facts and pending result bindings in native observations, exact forks/search and the canonical state witness (`src/state/hash.rs`, with its hash-version contract). Project observable scalars/relations through `agent/observation*.rs`, `src/python/*bindings.rs`, named v11 buffers or typed fact rows, `manabot/semantic/text_observation.py`, and Academy's native decision projection. Extend `src/cards/tokens.rs`, `agent/ability_description.rs` and `record_vocabulary.rs` for new authored expressions and record kinds. Keep saved learning-token meanings, column prefixes, v9/v10 and frozen checkpoints intact. Use public object joins for visible relations. Keep hidden identities and raw storage IDs out of both text and tensors.

### Delete when the shared reader takes over

- Source power, power tapped to pay and the graveyard target bound share `permanent_power_or_lki`. Triggering power and target toughness still use a battlefield slot and its departure record. Preserve Combustion Man's existing reentry behavior and Sandbender Scavengers' graveyard-target bound. Replace the bespoke bound in `TargetSpec::GraveyardCardYouOwnWithinSourcePower` when the shared clause can express its full behavior. Preserve authored token bytes for existing definitions, including compatibility mappings if representation changes.
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
