# MTG-418 · Event history and contextual queries

## Outcome and cohort

Implement one native way to ask what happened, to whom, how often, and with what result. Start with Fumigate + Exsanguinate (instruction results), Cat Collector + Venom Connoisseur (different clocks), and Heartfire Immolator + Fiendish Panda (exact sources). Each pair exercises a different distinction. Then author the rest of the accepted cohort through the same rules types.

At checkout revision `0b1cc9cfef9c1a98430c825b9764a49237182696`, the [FDN ledger](../.lf/wave/coverage/research/standard-2026-11/sets/fdn.jsonl) contains **47** primary `event_history` and **34** primary `context_queries` rows. Use their Oracle IDs, obligations and secondary dependencies as the worklist. Heartfire Immolator and Fiendish Panda are additional named witnesses classified under `object_lifetime`. Among cost-primary rows, 17 cite either dependency, and the Task also names Incinerating Blast, a cost-primary card with a `choice_continuations` dependency. Check the remaining cost obligations for newly expressible cards as well. A dependency tag alone does not settle delivery.

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

The initial projection matched a direct damage expression. The unified query slice below replaced that branch with the shared recursive program reader.

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

Behavior review at `41cbdd99`: both Hunter card tests passed again. The FDN ledger links both fixtures. Hunter uses the shared optional sacrifice and conditional effect instructions. Its pending draws use the ordinary stack source and ability index, which the v11 encoder projects even after Hunter leaves. Three shared consumer controls passed for text stack links, hidden-hand permutations and manual Commands with stale rejection. Hunter adds no mana spend or token instruction. Its working slice is satisfied. The forks exercised native game clones; integrated replay, Academy parity and budget checks remain with gate.

## Implemented slice · Cackling Prowler

At `7ada41e6`, the graveyard move in `flow/zones.rs`, delayed death watchers and Hunter in `flow/triggers.rs` read the exact departing object’s `was_creature` field. Looking up the card in its destination had missed creatures whose copy effect ended on departure. As the sole departure, an Island copying Gray Ogre set the turn fact and fired the watcher on death. Exile did neither. Flipping only the turn fact changed the canonical hash. Its version is 23.

Restoring the native game clone after Hunter's sacrifice choice reproduced the end-step hash. Both viewers saw Prowler's pending ability only after sacrifice, and only that branch added a counter. The next turn started with the fact clear. Paying ward let Lightning Strike kill Prowler and declining kept it alive.

The initial v11 projection used a scalar fact on the active player row. The unified `player_turn_state.creatures_died` column now carries each seat’s deaths. The native restore fixture at `556b0dc2` compares end-step hashes for sacrifice and decline. Command replay and Academy parity remain integrated checks.

Behavior review at `9a119c98`: five focused Rust tests passed (both Hunter fixtures, Prowler ward, copied Island death versus exile, and default catalog token spans). Three Python controls passed for the public death fact in text, tensors and serialized observations, hidden-hand permutations, and manual mana Commands with stale rejection. Prowler uses the shared controller end-step condition, counter effect and ward payment. Both ward choices reached their expected outcomes. The Prowler working slice is satisfied. The restored branches exercised native clones; end-to-end Command replay and Academy parity remain open.

## Incinerating Blast · rules, replay and viewer privacy verified

Blast uses the existing damage and optional discard/draw effects in printed order. Gray Ogre had six damage marked when the caster could decline or discard either Forest or Mountain. Each paid fork put the selected card in the graveyard, kept the other in hand and drew once. Decline kept both cards and drew nothing. Exiling the Ogre from the pre-resolution fork returned directly to priority with the hand and library intact.

The fixture at `314de3ce` forks the native action space with lethal damage marked and the Ogre still on the battlefield. Each completed branch finds the Ogre in the graveyard. The existing `Effect::discard_draw(1, 1)` owns the conditional draw, so Blast needs no new history field or continuation type. The FDN ledger links the fixture in `managym/cards/tests/cards/fdn.rs`.

Behavior review at `31982cba`. The directed Rust fixture and default catalog token-span control passed. The shared selection moves the chosen hand card to the graveyard and draws from the committed moved count. A constructed Python first-main position with five Mountains, an opposing Gray Ogre and Blast, Forest and Island in hand reached three text offers after cast, target and two passes. On two independent clones per offer, submitting the same reference against each clone's own capture reproduced its digest. The three outcomes differed, the parent retained its digest and all six duplicate submissions returned `stale_capture` without mutation. Submitting the parent's capture on a clone returned `wrong_origin`. Three existing Python controls passed for selection text/tensors, stale rejection and hidden-card projection. The native rules slice is satisfied. The shared controls do not establish Blast-specific tensor or Academy privacy parity.

At `6c230ba7`, all three focused cases in `tests/academy/test_incinerating_blast.py` passed. Each bound Command reproduced the full transition, next decision frame and search witness on two restores. The three outcomes had distinct digests. Both seats saw one completion and the expected graveyard in Academy, including the selected card after discard. A spent Command returned `stale_revision` with the committed witness intact, and the parent retained its original witness.

With identical closed lists, substituting Plains for Island changed the caster’s native hand projection, encoded card rows, text choice and Academy offer. The waiting viewer’s native and Python JSON, every tensor buffer and Academy recovery frame stayed equal. Only the chooser receives an LLM decision capture. WebSocket resume returned the same three offers after the fixture initialized the session from the native snapshot. The fixture uses ordinary forced-action skipping and existing recovery and Command APIs.
Behavior review at `bca1f2d9` reproduced the native Blast fixture (1 passed) and the Command, viewer privacy and WebSocket fixtures (3 passed). The requested Blast slice is satisfied. The reconnect check resumes a live session initialized from a snapshot. Recorded Watch frames remain an integrated validation obligation. The working phase remains open for the other ledger rows and cross-set witnesses.

## Implemented slice · Bakersbane Duo

The native BLB fixture casts Gray Ogre for three, spends two on a Food activation, then casts Tolarian Terror for two with five instant cards in the graveyard. Both Duos present get one +1/+1 trigger. Restoring with the triggers pending and countering the Terror gives the same two pumps as letting it resolve. Casting a third Duo brings the total to seven. A subsequent full-price Hill Giant brings it to eleven and leaves the late Duo at 2/2. Both players’ totals and the pumps reset on the following turn.

`commit_spell_payment` binds the caster and the before/after totals from the spell’s retained receipt. Cast listeners use that caster directly. Native turn history participates in canonical hash v24. The initial v11 projection used fact 19. The unified `player_turn_state.mana_spent_casting` column now carries the payer’s total. The card uses `TriggerCondition::Expend(4)`, the existing Food token, and the existing temporary pump. Raccoon is classified in the native subtype table.

The directed native crossing and the trigger learning-token check passed. After rebuilding the CPython 3.13 extension, the named fact-layout check and the live two-viewer text/tensor test passed. The latter cast Serra Angel for five with a Duo in play, then passed priority. Each chooser saw five mana spent in the text and one fact joined to the caster, first as self and then as opponent.

Behavior review reproduced seven native debug fixtures and two Python checks. The discounted cast crossed from three to five on its committed two-mana receipt. Both original Duos reached 3/3 in each restored branch, including after Counterspell. The Food activation, later casts and late Duo produced no extra pump. Both viewers read the payer’s total from native history, and the Serra Angel control reproduced the viewer-relative v11 join and text. The shared trigger token, restricted-first auto spend, manual spend alternatives, floating mana and stale Command rejection controls passed. Expend introduces no choice of its own. The Bakersbane working slice is satisfied. Integrated replay and recorded Watch remain for gate, and the Unravel interaction remains in its planned batch.

## Implemented slice · Second draw

Erudite Wizard, Homunculus Horde and Mischievous Mystic use the committed draw ordinal. Their shared FDN fixture observed ordinals 1, 2 and 3, one pending ability per source, one counter, one Horde token and one 1/1 flying Faerie. A live Clone copying Horde and a restored pending trigger after Clone returned to hand both made a Horde token. Countering only Clone’s trigger preserved the other two rewards. The late Wizard missed the crossing and triggered on the second draw of the next turn. The opposing Horde triggered only on its controller’s draws. Both viewers saw the same three pending ability descriptions and draw count, with private hand rows confined to the owner.

Exact battlefield departure records retain altered copiable values for copied or transformed objects. Token creation reads those values after departure and reads the current object while it remains live. Canonical match-state hash v25 includes the departure record. After the CPython 3.13 rebuild, the Lunar Insight text-decision fixture passed: both choosers saw “drew 3 cards” and 0.3 in the drawer’s v11 player column. Recorded Watch comparisons remain in integrated validation. The three definitions now share `on_second_draw` in `managym/cards/src/fdn.rs`. The draw event already carries the ordinal, so future listeners can reuse the trigger without storing a count on each permanent.

**Behavior review at `de5a9f62`.** All six debug `cards` fixtures selected by `second_draw` passed, including the FDN interaction above and five existing TLA controls. The draw commit in `flow/zones.rs` supplies the ordinal. Trigger matching checks that ordinal and the drawer against the source controller. Placement offers one ChooseTrigger per remaining candidate in the controller’s group. The Lunar Insight text/v11 fixture passed through that ordering decision. The nine `reaches_the_learning_tokens` observability controls passed for the shared expression vocabulary.

The stale-object search fork control and Academy’s original Command import and stale-rejection control passed. These are shared boundary checks. The departed Clone fixture restores an in-memory clone of the pending game. Recorded Watch playback of this interaction remains an integrated check. Payment behavior was outside this diff. The second-draw slice meets its working obligations. The shared declaration work followed in the next slice.

## Implemented slice · Unified queries and attack declaration

The four query changes and all twelve attack/raid rows are implemented at
`5be4b4d5ba`, `8498836e03` and `c9bdc388ba`. Linden and Chart a Course were
already authored. MTG-422 is integrated: match-state hash v26 reconciles both
changes, and v11 permanent column 51 remains `entered_this_turn`. The
opening-hand conformance case and suspended Clone hash witness were refreshed.

- **Turn history.** `flow/turn.rs::PlayerTurnHistory` owns closed per-seat
  counters. The v11 `player_turn_state` suffix replaced scalar fact kinds
  for life gains, creature deaths and casting spend. Counts and amounts
  have separate columns, deaths and departures follow the departing
  controller, and turn-wide predicates read both seats. Sparse object and
  frame facts retain their own rows. `TriggeringDamage` became kind 16
  before release. Legacy v9/v10 layouts keep their eight-column prefix.
- **Declaration and placement.** `flow/tick.rs` commits raid and
  exact-object attack ordinals from the declared attackers. Creatures that
  entered attacking stay in combat without contributing to that suffix.
  `flow/triggers.rs` shares occurrence emission for declared attackers and
  combat hits. The doubler fixture changed from the earlier failure of
  three triggers to four, with two per attacker. First-attack conditions
  read `attacks_this_turn`, which records declarations made while an
  ability was absent. Printed once-per-turn triggers and resolutions key
  by object, printed face and ability index. `AttackSize::AtLeast` supplies
  count thresholds.
- **Common readers.** Counted mana outputs and stack effects use
  `Game::select` through `effect_value`. `Condition::Compare` reads shared
  amounts. Targeted spell spend and stack mana value are captured in the
  resolution targets, so the comparison survives the counter instruction.
  The discounted and full-price Tolarian Terror fixture verifies this
  primitive. The authored Unravel interaction is still open.
- **Stack and target authority.** `agent/observation.rs` uses
  `Effect::reads_source_power` for activated and triggered programs and
  exposes each spell’s actual `mana_spent`. Activated target selection now
  excludes the source when the clause says “another”. Deserter’s Disciple,
  North Pole Patrol and Earth Kingdom Protectors needed explicit
  `TargetRequirement::another` to preserve their printed options.

The FDN ledger links all twelve definitions to the shared
`managym/cards/tests/cards/fdn.rs::attack_batch_*` fixtures. The 0/2/3
declaration fixture covers Armasaur Guide, Skyship Buccaneer, Midnight
Snack, Battlesong Berserker, Goblin Boarders, Gorehorn Raider, Searslicer
Goblin, Ancestor Dragon, Jazal Goldmane and Storm Fleet Spy. Separate cases
cover Alesha’s departed-source target bound and Aurelia’s ability loss,
blink, entered-attacking case, restored partial declaration and next-turn
reset. The earlier Linden/Chart forks verified both completing and
declining the rest of a declaration, early raid remaining false, and an
empty extra combat preserving committed raid. The existing firebending
and native Command controls passed during that review.

The debug native workspace run at the initial unification passed 1,551
tests after the three TLA target clauses, Alesha’s Minotaur subtype and
the refreshed witnesses were in place. Compression at `c9bdc388ba` passed
34 focused native and 12 focused Python checks, including both bundled
pilot replays. The optional frontier checkpoint was unset. The
`PlayerTurnHistory` Python round trip retained all 13 fields without
changing the live game. The two-viewer Midnight Snack checks showed
two gains totaling five. In the constructed Alesha game, an end-step
trigger returned Immolator after Lightning Bolt killed Alesha. Both
viewers saw source power 3 in the Academy label and history-free decision
text, and 0.3 in the v11 stack row. The native cases and Python text
fixtures are in `cards/tests/cards/fdn.rs` and
`tests/semantic/test_text_observation.py`.

### Behavior review at `54b4c6042c`

MTG-422's `d12eb496b3` and the host-preparation merge `074bc49a64` are
ancestors of the reviewed head. All eight `attack_batch_` fixtures and 44
selected debug cards controls passed, including Linden/Chart, TLA firebending,
target authority, token coverage and payment. All 12 debug cases in
`search_state_contract` and `canonical_replay_tests` also passed. The earlier
bare per-attacker doubler failure is closed. Two declared members each retain
their own two triggers. The previous Linden formatting and hash-version
findings are closed.

- The 0/2/3 attacker fork produces one Armasaur trigger at three and one
  Battlesong trigger for each nonempty declaration. Both place their target
  through ordinary `ChooseTarget` offers. Existing attackers join Ancestor
  Dragon's and Jazal's resolution-time counts through the live attacking
  predicate. Raid reads the turn's committed declaration regardless of who
  remains in combat. Aurelia reads the exact object's first declaration, so
  becoming an attacker by entering combat and regaining a lost ability have
  different outcomes. These checks agree with CR 506.5, 508.3–4, 603.2d and
  608.2h. [Comprehensive Rules](https://magic.wizards.com/en/rules)
- Alesha's native target offers exclude Serra Angel at power three, and the
  pending trigger retains power three after Alesha leaves. Both native views
  and v11 stack rows show it. The Panda and Immolator witnesses above exercise
  the same exact source across reentry. Midnight Snack reads the five life
  gained from two events, including after sacrificing the Snack. Food and
  Goblin creation use the catalog's existing token path. The distinct gain
  and amount columns and text checked out on the rebuilt extension.
  [Foundations release notes](https://magic.wizards.com/en/news/feature/foundations-release-notes)
- The Tolarian Terror branches retain six or seven spent mana in the native
  spell and both v11 rows. The comparison still sees the chosen spell after
  Counterspell moves it off the stack. The counted mana ability produces one
  or two green in the live-board fork. The payment controls confirmed the
  restricted Ally pool pays before ordinary mana, Animal Attendant offers
  both spends for the green pip, and Uncle's Musings offers meaningful spends
  for its generic pips with different receipts. Waterbend taps and leftover
  floating mana stay distinct from the committed contributions. Bakersbane's
  two viewers also read the caster's committed turn spend in text and v11.

The installed CPython 3.13 extension predated `axis_buffers` from the host
preparation merge. After rebuilding and atomically replacing it, the focused
Python selection reported 39 passes and the six Academy fixture failures
below. Five additional checks for v11 prepared joins and Torch host admission
passed. The passing controls include named layout and v10 model ingress,
text and tensors for public spend and draws, manual Commands and stale
rejection, hidden-hand permutation, and a serialized snapshot that replays a
Command with issued object bindings.

Two in-memory restores per offer replayed the yes and no Commands from one
partial Linden/Savannah Lions declaration. Each duplicate returned
`stale_revision` with its witness intact, and both branches preserved the
root. Finishing the declaration produced one or two committed attackers in
both viewer JSON round trips, v11 permanent rows, history-free LLM text and
the Academy Python projection. A separate pending Serra Angel cast showed
five spent mana to both choosers through the same surfaces.

**Power batch behavior review at `59a8f767a2`**

The working phase for the power batch is satisfied. All 13 `power_batch_` debug
cases passed, as did the whole debug `cards` target (877), the `managym` lib
(358), and `search_state_contract` plus `canonical_replay_tests` (12). After
the extension was rebuilt at this head and replaced atomically, 120 focused
Python checks passed and one skipped (the optional u1952 checkpoint). They
covered text, effect execution, v10 input, named layout, pinned pilots and
Immolator source power. All eight ledger rows link to their source and fixture.

- The definitions match the printed rules. Goblin and Ruby check "while"
  only when the trigger fires (`ActiveIf`). Hornbeetle and Surrak use
  intervening-if conditions that resolution checks again (CR 603.4), and
  Surrak's total includes negative power. Ashroot and the Partners fix X at
  resolution from live power or LKI (CR 608.2h). For negative power, Bulk Up
  follows the Unleash Fury doubling ruling. Bite Down does nothing if either
  target is illegal (CR 608.2b).
- `match_trigger_wrappers` removes `WasCast`, `FromZone`,
  `WhileSourceAttacking`, `ActiveIf` and `OnceEachTurn` before the leaf
  emits members. The battlefield and graveyard paths share it. A
  once-per-turn limit keeps the first member that passes the intervening
  condition and caps doublers at one. There is no official ruling for that
  cap; it follows a judge answer (Teysa/Sefris) and is recorded as the rules
  reading. Grant clocks key on (recipient object, grantor, timestamp, static
  slot, ability index), so a reentered recipient or grantor gets a fresh limit.
  Used grants count toward the public `once_each_turn_used` and enter the
  match hash only when nonempty.
- Fixed during review: `flow/triggers.rs` imported `Condition` without using
  it, so the debug build warned and clippy would fail under `-D warnings`.
  The import is gone.
- Constraints for later batches, not defects here:
  `EffectValue::Property { of: Target(_) }` reads `frame.primary_target()`,
  so a card that reads the property of its second target needs the clause's
  own target. Static `Condition::Compare` in `layers.rs` reads power at the
  current point in the layer system and mana value from the live card, not a
  layer-1 copy. No authored static depends on either case. No fixture changes
  Ashroot's power after resolution to show the pump stays fixed; the
  `ModifyPt` path already stores a fixed value.

**Life batch behavior review at `67f9d3ee34`**

The working phase for the life batch is satisfied except Finding 6 below. These passed at this head, all in debug: the cards target (887, including the ten life-batch cases and the Sozin's Comet/Katara control), the `managym` lib (359, including the live-attacker heuristic control), and `search_state_contract` plus `canonical_replay_tests` (12). The extension was built at `a0b213f734`, and the compression after it changed no behavior. With it, 109 text-observation and named-layout Python checks passed. All eight ledger rows name their first-printing source and fixture. Conqueror's row also records the deferred CR 104.4b draw as pending.

- The definitions match the printed rules and the Scryfall rulings. Seraph's `nth: Some(1)` reads the turn's committed gain ordinal, so a Seraph that arrives after the turn's first gain waits for the next turn. Conqueror binds the life actually lost, so Teferi's Protection gives no event. Life paid as a cost goes through `commit_life_change` and triggers it, per the ruling. Bloodthief and Collector use CR 603.4 intervening-if conditions. Second Rite reads the life total on resolution (CR 608.2c). Ayli reads the sacrificed creature's LKI toughness, shows the pending amount to both viewers, and checks "activate only if" only on activation (CR 602.5b, matching the ruling). Indulgence's discount is generic mana only.
- Findings 1 and 2 are closed. Firebending grants are ordinary attack triggers that pass through the shared occurrence path, and the heuristic reads the live `attacking` permanents.
- Rules reading recorded in questions.md, not a defect: each combat source's damage to a player is its own `LifeChanged`, so two unblocked attackers give two Conqueror triggers (2 and 3) rather than one trigger for 5. The total gained is the same. No CR rule or card ruling settles whether simultaneous damage from several sources is one life-loss event; CR 119.9 only covers gains.

**Mana-split batch behavior review at `23797c77f9`**

The working phase for the mana-split, `DealsDamage` source, `*/*` CDA, move-result and multi-player mill batch (`f5e7244bf4`–`3be3b3295f`) is satisfied, with no new finding. These passed at this head in debug: the cards target (944), the `managym` lib (361), `rules_tests` (190, 2 ignored), `search_state_contract`, `canonical_replay_tests` and `match_state_hash_tests`. The compression `3be3b3295f` changed no behavior: a pattern rewrite in `value_description` and the `object_pt(..).0` rename. The focused Python result recorded at `965b0e0ea1` (116 passed, 1 skipped) therefore still applies. This review did not rebuild the extension.

- The definitions match the printed rules and CR. Kiora's and Crypt Feaster's threshold is an intervening-if that is checked again on resolution (CR 603.4), and the Feaster drop-below-seven fork shows it. Stroke of Midnight gives the Human to the target's last controller, including when an indestructible target survives. Feed the Swarm reads mana value by LKI. Maelstrom Pulse destroys nothing when its only target is illegal (CR 608.2b). `accepts_name` rejects an empty name, so a nameless target has no namesakes (CR 201.2a). Cemetery Recruitment draws only for a Zombie returned to the hand. Revenge of the Rats counts on resolution and is exiled after flashback. CDAs apply in every zone (CR 604.3): Crusader and Aberration in hand, the v11 card row and stack rows, all through `object_pt`.
- Payment: probe not kept. Archdruid, an Elf token and one Forest pay for Llanowar Elves. Manual payment offers the Archdruid activation, and tapping it commits one G with one G floating separately in the pool (`spent=1`, pool G=1). Auto payment taps the Forest and leaves the Archdruid untapped, so it wastes no mana. Every `mana_choices` consumer (offers, autotapper, preview, decision memory) reads finalized layered views. `current_mana_output` returns a concrete output for every `is_mana_ability` leading effect, so no dynamic ability reaches those consumers unfinalized.
- Constraints for later batches, not defects here. The `DamageDealt` binding falls back to `Target::StackSpell(source)` for a source that is neither a permanent, its LKI nor a stack spell (for example, a hand ability such as channel). No authored card reads "that source" for such damage, so the fallback is unexercised. `TargetAndSameName` iterates seats 0 and 1, as `Scope::Battlefield` does. Dread Summons' replacement-exile control still needs a catalog replacement.

**Findings** (Finding 10, open, is under the closing-slice behavior review)

3. **Closed in `cf12733e50`.** Six Academy fixtures still assumed reset starts in priority: two cases in
   `test_semantic_boundary.py` ask for `pass_priority` at the opening Keep
   offer; `test_combat_preview.py` and all three Blast cases inject a board
   while the action space is `Modal`. Keep both opening hands through native
   offers before reaching the fixture position, then rerun the offer,
   stale-command, recovery and privacy assertions.
4. **Closed in `6aa5452f11`.** Card fixtures for the attack and power batches lived in
   `managym/cards/tests/cards/fdn.rs` even when the card's first-printing
   module is another set (Bite Down `dmu.rs`, Surrak `dtk.rs`, Halana and
   Alena `vow.rs`, Ruby `woe.rs`, among others). The repository rule puts a
   one-card test in its set's test module. Move single-card fixtures there and
   update the ledger `fixture` paths; keep cross-card interaction tests where
   their subject card lives.
5. **Closed in `6aa5452f11`.** Six early ledger rows lacked fixture pointers (Cat Collector, Venom
   Connoisseur, Predator Ooze, Steel Hellkite, Fumigate, Exsanguinate). Their
   fixtures are named in the slices above.
6. **Closed in the death slice.** **One lifelink source hitting several recipients is several life-gain events.** The rulings for Exemplar of Light, Sanguine Bond and Vito say "a single creature with lifelink [dealing] combat damage to multiple creatures, players… at the same time… triggers only once." `deal_damage` (`flow/damage.rs`, the `lifelink_controller` tail) calls `commit_life_change` for each damage event, so each recipient becomes a separate gain. Reproduction: an Exemplar of Light and a Hill Giant with lifelink and trample; Savannah Lions blocks the Giant. Life goes 20 → 23 and 20 → 18 as expected, but `turn.history[0].life_gains == 2` and Exemplar gets two counters where the rulings give one. Fix: within one simultaneous damage batch (`event_batch`, combat or a multi-recipient spell), sum lifelink gains per source and commit each source's sum as one gain. Two lifelink sources still make two events. Add this fixture, plus a two-source control, to `fdn.rs` beside Exemplar's test, and record the gain count in its ledger row. The defect predates this batch, but the batch's gain readers (Exemplar, and any later "Nth time" gain) are what expose it.

7. **Closed in `4502508574`.** **A stacked "up to" modal trigger that chose no modes reads as all of
   them.** `trigger_description` (`agent/ability_description.rs`) only
   prefixes "chosen mode N" when `chosen_modes` is nonempty and otherwise
   describes the printed `TriggerModes`. Before the death slice an empty
   choice became `inline_effects = Some([])`. Reproduction: Bumi, King of
   Three Trials enters with no Lesson in its controller's graveyard (X = 0)
   and the trigger is placed after one priority pass. Both viewers' stack
   row reads "Choose up to 0 plus the number of Lesson cards… Put 3 +1/+1
   counters…; or … Scry 3; or … Earthbend 3", `chosen_mode` is `None`, and
   the frame has no program. It resolves to nothing (0 counters), which is
   correct. Fix: a stacked trigger has always chosen its modes. Pass
   whether the choice is made (stack: always; placement: `modes_chosen`)
   and render a made empty choice as "no modes chosen". Add the zero-Lesson
   case to `tla.rs::bumi_chooses_up_to_one_mode_per_lesson_in_your_graveyard`,
   asserting both viewers' text.

Still open beyond findings: recorded Watch playback of a partial
declaration, the full debug and browser gate, frozen pilot compatibility on
the merged preparation path, and Engine Budget comparisons.

## Remaining mechanic batches

Jack Heart, comment a15ea74c-454c-4a55-9b26-fc62e717cac6 (and local sync steer d8007389-a447-4731-a7fc-2ed791fd4e56):

> Stop doing one card per coverage-working loop. In realign, group the remaining in-scope ledger rows by shared mechanics, enumerate the cards in each batch, and select a whole batch for implement. Implement every feasible row in that batch with its directed checks before handing off to compress and review. Review the batch together. Record completed rows and concrete blockers. If one row is blocked, continue with independent rows. Apply this to the current run and subsequent passes.

All 81 primary FDN rows (47 `event_history`, 34 `context_queries`) have
definitions and native fixture pointers, with no `pending` note (checked
against `fdn.jsonl` at `a1f79ee437`). Fiendish Panda and Heartfire Immolator
satisfy the additional exact-source witnesses. Cost-primary rows citing either
dependency: 15 of 17 delivered; Quilled Greatwurm (`play_permissions` plus
distributed counter removal) and Ovika (composite mana-and-life ward) carry
recorded blockers. Keep the established cross-set and integration boundaries.

### Implemented slice · Power at attack, combat and resolution

Seven primary rows and the cost-primary Ruby row now link to their first-printing modules and directed tests. The thirteen directed cases in the cards target cover all eight cards, wrapped attack and hit members, both trigger limits, the pending condition-only projection and catalog tokens.

- Courageous Goblin and Ruby qualified when a Gray Ogre reached power four before declaration. Moving that power increase until after declaration left both attackers at their printed power. Removing it from the qualified branch kept Goblin at 3/2 with menace and Ruby at 3/4. Ruby's separate mana test (`fdn::power_batch_ruby_taps_for_either_color`) restored the same choice and added one red or one green.
- A 4-power Hornbeetle needed another 4-power creature. Returning that other creature to hand before resolution kept Hornbeetle's counter count at two. Surrak counted 5 + 3 − 1 as seven. At eight, all three creatures were legal haste targets. Returning to seven before resolution left the selected Ogre without haste.
- Ashroot Animist and Halana and Alena each offered the other two creatures. The Ogre's old object precondition was rejected after exile and return. Growing the source by two changed the pending power to six and four respectively, visible to both seats even after source reentry. Ashroot pumped and granted trample. The Partners put counters on the Ogre and granted haste. A separate printed trigger whose condition alone read source power showed three, then two, and its life gain failed the resolution recheck.
- Bulk Up read the Ogre at −2 or 3 on resolution and doubled it to −4 or 6. Spending {4}{R}{R} from the same turn's graveyard doubled the result again and exiled the spell. Bite Down used the Ogre's updated power of three to take a five-loyalty planeswalker to two. Reentering either target before resolution left loyalty at five. The existing Felling Blow, Heartfire Immolator and Alesha controls passed.
- Wrapping the doubled attack listener in `ActiveIf` gave two occurrences per attacker. A granted, wrapped, once-per-turn listener bound the counter-bearing member and fired once across both combats. Suki under Katara created one Ally across two departures. Two wrapped combat hits of two and three under Katara placed four and six counters on the respective creatures. Two Ogres that received the same static ability simultaneously each granted Katara one trigger. Both fired once. The turn record distinguishes the grantor, timestamp and ability slot.
- The final native selection passed all 19 checks, including the layer comparison after tapping and untapping, last-known mana value, source-name filtering, catalog conformance and the Clone replay hash. Academy's `svelte-check` reported zero errors and warnings.
- The rebuilt CPython 3.13 wheel passed ten focused Python checks. Both viewers saw pending power in the v11 stack, Academy and semantic text. The named layout, effect execution, Torch/MLX parity in both shape modes, and the bundled v9 and v11 pilot replays passed. The optional u1952 replay skipped because its checkpoint was unavailable.

### Implemented slice · Life events, totals, payment thresholds, and Findings 1–2

Six primary rows (Vanguard Seraph, Bloodthirsty Conqueror, Exemplar of Light, Stromkirk Bloodthief, Bloodtithe Collector, Hidetsugu's Second Rite) and the cost-primary Sanguine Indulgence and Ayli rows link to their first-printing modules and directed tests. The FDN ledger has 38 of 81 primary rows.

Shared changes:

- One event amount. `triggering_damage` became `triggering_amount` everywhere (pending trigger, stack, frame, continuation, v11 fact slot 16 now named `triggering_amount`, text, Academy pending line). The name was branch-only, so `EffectValue::TriggeringDamage` was renamed rather than kept. `TriggerCondition::LosesLife { who }` matches a committed decrease and binds `old - new`; combat hits still bind their own amount.
- `TriggerCondition::PutsCountersOn { who, kind, subject }` sits beside the legacy `PutsCounters`.
- `ObjectSelector::Paid(How)`: property reads use each paying object's LKI. A pending activation publishes the first amount it reads from what paid (`StackObjectData.paid_amount`, appended v11 stack column 42 `paid_amount` in tenths, description suffix "from what paid its cost: N", Python getter).
- `StaticAbility::CostsLess { amount, condition }`; `Player.starting_life` (omitted from the record at 20) with `EffectValue::LifeAboveStarting`; `ActivatedAbilityDefinition.condition` enforced in `check_activation_terms`, which mana-ability offers and the autotapper probe now call instead of their own `own_turn_only`/life checks.
- Finding 1: `Modification::Firebending(n)` pushes an ordinary `Ability::firebending(n)` trigger; `granted_firebending_triggers` is deleted. One and two Sozin's Comets under Katara give two and four 5-mana triggers; removing one leaves 15.
- Finding 2: the heuristic reads live `attacking` and keeps only recorded blocks; the `CombatAttackersDeclared` branch is gone. A vigilant or entered-attacking unblocked creature is pumped for lethal.

Checks: the debug cards target (887) and engine lib (359), rules/content-pack/engine/hash targets, `search_state_contract` and `canonical_replay_tests` passed. The Clone state hash moved with the eight new catalog ids. After the rebuild, the focused Python selection passed (120 passed, 1 skipped optional checkpoint).

Compression (`86d1c65815`) replaced the per-reader contribution filters with `PaymentReceipt::paid_as(How)` and dropped a `RefCell` from the paid-cost amount search.

**Conformance evidence is stale, not drifted (behavior review at `67f9d3ee34`).** All 14 debug `frozen_*` cases and the inventory test fail at this head, not only `sacrifice-162/163`. The evidence was last recorded at `5be4b4d5ba`; the power and life batches since added cards. Each `ReplayCase` embeds the default content pack's `content_digest`, and its state and action-space digests hash catalog ids, so every trace hash moves when the catalog grows. Dumping each regenerated case's prompts, actions, event boundaries, game-over flags and winners at `5be4b4d5ba` and at this head gave byte-identical sequences (14 cases, 3048 commands). No rules behavior moved. Before publication, re-record at the final head: `cargo run --release --manifest-path managym/Cargo.toml --bin semantic_conformance -- record --root <staged copy>`, then copy its `evidence.json` over `evidence-opening-v1.json` and note the cause ("catalog grew: power and life batches") in the commit.

Not built in this slice:

- **CR 104.4b mandatory-loop draw** (`GameOutcome::Draw` and cycle detection in `flow/tick.rs`). Conqueror's ledger row records it as pending. It needs a loop partner such as Vito, Thorn of the Dusk Rose, which isn't defined yet. It also touches every outcome reader: `winner_index`, Env rewards and Academy. It should be its own slice with that partner card.
- Vanguard Seraph's waiting-viewer privacy during the surveil choice, and Exemplar under a trigger doubler or with CR 122.6 entry counters. The shared machinery has existing controls, but these cards have no directed check.

Findings 3–5 (opening-hand Academy fixtures and relocating earlier single-card tests/ledger pointers) closed in `cf12733e50` and `6aa5452f11`.

### Implemented slice · Creature deaths and turn death history

Eight primary rows (Tragic Banshee, Vengeful Bloodwitch, Needletooth Pack, Slumbering Cerberus, Wardens of the Cycle, Midnight Reaper `grn`, Massacre Wurm `mbs`, Garna `dmu`) and the cost-primary Kalastria Highborn (`wwk`, a new set module) link to their first-printing modules and directed tests. The FDN ledger has 46 of 81 primary rows.

Shared changes:

- Finding 6 closed: damage results carry their lifelink gain and the batch commits one gain per source (`commit_lifelink_gains`). Sire of Seven Deaths trampling through two Lions gives one gain and one Exemplar counter; two Healer's Hawks give two.
- `ObjectLki` records layered `name`, `types`, `subtypes`, `colors`, `is_legendary` and `was_attacking`/`was_in_combat`/`was_tapped`; `was_creature`, `was_land` and `creature_types` became methods. One `PermanentFacts` matcher (`flow/layers.rs::matches_permanent_facts`) answers live permanents and `Game::lki_matches_predicate`. Moved-object subjects and `additional_trigger_times` read LKI, never the printing. Hash contract v27. `retained_departure_predicate_uses_last_known_copy_after_reentry` now asserts the copied type (an Ogre copying a land left as a land).
- `Condition::TriggeringMatches(CardPredicate)` (Garna's "if it was attacking"); legacy `TriggeringHadCounters` kept.
- Stacked and pending trigger frames that read their triggering permanent project its departed LKI (`effect_execution::project_stack_object`, `project_pending_trigger`); the summary says "attacking"/"blocking". Departed `Target::Permanent` references keep their definition.
- `event_player`: dies/leaves/enters/sacrificed bind the controller as it last existed; graveyard triggers keep the owner.
- Printed modal triggers keep `inline_effects: None` and carry `chosen_modes` (stack) / `modes_chosen` (placement). `with_chosen_modes` expands them at resolution, retarget and requirements. This fixed a pre-existing gap: modal triggers skipped the CR 603.4 recheck. A single chosen mode fills `StackObjectData.chosen_mode` (v11 `stack::MODE`) and `ContinuationFrame.mode`; several project their program; descriptions read "chosen mode N".
- `Phyrexian` joined the subtype table; `play::bolt` is a shared fixture primitive.

Checks: debug cards target (899), engine lib (359), rules (188), content pack, engine, match-state hash, scenario, `search_state_contract`, `canonical_replay_tests` passed. The Clone hash in `lea.rs` moved to v27. After rebuilding the cp313 extension, the focused Python selection passed (116 passed, 1 skipped optional checkpoint).

Compression (`e4579800a7`) gave stacked triggers one effects reader, `Game::stacked_trigger_effects` (`flow/resolution.rs`), used by retarget and the continuation projection. Pending triggers keep their own reader in `flow/triggers.rs::trigger_requirements` because their source definition comes from a different place.

Taught: the plan's Banshee text was wrong; the printed card is -1/-1, or -13/-13 after a death. `Scenario` does not skip single-action decisions, so fixtures choose the lone target themselves.

Conformance evidence was re-recorded at hash v30 (`00da7aaf31`); Findings 3–5 closed in the closing slice; Finding 7 was fixed in `4502508574`.

**Compression at the draw/cast slice.** Stack lookups now go through `spell_on_stack` and `stack_index_of(Target)`; X-inclusive mana value is `Object::mana_value_with_x`; one `trigger_context` binds the event object for the intervening-if and the stacked ability; pending trigger text and definitions share `pending_trigger_description`/`pending_trigger_definition`. Finding 7 is fixed there: a modal trigger with zero chosen modes reads "no modes chosen". Debug workspace: 1588 passed; the 15 stale `frozen_*`/inventory cases fail as recorded above, and `compatibility_names_and_content_allocations_are_metamorphic` fails (state hash diverges at step 47 after renaming objects). It fails identically at `1ca881442e`. Cause (read at `f11b0b882c`): the test rewrites every `Object.name` (`conformance.rs::rename_compatibility_cards`), and the death batch's v27 `ObjectLki` serializes the layered `name` (`state/src/game_object.rs:307`) into the hashed state, so the first departure diverges. `SpellLki` carries no name. Remaining: keep the departed name out of hashed identity (it follows from `source_base` and copy provenance, or hash the definition key instead), or rename it in the test alongside `Object.name`; decide which before re-recording evidence. Focused Python: 116 passed, 1 skipped.

**Death batch behavior review at `4581548f6f`**

The working phase for the death batch is satisfied except Finding 7, a text-only defect in a TLA card that the modal-trigger change exposed. These passed at this head, all in debug: the cards target (899), the `managym` lib (359), rules (188), `match_state_hash_tests`, `search_state_contract` and `canonical_replay_tests`. The installed extension predated the compression commit `e4579800a7`. After rebuilding it at this head and replacing it atomically, the five-file focused Python selection passed (116 passed, 1 skipped optional checkpoint).

- All nine definitions match their Scryfall Oracle text, cost, types and stats. Banshee and Needletooth read death history as the ability resolves or as the intervening "if" requires (CR 608.2h, 603.4). Cerberus's `DoesntUntap` blocks only its untap step, not its own untap effect. Wardens rechecks its intervening "if" after choosing a mode. Reaper and Bloodwitch include their own deaths (CR 603.10a), and Reaper's damage comes from its LKI source. Massacre Wurm's "that player" is the stolen creature's controller, not its owner. Garna reads `was_attacking` from LKI and skips its own death. Kalastria targets before offering {B} (CR 603.3d) and reads a creature's last known types.
- One matcher (`matches_permanent_facts`) serves live, layer-context and departed permanents. Departed supertypes and mana value come from `source_base`, the same printing a live copy reads. `recorded_object_matches_predicate` and `entry_matches_predicate` remain separate printing-level paths for non-battlefield appearances, which is correct for their callers. `ObjectLki` is `Serialize`-only, so no stored snapshot reader breaks. The hash moved to v27.
- `event_player` strips trigger wrappers through `condition.event()`. Entering binds the entry controller and departure binds the controller from LKI (`zones.rs` `event_controller`). `PutIntoGraveyard` keeps the owner.
- Departed trigger context projects to both viewers. A creature dying is public, and a face-down permanent is revealed as it leaves (CR 708.9), so no new disclosure was found.
- Fixed during review: the Exemplar ledger row still carried Finding 6 as `pending`; it was removed (its `lifelink_gains` fixture pointer stays).
- Payment did not change in this batch. Kalastria's {B} uses the existing optional-payment path, so no competing-spend case is new.
- Not covered by this batch's tests: granted modal triggers are still offered no modes (`unchosen_trigger_modes` returns `None` for inline effects), as before. No granted modal trigger exists in the catalog.

#### CR 104.4b mandatory-loop draw

Delivered in the closing slice (`b416a210b2`).

### Implemented slice · Draw listeners, cast triggers, and spell LKI / counter results

Built in `d061f9a14a`, `5d840eeb12` and `7cdf10646c` on hash contract **v28**. Eight primary FDN rows (Scrawling Crawler, Dictate of Kruphix `jou`, Brineborn Cutthroat `m20`, Mindsparker `m14` (new module), Gnarlback Rhino `m20`, Mold Adder `m10` (new module), Firespitter Whelp, An Offer You Can't Refuse `snc`), the cost-primary Archmage of Runes, Painful Quandary `som` and Ramos, Dragon Engine `c17`, and EOE Unravel link to their first-printing modules and directed tests. The FDN ledger has 54 of 81 primary rows; Unravel's EOE row carries the same implementation record.

Shared changes:

- `PlayerSelector::EachPlayer` (APNAP, CR 101.4). `Effect::ForPlayer` runs once per selected player, each run ending with a `RestoreController` cursor naming the next player; a suspended choice inside one run restores with no other frame state.
- `SpellLki` (`state/src/stack_object.rs`) is written to `GameState::spell_lki` where a spell leaves the stack (`zones.rs::commit_zone_move`), copies included. `Game::spell_info` answers live and departed spells and replaced `ResolutionTarget.spell` / `SpellPayment`. `TargetController` answers spells; `SpellManaSpent`/`SpellManaValue` (target or triggering), `target_property` and the legacy `TriggeringManaValue` read it, so a departed X spell keeps its stack mana value (previously the card it became). `ObjectProperty::Colors` reads live, departed and spell colors. `TriggeringSpellMatches` still uses `recorded_object_matches_predicate`, which already answers departed spells.
- `ContextRead::TriggeringObject` (renamed from `TriggeringPermanent`) covers spell readers (`TriggeringManaValue`, `TriggeringSpellMatches`, `CopyTriggeringObject`, `CopyTriggeringSpellForEach`); `asks_triggering_spell` is deleted. A cast trigger that reads its triggering object binds `Target::StackSpell`. Departed spells project through `effect_execution` (`last known spell`: mana value, colors, spent) and `continuation::target_reference` keeps their definition.
- `TriggerCondition::Casts { targets: SpellTargetFilter }` (`Any`, omitted when serialized, or `TargetsThis`); legacy `SpellTargetsOnlyThis` stays. `Condition::Not` with `Condition::not`.
- A mana ability's `remove_counters` cost is now paid (`complete_mana_upfront`), checked before it is offered, planned as one activation, and counts as a side effect. Before, Ramos added mana without removing counters.
- `Fungus` and `Merfolk` joined the subtype table. Search-state authority covers `spell_lki`; the Clone hash in `lea.rs` moved to v28.

Checks, all debug: cards target (912), `managym` lib (359), rules (188), `match_state_hash_tests`, `search_state_contract`, `canonical_replay_tests`, content pack, engine, scenario. After rebuilding and atomically installing the cp313 extension, the focused Python selection passed (116 passed, 1 skipped). Both viewers' departed-spell projection is checked natively (`som::a_countered_x_spell_keeps_its_stack_mana_value_for_its_cast_trigger`, `c17::ramos_…`): equal, nonempty, and the acting text reads "Last known spell Exsanguinate: mana value 5".

Not covered (see `scratch/questions.md`): a noncreature Dragon for Firespitter's overlap, a multi-target spell for Rhino, spell copies for Unravel and An Offer, and a stolen spell for An Offer.

**Draw/cast batch behavior review at `b2de283336`**

The working phase for this batch is satisfied except Finding 8 (text only). At this head, all 16 directed debug cases passed in the cards target: the twelve card fixtures, the countered-X witness and the three Bumi cases. The eight FDN rows and Unravel's EOE row name existing first-printing sources and fixture functions. Python evidence is the slice's run (116 passed, 1 skipped on the rebuilt cp313 extension); this review did not rebuild the extension.

- The definitions match the printed rules. Crawler draws in APNAP order (CR 101.4), and each opponent draw drains once per Crawler, including draws inside its own trigger. Dictate is the triggering player's draw. Cutthroat's `ActiveIf(Not(YourTurn))` is checked only when the trigger fires, which matches its wording (no intervening if). Rhino triggers once for Fancy Footwork targeting both Rhino and an Ogre, and not for an opponent's spell. Mindsparker requires both color and card type. Whelp's single `any_of` predicate gives a Dragon creature spell one trigger. Archmage's discount applies only to its controller's spells (`play.rs` scans that player's battlefield). Mold Adder's "may" uses the catalog's zero-cost `OptionalPayment` convention, and Painful Quandary's discard uses the `may_discard` hand-to-graveyard convention, chosen by the casting opponent.
- Spell LKI: Ramos counts a countered Foggy Swamp Spirit Keeper's two colors. Unravel compares spent mana with stack mana value (CR 608.2h): a Tolarian Terror discounted by four draws, while a full-price Terror and an X = 2 Exsanguinate do not. An Offer pays the spell's controller even when the spell can't be countered (Rending Volley). A review probe (not kept) confirmed CR 608.2b: when the targeted Bolt is countered first, An Offer does nothing, and neither player gets a Treasure.
- Payment: Ramos's counter-removal mana ability is never auto-tapped. `automatic_choice_taps` excludes `mana_activation_has_side_effects`, which includes `remove_counters`, so Auto never spends counters silently. Ramos isn't offered below five counters and is offered once each turn.
- Constraint, not this batch's defect: `on_step_start` emits `StepStarted { Draw }` on the first player's turn 1, even though CR 103.8a skips that step. So a "beginning of each player's draw step" trigger (Dictate, and the existing `fut.rs` card) would fire on that turn. In ordinary games no such permanent can be on the battlefield before that step; only constructed positions reach it. The concept review kept turn 1's step schedule for frozen replays. Any fix belongs with that schedule.
- `scratch/questions.md` calls Rhino's multi-target case untested; Fancy Footwork covers it.

8. **Closed in the departed-name and damage slice:** cast triggers named their spell "that permanent".


### Implemented slice · Departed-name LKI repair and damage recipients, amounts and results (Hash v29)

Built in `2279c515b2` on hash contract **v29**. Four primary FDN rows (Goblin Negotiation and Niv-Mizzet, Visionary `fdn`; Gateway Sneak `rna`, new module; Trygon Predator `dis`, new module) and the cost-primary Lathril, Blade of the Elves (`khc`, new module, with `token.green_elf_warrior`, the green Lorwyn printing beside TLA's green-white one) link to their first-printing modules and directed tests. The FDN ledger has **58 of 81** primary rows. Quilled Greatwurm stays pending with its blocker in the ledger: its trigger watches combat damage to any recipient, but `DealsCombatDamage` fires only on `CombatDamageToPlayers`; its graveyard cast needs `play_permissions` and distributed counter removal.

Shared changes:

- `ObjectLki.renamed: Option<String>` replaces `name`; `snapshot_current_permanent` fills it only when the layered name differs from the object's copiable name. `Game::lki_face_printing` (copy/back face, else retained base) and `Game::lki_mana_printing` (front face, CR 712.8e) serve `lki_matches_predicate` (via `PermanentFacts::departed`) and `departed_property`. The departed Clone bug was real: a Clone that left as a Gray Ogre matched Clone's mana value 4 (`resolution_tests::departed_copy_matches_predicates_as_the_copied_face`).
- Namesake matching requires a nonempty name (CR 201.2a) in `matches_permanent_facts`.
- `GameEvent::DamageDealt { combat }` (omitted when false). `Game::source_controller` (now `pub(crate)`) reads a live then departed spell's controller before owner provenance, so lifelink and damage attribution follow a stolen spell.
- `TriggerCondition::DealsNoncombatDamage { source: Actor, to: Actor }` on `EventKind::DamageDealt`; `event_amount` and `event_player` bind the damage and the damaged player, so `TriggeringAmount` and "that player" need no new columns. Text: "Whenever a source you control deals noncombat damage to an opponent". Observability guard and witness added; `docs/engine.md` describes it and the departed-face readers.
- `EventBinding::Damage` always sets `frame.excess_damage` (0 when none).
- Finding 8 fixed: `name_triggering_object` rewrites `ObjectSelector::Triggering`'s "that permanent" to the event's noun (`spell` for cast triggers, `card` for graveyard arrivals) in trigger and granted-trigger text. Ramos's stack text is asserted in `c17.rs`.
- `Vedalken` joined the subtype table. The Clone hash in `lea.rs` moved to v29.

Checks, all debug: cards target (917), `managym` lib (360), rules (188), `match_state_hash_tests`, `search_state_contract`, `canonical_replay_tests`, content pack, engine, scenario, authored match, and `compatibility_names_and_content_allocations_are_metamorphic` (passes again). The 14 `frozen_*` cases and the inventory test still fail only as stale evidence (catalog grew; re-record at the final head as described above). After rebuilding and atomically installing the cp313 extension, the focused Python selection passed (116 passed, 1 skipped).

Witnesses: Niv-Mizzet draws 3 for a Bolt to the opponent, 0 when the Bolt's controller is changed to the opponent (stolen spell), 0 to its controller or a creature, 1 for a sacrificed Fanatical Firebrand (departed source), and 0 for its own combat damage. Goblin Negotiation makes 3 Goblins at X = 5 on a 2/2, 0 at X = 2, and 0 when a later damage instruction to a player follows (the stale-excess control). No catalog card prevents damage to a creature, so "prevented excess" is covered by that later-instruction control rather than by real prevention.

**v29 batch behavior review at `21929d9111`**

The working phase for this batch is satisfied; one text nit (Finding 9) moves with the next slice's damage-trigger unification. All in debug at this head: the five card fixtures, `departed_copy_matches_predicates_as_the_copied_face`, Ramos's text check, Razor Rings and both Last Agni Kai excess controls, the observability module (92 selected), `compatibility_names_and_content_allocations_are_metamorphic`, `search_state_contract`, `canonical_replay_tests` and `match_state_hash_tests`. Python evidence is the slice's run on the rebuilt cp313 extension; this review did not rebuild it.

- Definitions match Oracle text. Goblin Negotiation measures excess against remaining toughness after marked damage, with deathtouch making one damage lethal (CR 120.4a, `excess_for_damage`). Niv-Mizzet triggers once per damage event to an opponent. Zero damage (`commit_damage` returns before emitting) and damage prevented by protection emit no `DamageDealt`, so no zero-card trigger exists (CR 120.8). Lathril's ten Elves exclude itself and may be summoning-sick tokens (CR 302.6 restricts only {T}). Trygon's "that player" is the opponent in two-player play, and its "you may" uses the zero-cost optional payment. Gateway Sneak's two triggers are independent.
- The per-instruction excess reset leaves both existing readers correct. Razor Rings has one damage instruction. Agni Kai's fight measures its victim, and a fight that does not happen leaves 0.
- `event_player` and `event_amount` now bind `DamageDealt` to a player. Only `DealsNoncombatDamage` listens to `EventKind::DamageDealt`, so no existing trigger's "that player" changed.
- `ObjectLki.renamed` compares the layered name against the object's current copiable definition, so a copy or a back face leaves it `None`. No battlefield face-down permanents exist yet (face-down is exile-only), so the doc comment's empty-name case is prospective.
- Probe (not kept): Niv-Mizzet over a Bolt to the opponent. Both viewers see the same stacked text, and the continuation frame carries `triggering_amount` 3.
- Payment: Lathril's tap cost uses the existing `tap_cost_candidates` path, and Auto spends no mana for it. No mana payment changed.

9. **Nit, open.** The `TriggeringAmount` fact's native description (`agent/observation_layout.rs:826`, "combat damage dealt or life lost") and the effect text "(damage dealt or life lost)" do not mention noncombat damage, though Niv-Mizzet now binds it. Correct the layout description when `DealsDamage` replaces `DealsNoncombatDamage`. If the description enters a layout digest, record that.

Shape review at `cf15cce2e7` identified two structural seams (`name_triggering_object`'s blanket string rewrite and `DealsNoncombatDamage` splitting damage triggers by variant name). The post-v29 concept review at `d7ef237481` (`scratch/concept-review.html`) folds both repairs directly into the next slice alongside the live-board, name-predicate and turn-departure unifications.

### Implemented slice · Damage-trigger unification, triggering-noun threading, live-board counts, name predicates and turn departures

Built on hash contract **v29** (no serialization change; only the Clone witness in `lea.rs` moved, because its hash includes catalog-order definition ids). Ten primary FDN rows (Hare Apparent and Ingenious Leonin `fdn`; Brass's Bounty `rix`; Krenko, Mob Boss `m13`, new module; Dwynen's Elite `ori`; Inspiring Call `dtk`; Leonin Vanguard and Volley Veteran `m19`, new module; Archway Angel `rna`; Dragonmaster Outcast `wwk`), the cost-primary Cryptic Caves (`m20`) and Ghalta, Primal Hunger (`rix`), and the cross-set Plasma Bolt (`eoe`) and Michelangelo, Game Master (`tmt`, new module) link to their first-printing modules and directed tests. New tokens: `token.rabbit` (first printing tCLB) and `token.dragon_5_5` (the plain 5/5 flier, first printing PR2). `Ninja` joined the subtype table. The FDN ledger has **68 of 81** primary rows.

Shared changes:

- `TriggerCondition::DealsDamage { kind: DamageKind, source: TriggerSubject, to: DamageRecipientFilter }` replaces `DealsNoncombatDamage`; `subject_matches_damage_source` (`flow/triggers.rs`) matches the source live, by LKI or as a recorded spell, with `Game::source_controller`. `event_amount` binds every `DamageDealt`. Niv-Mizzet moved to it unchanged in behavior; `rules_tests::engine_events::combat_damage_to_a_creature_triggers_deals_damage` is the combat-to-creature witness, and the observability witnesses list both forms. Finding 9 is fixed: the `TriggeringAmount` fact describes "damage dealt or life lost" (not in any layout digest; the pinned-pilot test passed).
- `name_triggering_object` is gone. `condition_text`, `selector_description`, `value_description`, `effect_description`, `move_description`, `copy_description` and the `_in` forms of `effects_description` / `continuous_description` take the triggering noun; the public wrappers pass `"permanent"`. A granted trigger names its own object (`ability_description::tests::each_trigger_names_its_own_triggering_object`).
- `CardPredicate.named` (nonempty equality, CR 201.2a) in `matches_with_types`, `matches_permanent_facts`, `ReadSet::predicate` (`NAME`) and `predicate_description`; both matchers share `CardPredicate::accepts_name` (`4cfa46c6e6`).
- `EffectValue::Scaled { factor, value }`: Archway Angel's three Gates are one gain of six (`life_gains == 1`).
- `Condition::PermanentLeftThisTurn { who, nonland }` and `Condition::WarpedThisTurn { who }` read existing turn history per seat through `TurnState::any_history`.

Not built, by choice (recorded in `scratch/questions.md`): `Game::record_warped_cast` (nothing writes the warp flag until casting permissions lands; Plasma Bolt's test sets the history flag), and binding `DamageDealt`'s source as the triggering object (no card in this slice reads it; Quilled Greatwurm will). Elvish Archdruid is blocked: `ActivatedAbilityDefinition::mana_choices` accepts only `AddMana` with a fixed amount of 1, so "Add {G} for each Elf" would use the stack; its ledger row records the blocker. Quilled Greatwurm's blocker now names the source binding, `play_permissions` and distributed counter removal.

Checks, all debug: cards target (931), `managym` lib (361), rules (189), `match_state_hash_tests`, `search_state_contract`, `canonical_replay_tests`, content pack, engine, scenario, authored match and `compatibility_names_and_content_allocations_are_metamorphic`. After rebuilding and atomically installing the cp313 extension, the focused Python selection passed (116 passed, 1 skipped). The 14 `frozen_*` cases and the inventory test were not rerun; they remain stale evidence to re-record at the final head.

**Live-board batch behavior review at `ad75001de9`**

The working phase for this batch is satisfied; no new finding. Finding 9 is closed. All in debug at this head: the cards target (931), the `managym` lib (361), rules (189, 2 ignored), `search_state_contract`, `canonical_replay_tests`, `match_state_hash_tests`, content pack, engine, scenario and authored match. The batch's 14 card fixtures, Niv-Mizzet and the observability module (98 selected) passed separately. Python evidence is the slice's run on the rebuilt cp313 extension; this review did not rebuild it. The ledger reads 68 of 81 primary rows with source and fixture pointers.

- Definitions match the Scryfall Oracle text in `scryfall_cards.json`. Leonin Vanguard, Dragonmaster Outcast and Dwynen's Elite use CR 603.4 intervening-if conditions, so the condition is checked again on resolution. Cryptic Caves counts itself among the five lands, and its "activate only if" is checked on activation. Krenko and Volley Veteran count themselves as Goblins. Ghalta sums power (negative power reduces the total) and clamps the reduction at zero before reducing generic mana only (`flow/play.rs`). Inspiring Call's `with_counter` reads +1/+1 counters only (`layers.rs:1535`). Its draw and its grant select the same set, because nothing can change between the two instructions. Ingenious Leonin checks Cat after placing the counter. Archway Angel is one gain event.
- Michelangelo and Plasma Bolt read the existing per-seat counters written in `flow/zones.rs` under the departing object's last controller, so they follow "under your control" and "any player". Those counters and `warped_this_turn` were already v11 `player_turn_state` columns, so the policy input needs no new column. Warp is still never written by a real cast; that end-to-end obligation stays with casting permissions, as recorded.
- `DealsDamage` gates on kind, recipient and source as the plan states. `event_amount` binds every `DamageDealt`. Only `DealsDamage` listens to that event, so no other trigger's amount changed. `subject_matches_damage_source` falls back to `recorded_object_matches_predicate` for a spell. Only empty predicates use it today, so a "red source" style predicate over a live stack spell is unexercised. Add a witness when a card needs one.
- `EffectValue::Scaled` is handled by every evaluator (`resolution.rs`, `layers.rs` comparison and read sets) and by the `reads*` and `is_resolution_only` helpers. `CardPredicate.named` reaches learning tokens as a hashed `Text` token through the serde serializer, as other printed strings do.
- Probe (not kept): with two Hare Apparents on the battlefield, a third enters and its trigger is placed. Both viewers' stack text reads "…where X is the number of each other creature named Hare Apparent you control". Moving the entering Hare to the graveyard before resolution still makes two Rabbits (CR 201.5c, 608.2h). The "number of each" wording is the existing count template; it is not new in this batch.
- Constraint, not a defect: the engine has no deck copy limit, so Hare Apparent's deck-construction line has nothing to override. A future four-copy rule needs that exception.
- Payment: no payment path changed. Cryptic Caves' sacrifice-and-tap cost and Ghalta's discount use existing paths.

### Implemented slice · Mana-ability split, `DealsDamage` source binding, all-zone `*/*` CDAs, move results and multi-player mill

Built in `f5e7244bf4` and `965b0e0ea1` on hash contract **v29** (no serialization or state change; only the Clone catalog-order witness in `lea.rs` moved). Twelve primary FDN rows (Kiora, the Rising Tide, Crypt Feaster and Revenge of the Rats `fdn`; Enigma Drake `akh`, new module; Crusader of Odric `m13`; Consuming Aberration `gtc`; Stroke of Midnight `woe`; Feed the Swarm `znr`, new module; Scavenging Ooze `cmd`, new module; Cemetery Recruitment `emn`; Dread Summons `c15`, new module; Maelstrom Pulse `arb`, new module) and the cost-primary Elvish Archdruid (`m10`) link to their first-printing modules and directed tests. New tokens: `token.rat` (tSHM), `token.zombie` (tUGL) and `token.scion_of_the_deep` (tFDN). `Horror` and `Octopus` joined the subtype table. The FDN ledger has **80 of 81** primary rows with source definitions; Mazemind Tome remains.

Shared changes:

- `ActivatedAbilityDefinition::is_mana_ability` (CR 605.1a classification, including counted `AddMana`, `AddOneColorManaValue`, mana-only `If` and fixed `Modal`) and `has_dynamic_mana_output` are separate from `mana_choices` (finalized outputs). Layer 6 `lose_nonmana_abilities` keeps every mana ability, `mana_suffix` reads the classification, and `finalize_characteristics` materializes the leading output beside any suffix; `current_mana_output` recurses into a branch's counted output. The Archdruid fixture's "nonmana abilities lost" case is the Layer 6 control (it failed before: the dynamic ability was dropped).
- `trigger_context` binds `DamageDealt`'s source (live permanent, its LKI permanent, else the spell) as the triggering object: `rules_tests::engine_events::deals_damage_binds_source_as_triggering_object`.
- `StaticAbility::CharacteristicPt(PowerCda)` and `PowerCda::GraveyardMatchingBy { who, predicate }`. `PrintedCharacteristics::pt_cda()` replaces `power_cda()`; layer 7a `Operation::Cda { cda, sets_toughness }`; `Game::object_pt` (with `object_power` reading it) serves stack and card rows, so off-battlefield toughness follows the CDA in the policy input.
- `Effect::MillCards.player`, `Effect::MillUntil` (one `CardRevealed` per card, one batch), shared `Game::mill_objects`; `EffectValue::Objects.in_zone`; `ObjectSelector::TargetAndSameName` (target's layered name through `CardPredicate.named`, so CR 201.2a holds; nothing once the target is illegal).
- Scavenging Ooze uses the existing `Condition::TargetCardExiled` rather than an `in_zone: Exile` duplicate (recorded in `scratch/questions.md`). No catalog card exiles a milled card by replacement, so Dread Summons' replacement control waits for one.

Checks, all debug: cards target (944), `managym` lib (361), rules (190, 2 ignored), `match_state_hash_tests`, `search_state_contract`, `canonical_replay_tests`, content pack, engine, scenario, authored match, `conformance_tests` (Scryfall rows for all 13 cards) and `compatibility_names_and_content_allocations_are_metamorphic`; the workspace's test targets compile. After rebuilding and atomically installing the cp313 extension, the focused Python selection passed (116 passed, 1 skipped). The 14 `frozen_*` cases and the inventory test were not rerun; they remain stale evidence to re-record at the final head.

Next: the closing slice below.

### Implemented slice · State triggers, counter costs/LKI, untap events, CR 104.4b draw, Findings 3–5

Built in `6939044ee7` (state triggers, Mazemind Tome), `0d00e9116a` (counter property, Ravenous Amulet), `eed6c2495a` (Mentor of the Meek, Immersturm Predator), `e5b5bdce84` (untap events, Fishing Pole), `b416a210b2` (CR 104.4b draw), `6aa5452f11` (Findings 4–5) and `cf12733e50` (Finding 3). The FDN ledger has **81 of 81** primary rows with source definitions and fixtures, no `pending` note left (Bloodthirsty Conqueror's loop fixture closes it). Cost-primary rows citing `event_history`/`context_queries`: **15 of 17** delivered; Quilled Greatwurm (`play_permissions` + distributed counter removal) and Ovika (composite mana-and-life ward) carry blockers. New: `m21.mazemind_tome`, `fdn.ravenous_amulet`, `fdn.fishing_pole`, `isd.mentor_of_the_meek`, `khm.immersturm_predator`, `token.fish` (tWHO); `Book` joined the subtype table; a new `thb.rs` test module. Hash contract **v30**: untaps now add `PermanentUntapped` to the hashed event ledger.

Shared changes:

- `TriggerCondition::State(Condition)` and legacy `NoCounters` both index under `EventKinds::STATE_TRIGGER`; `check_state_triggers` asks either, and `state_trigger_waiting` covers pending, placement and stack (CR 603.8). `ActivatedAbilityDefinition.add_counters` is paid as `CostComponent::AddCounters` with the other costs (and on the mana path). `CounterKind::{Page, Soul, Bait}`.
- `ObjectProperty::Counters(kind)` reads live, departed (LKI) and layer boards. `ContextRead::SourceLki` (`Effect::reads_source_lki`, `Ability::reads_source_lki`) makes `project_stack_object` retain a departed source's LKI, so both viewers' stack frames carry the sacrificed Amulet's counters; the existing LKI summary text is unchanged (see questions).
- One untap path: `Game::untap_permanent` (untap step, other-untap-steps, effects) returns the transition and emits `GameEvent::PermanentUntapped`; `Permanent::untap` reports whether a stun counter replaced it. `TriggerCondition::BecomesUntapped`, `TriggerSubject::EquippedCreature` (shares attachment matching), `ActivatedAbilityDefinition.tap_grantor` (`CostComponent::TapGrantor`, `Game::can_tap_grantor` at offer and payment). Fishing Pole's "remove a bait counter. If you do" is an `If` over `SourceCountersAtLeast` (see questions), not a new effect.
- `GameOutcome::Draw { mandatory_loop }`; `tick_priority`'s `MandatoryLoopWatch` declares it after 64 automatic resolutions with no new low in stack depth, library size or a losable player's life and no new poison high (cap 10,000). Env reports `draw_reason`. Academy `orientBoard` takes the recorded winner, so a draw leaves both seats unwon.

Checks: the debug workspace under nextest (`--profile ci`, 1649 tests) failed only on stale evidence before re-recording (14 `frozen_*` cases and the inventory) plus three pins this slice moved: the heavy fixture's event count (558 → 617, exactly its 59 `PermanentUntapped` events, which also fixed both `search_state_contract` cases), the Clone v30 witness in `lea.rs`, and `definitions.rs` (a state trigger's condition is a source question). Semantic-kernel w17-v6 re-recorded in `00da7aaf31`: 14 cases, 3049 commands and every outcome unchanged; `semantic_conformance_tests` 20/20. The opening authored-match authority (193 decisions, only digests and the catalog listing moved) and live/headless parity receipt were refreshed in `03deb316db`; authority and parity tests 8/8. After rebuilding and atomically installing the cp313 extension: the focused selection plus pinned pilot and the Finding 3 fixtures passed (124 passed, 1 skipped). `test_public_commitment_parity.py` fails before and after this slice for an older reason (see questions).

**Closing-slice behavior review at `2586242711`**

The working phase for this slice is **not satisfied**: Finding 10 below (a false CR 104.4b draw) and the three concept-review mismatches, all reproduced, are open. The rest checked out. Debug at this head: the seven closing-slice card fixtures (Mazemind Tome, Ravenous Amulet, Fishing Pole, Mentor of the Meek, Immersturm Predator, Conqueror loop) and the two `mandatory_loop_tests` passed. With the installed cp313 extension (built 06:21, before the behavior-neutral compression `a1f79ee437`), the eight Finding 3 Academy fixtures passed (`test_semantic_boundary.py`, `test_combat_preview.py`, `test_incinerating_blast.py`). The probes below were not kept.

- Findings 3–5 are closed. All 81 primary rows name a source and a fixture whose test function exists. The five rows still pointing at `fdn.rs` with a non-FDN source (Ruby, Ancestor Dragon, Jazal, Storm Fleet Spy, Halana and Alena) are multi-card interaction fixtures, which the rule leaves with their subject.
- Fishing Pole matches "remove a bait counter. If you do": `SourceCountersAtLeast` reads only the live source (`statics.rs` `source_permanent`, not LKI), so the trigger makes no Fish when the Pole has no bait or has left the battlefield. A stun counter that replaces untapping emits no `PermanentUntapped` and makes no Fish (fixture). Ravenous Amulet's stack frame keeps the sacrificed source's three soul counters for both viewers and drains 3.
- **Concept-review item 1 reproduced.** The Conqueror + drain + Herald loop with one Mountain and a Lightning Bolt in the opponent's hand did not end after 5,000 `pass_priority` calls (opponent at −2482 life). Without the instant and with `skip_trivial = false`, it did not end either (−1232). The watch is local to one `tick_priority` call.
- **Concept-review item 3 reproduced (focus/description).** With two Fishing Poles equipped to one Gray Ogre, priority offers two granted activations, both with `description = None` and `focus = [ogre]`. They tap different Poles but are indistinguishable to a human, LLM text and the action focus. The `reserve_announced_payment` gap and `state_trigger_waiting`'s missing `inline_effects` check are found by reading the code only: no catalog grantor taps for mana or waterbends, and no catalog effect grants a trigger to Mazemind Tome.
- **Concept-review item 2 confirmed by reading.** `normalize_observation_to_hero` (`trace.py:366`) negates `won` after a side swap, so a drawn game whose last trace observation is from the opponent's perspective reports `won: true` in trace replays and `recording.py::person_board`. `server.py::viewer_view` has the same negation, but nothing outside tests calls it except `hero_view`.

10. **A finite automatic chain that consumes an untracked resource is drawn as a mandatory loop.** `LoopProgress` tracks only stack depth, library sizes, poison and losable life. Any other resource a finite chain uses up (counters, permanents, hand, graveyard) counts as no progress. Reproduction (probe, debug `cards`): Bloodthirsty Conqueror; a Gray Ogre with 100 +1/+1 counters and an injected "whenever you gain life, if this has a +1/+1 counter, remove one and gain 1 life"; empty hands; `skip_trivial = true`; cast Lightning Bolt at the opponent and pass. The chain must end after 100 resolutions. Instead the game ends `Draw { mandatory_loop: true }` with 37 counters left, after 64 resolutions without a new extreme. CR 104.4b applies only to a loop "with no way to stop". Required fix: declare a loop only when the rules-relevant state returns to an earlier fingerprint, not when no tracked extreme moved. Mask life totals in the fingerprint and keep "no new low of a losable player's life since that fingerprint" as the progress test. Keep the absolute cap for loops that grow without bound. Making the watch persistent (item 1) without this fix extends the false draw to every finite chain the players pass through. Add the 100-counter chain to `fdn.rs` as a finite control beside the Conqueror loop, with the counter count above `REPEATS`.

Remaining for this slice: implement concept-review items 1–3 with Finding 10, then re-record hash v31 evidence as planned.

### Implemented slice · Closing-slice concept alignment (persistent CR 104.4b watch, draw orientation, grantor-cost authority and state-trigger identity)

Built in `86c6f44a8a` on hash contract **v31** from the full-branch concept review at `65cebaf342` (`scratch/concept-review.html`), which found the watch reset whenever `tick_priority` returned a space (so CR 104.4b never fired with any unrelated instant or ability available), Python Academy inverting `won` on draws, and grantor costs checked only for `exile_grantor`. What exists:

- `MandatoryLoopWatch` lives on `PriorityState.loop_watch: Option<Box<_>>` and survives passes; poison counts as progress only for a seat that `can_lose`. Since `89632b774e` `repeats` takes a sample closure and fingerprints only past `UNWATCHED`, so short chains pay nothing.
- `ActivatedAbilityDefinition::has_grantor_cost` covers both grantor kinds in payment reservation, action focus and `action_description`; `state_trigger_waiting` ignores triggers with `inline_effects`, so a granted trigger never holds a printed state trigger.
- The watch (`priority.rs::MandatoryLoopWatch`) counts resolutions; after `UNWATCHED` (64) it records `GameState::loop_fingerprint` (`state/hash.rs`): the canonical match state minus life totals, event logs, priority, id/timestamp allocation and waiting-trigger enqueue ids, with turn-history counts saturated at 16. A loop is a repeated fingerprint with no new life low or poison high for a seat that can lose; `CAP` (10,000) still ends unbounded growth. A probe of the Herald loop showed only waiting-trigger ids differing between cycles once those fields were masked.
- Cleared by: any non-pass priority action (`execute_action`), mana abilities (`on_non_pass_action`), `reset`, `on_step_start`, publishing a non-priority space with more than one action (a real choice), and a resolution leaving the stack, pending triggers and placement empty.
- Python: `viewer_view(obs, viewer, *, winner)` and `normalize_observation_to_hero(observation, human_seat, *, winner)` (and `person_board`) take the winning seat; the unused `hero_view` is removed.
- Tap-grantor text: "pay {1}, {T}, and tap Fishing Pole (permanent N)"; exile-grantor text is unchanged.

Tests: `fdn.rs` `bloodthirsty_conqueror_loop_draws_while_players_pass_with_choices`, `bloodthirsty_conqueror_loop_ends_when_a_player_kills_the_drain`, `a_finite_chain_over_counters_is_no_mandatory_loop` (Finding 10, 100 counters), `two_fishing_poles_on_one_creature_offer_activations_naming_each_pole`; `m21.rs` `mazemind_tome_state_trigger_is_not_held_by_another_trigger_of_the_tome`; `priority.rs` unit tests; `tests/academy/test_play_modes.py` viewer and trace draw cases. The grantor tap reservation has no catalog witness (no grantor taps for mana or waterbends), so it is covered only by reading.

Checks: debug workspace under nextest (`--profile ci`, 1655 tests) failed only on the v31 pins (14 `frozen_*` semantic-kernel cases, the Clone witness) and a unit-test expectation, all since fixed. Semantic-kernel w17-v6 re-recorded in `3049b48fdb`: 14 cases, 3049 commands, every non-digest field unchanged. Opening authored-match authority (193 decisions, digests only) and live/headless parity (`RUL5_PARITY_OK`, no divergence) plus the Academy demo/choice-navigation fixtures refreshed in `3e7ca196fd`. With the rebuilt cp313 extension: authority/parity tests and the focused Python selection (text observation, effect-execution input, v10, named layout, pinned pilot, Academy boundary, combat preview, Blast, learn setup, decision surface) pass except four Academy tests listed in `scratch/questions.md` that do not touch these paths.

The lazy-sampling refactor (`89632b774e`) and the main merge `7500c5d58f` (MLX learner only, no engine paths) have not been rerun in debug.

**Pre-review alignment fix (concept review at `31c81eda80`, `scratch/concept-review.html`).** Reviewing the whole built branch at hash v31 confirmed that event history, contextual queries, LKI, CDAs, mana-ability split, untap/state triggers, and cross-consumer draw orientation align end-to-end, but exposed four seams in the new CR 104.4b loop-watch contract plus one residual tap-grantor reservation mismatch. Fix all five before the closing-slice behavior review (no hash-version bump; hash v31 and its refreshed receipts stay intact):

- **Validated loop-watch reset (`flow/action.rs::execute_action`).** Clear `self.state.priority_mut().loop_watch = None` only when a non-pass priority action (`PlayLand`, `CastSpell`, `ActivateAbility`, `ActivateCardAbility`, `Foretell`) returns `Ok(())`, never before validation. A rejected/stale priority action while `loop_watch.is_some()` must leave `witness(&game)` unchanged.
- **No watch increment on suspended resolution (`flow/tick.rs::tick_priority`).** After `resolve_top_of_stack()`, if `self.state.suspended_decision.is_some()`, `continue` before calling `watch_for_mandatory_loop()` so suspending spells (which remain on the stack under CR 608.2m) do not allocate `Box<MandatoryLoopWatch>` or count an unfinished resolution.
- **Fingerprint/watch contract parity (`state/hash.rs::loop_fingerprint`).**
  - Mask `players[*].counters["poison"]` (removing `"counters"` if empty, matching `Counters::is_empty`), so `MandatoryLoopWatch`'s `poison_high` / `!can_lose` tracking actually fires on poison loops against *Herald of Eternal Dawn* instead of falling through to `CAP = 10_000`. Leave any non-poison player counters in the fingerprint.
  - Strip `"id"` inside `stack_objects[*]` entries (just like `pending_triggers` and `trigger_placement`), so loops with ≥2 objects on the stack at resolution time repeat their fingerprint instead of burning to `CAP`.
  - Saturate only the count element (`entry[2]`) in `turn.ability_triggers_this_turn` and `turn.ability_resolutions_this_turn`, rather than recursively clamping `ObjectId` and `CardDefId` to 16.
- **Tap-grantor sacrifice reservation (`flow/mana.rs::reserve_announced_payment` and `announced_model`).** Set `sacrifice: true` for `tap_grantor` / `CostComponent::TapGrantor`, matching `owed_reservation`, `payment_reservation`, and `ManaUpfront` (a permanent owed for tapping cannot be sacrificed to pay another cost).
- **Directed checks (and rerun of `89632b774e`).** Add four directed Rust tests: (a) rejected/stale priority action while `loop_watch.is_some()` leaves `witness(&game)` identical; (b) poison loop vs. *Herald of Eternal Dawn* draws shortly after `UNWATCHED` (well below `CAP`); (c) loop with another object remaining on `stack_objects` draws shortly after `UNWATCHED`; (d) suspending spell leaves `priority.loop_watch.is_none()`. Rerun `priority.rs` units, `fdn.rs`/`m21.rs` loop and state-trigger tests, `search_state_contract`, `canonical_replay_tests`, and `semantic_conformance_tests` to confirm zero v31 digest drift.

Next: implement the pre-review alignment fix above, then run the closing-slice behavior review and gate. Engine Budget will see moved digests (untap events in the hashed ledger, hash v31) and the per-resolution watch; adopt the PR's run as the reference if it stays within 10%. Gate also confirms whether the four Academy failures in `scratch/questions.md` fail on main.

Bakersbane Duo’s receipt-backed crossing supplies the integrated Unravel control.

## Remaining shared work

- **Pinned-pilot serialized compatibility (resolved — keep intact):**
  `Effect::MillCards` (`target: Option<TargetSpec>` + `player: Option<PlayerSelector>`, used in 9 TLA cards) and `StaticAbility::CharacteristicPower` beside `CharacteristicPt` (used in 3 TLA cards) already share unified runtime readers (`mill_objects`, `pt_cda`, `object_pt`, `Operation::Cda.sets_toughness`). Collapsing either serialized shape would alter those 12 existing TLA cards' learning-token streams and break `tests/model/test_pinned_pilot.py` (`token_renames.rs` only maps 1-to-1 renamed tokens, not structural arity changes). Keep both legacy serialized shapes intact alongside legacy `PutsCounters`, `DealsCombatDamage`, `TriggeringHadCounters`, and `NoCounters`.
- Replace `TargetSpec::GraveyardCardYouOwnWithinSourcePower` when the shared
  target clause can express its complete bound, including announcement,
  retarget and resolution reads. The special case still sits in
  `flow/resolution.rs::target_meets_requirement`, and the stack projection
  recognizes it by variant. Source power and power tapped to pay share
  `permanent_power_or_lki`; `EffectValue::PowerTappedToPay` is now the
  clamped sum of power over `ObjectSelector::Paid(How::Tapped)` and can
  become a property query once the clamp has a home. Triggering power and target toughness use the
  battlefield slot and its departure record. Preserve the Combustion Man,
  Panda and Sandbender controls and existing learning-token meanings.
- Keep occurrence and amount queries with their event, turn, game or
  instruction owners. Extra combats keep turn history and extra turns reset
  both seats. Payment queries read committed receipts. Floated mana,
  ability costs, rolled-back spends and uncast copies contribute zero to
  Expend. Preserve native chooser, complete targets, manual spend
  alternatives and restricted-first auto payment. History remains
  independent of presentation and optional decision memory.
- Keep public object joins and pending history consistent through native
  projections, Python, v11 and text as new queries land. Save prefixes and
  frozen evidence, and keep hidden identities out of both projections.
  The semantic-kernel, authored-match and live/headless parity receipts are
  refreshed at hash v31 (`3049b48fdb`, `3e7ca196fd`). The closing-slice behavior review, integrated replay and privacy
  checks, and Engine Budget remain open.

## Directed interactions for integrated validation

These are expected outcomes for constructed games. Record the actual Commands, offered alternatives, fixture paths and results alongside them during implementation.

| Interaction | Observable check |
|---|---|
| Fumigate with two destructible creatures and indestructible Predator Ooze | Two creatures destroyed, two life gained in one event. With only the Ooze, gain zero. Separately use Exsanguinate for X=3 against an opponent under Teferi's Protection. Life lost and gained are both zero, versus three of each in the unprotected control. |
| Cat Collector's Food, Fiendish Panda and BLB Lunar Convocation | A Food activation gains three, grows Panda once and creates one Cat on the first gain of the controller's turn. A later gain cannot create another Cat, including when the Collector entered after the first gain. Gain and lose equal amounts in one turn and preserve both facts for Convocation. Check each player's next turn. [FDN release notes](https://magic.wizards.com/en/news/feature/foundations-release-notes) |
| Heartfire Immolator and Fiendish Panda | Sacrifice a prowess-boosted Immolator and use its last power for damage after reentry. Grow Panda before it dies. Offer every other eligible non-Bear creature card up to that last power, exclude Bears and the card just above the bound, and reject a target that left and returned after announcement. |
| Venom Connoisseur, Predator Ooze and Steel Hellkite | Counter the first Connoisseur trigger and resolve two others. The first resolution grants deathtouch to that Connoisseur and the second grants it to the team. Blink it and check the first resolution of the new incarnation. For Ooze compare a damaged creature dying later that turn with its blinked successor dying. Hellkite's X ability filters nonlands by mana value and by controller dealt combat damage by that exact Hellkite this turn. A second copy and noncombat damage provide negative controls. |
| Lunar Insight, Felling Blow and Drake Hatcher | Vary duplicate and distinct nonland mana values while holding permanent count fixed. Felling Blow reads power after adding its counter. Hatcher uses actual combat damage to a player for incubation counters. Exercise its three-counter removal and resulting Drake token. |
| High-Society Hunter and Cackling Prowler | Hunter can sacrifice another creature or decline. It draws for each other nontoken creature dying on either side, including simultaneous deaths with Hunter. A sacrificed creature token can turn on Prowler at its controller's end step without a Hunter death draw. |
| BLB Bakersbane Duo and EOE Unravel | Start at three mana spent on spells. A discounted spell crossing four triggers the Duo once even if Unravel counters it. Compare full price, a mana ability activation, a countered cast, and a Duo arriving after four. Draw with Unravel when the target cost less mana than its stack mana value. [BLB rulings](https://magic.wizards.com/en/news/feature/bloomburrow-release-notes), [EOE rulings](https://magic.wizards.com/en/news/feature/edge-of-eternities-release-notes) |
| EOE Plasma Bolt (void) and TMT Michelangelo, Game Master (disappear) | Your sacrificed Food qualifies for both. Your departing land qualifies only for disappear. An opponent's departing nonland qualifies only for void. A spell cast for its warp cost qualifies for void with no departure. Test the cast-provenance fact at the native commit boundary and record whether the actual warp cast is reachable at the integration revision. Preserve that end-to-end obligation for the casting-permission integration. |

**Remaining integrated checks.** Use the verified Blast fixture above as the replay and hidden-hand control. Check viewer-safe recorded replay frames as well as live recovery. For public learnability, construct equal visible boards with different earlier gains, damage relations or mana spends and check that the legal consequences and their native facts, encoded values and text differ where relevant.

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
