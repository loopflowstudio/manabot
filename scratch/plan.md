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

The latest merge `074bc49a64` adds main’s shared Torch/MLX host admission
and native extent measurement from `b9b09bc6d9`. The evidence above predates
that merge. Check the history rows and saved v9/v11 prefixes on the merged
preparation path with the next extension rebuild. Recorded Watch playback
of partial declaration remains open. The heuristic’s block map still
populates empty entries from `CombatAttackersDeclared`, so an unblocked
creature already attacking before declaration is absent from that map.
Exercise this case with a live attacker projection if the cohort makes it
reachable; keep declaration events scoped to the declared attackers.

## Remaining mechanic batches

Jack Heart, comment a15ea74c-454c-4a55-9b26-fc62e717cac6 (and local sync steer d8007389-a447-4731-a7fc-2ed791fd4e56):

> Stop doing one card per coverage-working loop. In realign, group the remaining in-scope ledger rows by shared mechanics, enumerate the cards in each batch, and select a whole batch for implement. Implement every feasible row in that batch with its directed checks before handing off to compress and review. Review the batch together. Record completed rows and concrete blockers. If one row is blocked, continue with independent rows. Apply this to the current run and subsequent passes.

The FDN ledger has 25 of 81 primary rows with definitions and native
fixture evidence. The 56 pending primary rows each appear once below.
Fiendish Panda and Heartfire Immolator satisfy the additional exact-source
witnesses. Hunter, Hatcher, Prowler and Blast are covered cost-primary
witnesses. Daggers identify pending cost candidates whose full ledger
obligation decides delivery. Keep the established cross-set and integration
boundaries. Later batch order is provisional.

### Next slice · Power at attack, combat and resolution

Land four shared rules/layer/projection unifications first, then author and verify all seven primary rows and one cost candidate in the same batch. Use the existing native `StaticAbility::Flashback` path for Bulk Up and `Effect::TargetCreaturesDealPowerDamageToLastTarget` for Bite Down. If a specific printed obligation is blocked, record the missing dependency and complete the independent rows.

#### Shared unifications (before card authoring)

1. **Recursive trigger matcher (`flow/triggers.rs`).** Unify `collect_occurrences` so it recursively peels all `TriggerCondition` wrappers (`WasCast`, `FromZone`, `WhileSourceAttacking`, `ActiveIf`, `OnceEachTurn`) before expanding leaf events (`Attacks { per: Per::Object }`, `DealsCombatDamage { per: Per::Object }`, or single-event `trigger_condition_fires`). Thread `once_each_turn: bool` through the recursion, evaluate `intervening_if` per matched member (passing the member's `Target::Permanent` when present rather than pre-evaluating once with `triggering_permanent(event)`), enforce `OnceEachTurn` for both printed and granted triggers, and cap `OnceEachTurn` emissions at 1 even with a trigger doubler (CR 101.2 vs. CR 603.2d). Add directed unit/rule tests for wrapped multi-attacker `ActiveIf` + doubler and `OnceEachTurn` + doubler.
2. **Composable object-property & aggregate queries (`ability.rs`, `resolution.rs`).** Add one composable `EffectValue::Property { of: Box<ObjectSelector>, property: ObjectProperty, aggregate: Aggregate }` with `ObjectProperty::{Power, Toughness, ManaValue}` and `Aggregate::{Sum, Max}`. For single-object selectors (`This`, `Target(_)`, `Triggering`), read live characteristics or CR 608.2h departed LKI (using signed values for `Sum`, so negative-power doubling obeys CR 701.10b and post-destruction mana value works for later Feed the Swarm); for `Each`, aggregate over `selected_permanents`. Keep legacy `EffectValue` variants (`SourcePower`, `TriggeringPower`, `TargetToughness`, `GreatestPowerYouControl`, etc.) intact so pinned TLA pilot tokens stay byte-identical. Teach `EffectValue::reads_source_power`, `reads_source`, and `is_resolution_only` to handle `Property` (recognizing `Property { of: This, property: Power, .. }` alongside `SourcePower`), and pass `self.source_permanent_id(frame)` from `Game::select` into `permanent_matches_with_source` so `CardPredicate::namesake` works in `ObjectSelector::Each`.
3. **Layer-safe `Condition::Compare` (`flow/layers.rs`, `flow/statics.rs`).** Handle `Condition::Compare { left, op, right }` in `layers.rs::ReadSet::condition` (via `ReadSet::value`, recording `POWER`, `CARD_TYPES`, `SUBTYPES`, `tapped`, and `mana` dependencies) and in `layers.rs::Context::condition` (evaluating board-reading `EffectValue`s against `self.objects` without re-entering `layer_board()`, while delegating state-only values safely). Guard `statics.rs::layer_state_condition` against board-reading `Compare`. Add a directed `layers.rs` / `statics.rs` test covering a static ability gated by `Condition::Compare` over creature power and tap state.
4. **Stack & Academy projection alignment (`agent/observation.rs`, `academy/ui/src/lib/types.ts`).** Include `intervening_if.as_ref().is_some_and(Condition::reads_source_power)` in `observation.rs::stack_object_data` so pending triggers whose intervening-if reads source power also expose `stack::SOURCE_POWER`. Sync `academy/ui/src/lib/types.ts` with `experience.rs` (`PlayerTurnHistory` fields `life_gained`, `creatures_died`, `permanents_left`, `nonland_permanents_left`, `warped_this_turn`; `PermanentState.attacked_this_turn`; `StackEntryView.mana_spent`).

#### Batch cards & directed checks (8 cards)

- **Courageous Goblin and Ruby, Daring Tracker†** exercise the attack-time power threshold (`ActiveIf` wrapping `Attacks { subject: This, per: Object }`) through the unified recursive `collect_occurrences`. In a paired fork, change power before declaration and after the trigger is on the stack: the former changes whether the trigger fires, while the latter leaves the attack-time qualification intact. For Ruby, branch the native red and green mana choices (`Effect::mana_choice`) and verify the committed output.
- **Nessian Hornbeetle and Surrak, the Hunt Caller** exercise intervening-if conditions at both beginning-of-combat and resolution. Hornbeetle uses `Condition::Compare(ObjectSelector::each_other_you_control(min_power: 4).count(), Ge, 1)` so its own power cannot satisfy its condition. Surrak uses `Condition::Compare(EffectValue::Property { of: Each(YouControl, creature), property: Power, aggregate: Sum }, Ge, 8)` (verifying signed total power including a negative-power creature) and retains the complete legal haste target offer. Check each threshold on both sides of a buff or departure.
- **Ashroot Animist and Halana and Alena, Partners** exercise source-power amounts (`SourcePower` / `Property(This, Power, Sum)`). From the same targeted trigger, change live power or remove and return the source, then compare the resulting pump or counters and remaining keywords. Check the public pending value in both native views, v11 `stack::SOURCE_POWER`, LLM text and Academy, and reject a Command using a stale target identity. Keep one `reads_source_power` reader across Immolator, Alesha and these effects.
- **Bulk Up and Bite Down** exercise target power at resolution. Bulk Up uses `EffectValue::Property { of: Target(Creature), property: Power, aggregate: Sum }` for `+power/+0` until end of turn and `StaticAbility::Flashback`: verify changing power between announcement and resolution, doubling negative power (CR 701.10b: `-2` becomes `-4`), and casting via flashback from the graveyard (exiling on leave-stack). Bite Down reuses `Effect::TargetCreaturesDealPowerDamageToLastTarget` with `CreatureYouControl` and `PermanentOpponentControls(creature or planeswalker)`: fork when each announced target leaves and returns, verify planeswalker loyalty loss, and exercise the existing Felling Blow control.

Record source and fixture paths in `.lf/wave/coverage/research/standard-2026-11/sets/fdn.jsonl` for the whole batch. Run the focused native card, trigger-wrapper, layer-condition and learning-token checks, then rebuild CPython 3.13 and check the merged Torch/MLX host preparation path (`074bc49a64`), named observation layout, semantic text, Academy `svelte-check`, and bundled v9/v11 pinned pilots.

### Later slices

- **Life events, totals and payment thresholds** — Vanguard Seraph; Bloodthirsty Conqueror; Stromkirk Bloodthief; Hidetsugu's Second Rite; Sanguine Indulgence†; Ayli, Eternal Pilgrim†; Exemplar of Light; Bloodtithe Collector.
- **Draw-step and individual draw listeners** — Scrawling Crawler; Dictate of Kruphix.
- **Creature deaths and turn death history** — Tragic Banshee; Vengeful Bloodwitch; Slumbering Cerberus; Needletooth Pack; Wardens of the Cycle; Midnight Reaper; Garna, Bloodfist of Keld; Kalastria Highborn†; Massacre Wurm.
- **Filtered casts, targets and spell attributes** — Archmage of Runes†; Brineborn Cutthroat; Painful Quandary†; Mindsparker; Gnarlback Rhino; Mold Adder; Ovika, Enigma Goliath†; Ramos, Dragon Engine†; Firespitter Whelp.
- **Stack payment and counter results** — An Offer You Can't Refuse; Unravel (EOE, discounted and full-price targets after an intervening cast, with announced X included in stack mana value).
- **Damage recipients, amounts and results** — Goblin Negotiation; Quilled Greatwurm†; Koma, World-Eater†; Niv-Mizzet, Visionary; Lathril, Blade of the Elves†; Gateway Sneak; Trygon Predator.
- **Count or filter live permanents** — Hare Apparent; Brass's Bounty; Krenko, Mob Boss; Dwynen's Elite; Elvish Archdruid†; Inspiring Call; Ingenious Leonin; Leonin Vanguard; Volley Veteran; Archway Angel; Dragonmaster Outcast; Cryptic Caves†.
- **Graveyard thresholds, counts and characteristic definitions** — Kiora, the Rising Tide; Crypt Feaster; Revenge of the Rats; Consuming Aberration; Enigma Drake; Crusader of Odric.
- **Move results and retained characteristics** — Stroke of Midnight; Scavenging Ooze; Cemetery Recruitment; Dread Summons; Maelstrom Pulse; Feed the Swarm.
- **Cost reductions from the board** — Arcane Epiphany†; Ghalta, Primal Hunger†; Ballyrush Banneret†; Gate Colossus†.
- **Kicked spells and entry provenance** — Sun-Blessed Healer†; Gnarlid Colony†; Grow from the Ashes†; Nullpriest of Oblivion†; Gatekeeper of Malakir†.
- **Resolving payments, sacrifices and choices** — Arbiter of Woe†; Vampire Gourmand†; Perforating Artist†; Frenzied Goblin†; Mentor of the Meek†; Desecration Demon†; Redcap Gutter-Dweller†; Springbloom Druid†.
- **Activation costs, thresholds, mana and tap events** — Zul Ashur, Lich Lord†; Fishing Pole†; Ravenous Amulet†; Wishclaw Talisman†; Immersturm Predator†; Savage Ventmaw†; Mazemind Tome; Three Tree Mascot†.
- **Turn departures and alternative casts** — Plasma Bolt (EOE) and Michelangelo, Game Master (TMT). The interaction table below distinguishes each predicate and retains the warp integration check.

Bakersbane Duo’s receipt-backed crossing supplies the integrated Unravel control.

## Remaining shared work

- Replace `TargetSpec::GraveyardCardYouOwnWithinSourcePower` when the shared
  target clause can express its complete bound, including announcement,
  retarget and resolution reads. The special case still sits in
  `flow/resolution.rs::target_meets_requirement`, and the stack projection
  recognizes it by variant. Source power and power tapped to pay share
  `permanent_power_or_lki`. Triggering power and target toughness use the
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
  The 81-row reconciliation, integrated replay and privacy checks, and
  Engine Budget remain open. Six early ledger rows record native passes
  without fixture pointers: Cat Collector, Venom Connoisseur, Predator
  Ooze, Steel Hellkite, Fumigate and Exsanguinate. Their fixtures are
  recorded above. Add links during ledger reconciliation. Once the cohort
  stabilizes, refresh the current semantic-kernel, authored-match and
  Academy parity receipts in `conformance/README.md`.

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
