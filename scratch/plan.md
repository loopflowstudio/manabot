# MTG-422 · Game procedure and terminal outcomes

Start at native reset. Either player can mulligan, choose the order of cards going to the bottom, and take opening-hand actions before the first turn. Time Stop and Sundial of the Infinite use the same route into cleanup. Poison, life, failed draws and explicit card instructions produce one authoritative result for play, search and replay.

The implemented cohort covers six Foundations primary rows and the cross-set witnesses below. Planeswalkers, tournament sideboarding, general event history, casting profiles and copiable forms have separate owners.

## Next slice

At `5390217b`, simulation began reading the committed native result before classifying a step-limit timeout. At `a2ac104a`, each worker returns its GameRecord directly to the summary and CLI. Five focused checks passed. In `tests/env/test_game_snapshot.py`, seed-23 Pact Command replay and indexed play both reported seat 1 winning at positive life. Four one-step simulations produced one win per seat, one simultaneous draw and one Herald-protected timeout. Each seat won 25% of all games.

Run the paired Engine Budget. Build the candidate and reference on the dedicated runner, run the sample and compare commands below, and reconcile the changed ordinary-game decision digests against real opening choices before judging the 10% limit. The local candidate measurements justify carrying the current implementation into review. A relative performance result is still missing.

Gate also owns the complete suite, schema and package checks, representative Academy play, model compatibility and restored mid-game self-play. The sync at `a3f33b18` brought in MLX tracked self-play training and resume. Include the affected snapshot consumer in that check.

### What changed in the last slice

Decision-memory verification assumed reset offered priority before first main. Reset now offers opening choices. At `1ddc2105`, decision-memory and text-observation fixtures share `reset_to_first_main` in `manabot/verify/text_observation.py`. Both keep the dealt hand, decline optional entries and pass priority through the live action space. Exercise both callers at gate.

The matched Herald boards below narrowed the optimization question. At size eight the pending loss added one allocation and 22 bytes across the damage pass. Both boards built layers nine times. Those counts support keeping the existing combat storage and loss check. They do not identify a redundant layer invalidation. Revisit that path only if the paired budget shows a regression there.

## Cards and behavior witnesses

Each row is an interaction for integrated validation. The expected results come from the set reviews, printed rules and release notes. The native definitions and directed set fixtures cover the cohort below.

| Foundations card (set module) | Constructed interaction and observable result |
|---|---|
| Leyline Axe (fdn) | Deal multiple copies, mulligan one hand away, bottom an Axe on a redraw, and give both players an opening action. Each can place any subset of retained eligible objects in any order, or finish. Each Axe enters unattached. Equip for three onto Fynn and verify the +1/+1, double strike and trample grants move with the attachment. |
| Fynn, the Fangbearer (khm) | Connect with Fynn and another deathtouch creature. Each qualifying creature hit on a player creates its own two-poison trigger. Double strike can trigger in both damage steps. Prevented damage, damage only to a blocker and noncombat damage yield none. Removing Fynn after triggering preserves the trigger. Test eight to ten poison with and without Herald. |
| Herald of Eternal Dawn (fdn) | Flash Herald in response to Pact's loss trigger or Maze's End. Test both prohibitions, loss of the static ability and change of controller. Removing the last Herald at ten poison or nonpositive life causes loss on the next applicable check. A previously prevented Pact loss or failed draw consumed at an earlier SBA check does not recur. |
| Demonic Pact (ori) | At upkeep offer each legal unused mode, including loss while protected. Cover four damage to any target and four life gained, targeted opponent discarding two, drawing two, and losing immediately on resolution. Mode use survives countering the trigger and changing control, resets on blink, and is separate across copies. With all four used, no fifth choice. |
| Maze's End (new dgm module) | Enters tapped and can tap for colorless. Activate its other ability for three, tap and return the source to its owner's hand before responses. Search privately for a Gate, allow failure to find, enter and shuffle. Check the effective names of controlled Gates on resolution even after finding nothing. Nine names plus a duplicate fails, a tenth distinct name wins unless prohibited. Use the existing rtr/gtc Guildgates. |
| Time Stop (chk) | Resolve it during the opponent's combat with a spell, a spell copy and abilities below it. Exile Time Stop itself and the other stack objects, remove combatants and verify no lower effect resolves. A delayed trigger for the next end step waits for the next end step that actually begins. Compare the same termination through Sundial's own-turn activation. |

The cross-set selection takes both non-planeswalker EOE player-procedure rows and both MSH-09 rows as additional consumers. Secret of Bloodbending already has a native definition and both-seat tests. In MSH the control card is **Construct a Cosmic Cube**. The card named *Cosmic Cube* has a separate attack/casting ability in MSH-05. [EOE ledger](../.lf/wave/coverage/research/standard-2026-11/sets/eoe.jsonl) · [MSH ledger](../.lf/wave/coverage/research/standard-2026-11/sets/msh.jsonl) · [TLA interactions](../.lf/wave/coverage/research/standard-2026-11/sets/tla.md)

| Cross-set card | Interaction for integrated validation |
|---|---|
| Virulent Silencer (eoe) | A nontoken artifact creature with deathtouch hitting a player qualifies for both Silencer and Fynn. A deathtouch artifact token qualifies only for Fynn. Damage amount does not multiply either trigger. |
| The Dominion Bracelet (eoe) | Equip for one. At sorcery timing the equipped creature's controller may target an opponent, lock the generic activation cost using current power including the +1/+1, then exile the granting Bracelet as cost. Vary power across the zero-cost boundary. The ability survives the Equipment leaving. With competing mana sources, check legal manual contributions, restricted-first automatic spending and the payment receipt. |
| Construct a Cosmic Cube (msh) | The controller's second draw each turn produces a black 2/1 Villain with menace and a plan counter. Crossing the seventh counter triggers sacrifice. Only a successful sacrifice produces the separately respondable targeted control trigger. Test crossing from six by multiple counters, removing the Plan before sacrifice and countering the reflexive ability. Under next-turn control, fork at a nested payment and spend the controlled player's resources. |
| Quicksilver, Brash Blur (msh) | Both Quicksilver and Axe qualify in the same kept hand. Verify Quicksilver's haste and power-up, both counters, one use on that incarnation, and the entry-turn cost reduction for a normal entry. Pregame entry is before turn one and earns no turn-one discount. |
| Secret of Bloodbending (tla, existing) | Retain paid waterbend for next turn, unpaid for next combat, self-exile, latest applicable control, extra turns, and private foretell inspection. End an actively controlled combat early with Time Stop. Verify the chooser reverts for cleanup under combat-only control and stays delegated through repeated cleanup under turn-long control. |

Rules references for the fixtures are CR 103.5–6, 104, 116, 514, 704, 724 and 725. [Comprehensive Rules, November 6, 2026](https://media.wizards.com/2026/downloads/MagicCompRules%2020261106.txt) · [Foundations release notes](https://magic.wizards.com/en/news/feature/foundations-release-notes) · [Edge of Eternities release notes](https://magic.wizards.com/en/news/feature/edge-of-eternities-release-notes) · [Marvel Super Heroes release notes](https://magic.wizards.com/en/news/feature/marvel-super-heroes-release-notes)

## Player-control cards

The Dominion Bracelet and Construct a Cosmic Cube use the existing next-turn control effect. Their native definitions and seed-161 fixtures landed at `0ea0c204`. At `a7cbb302` the Cube fixture forks one shared position where the sacrifice and countering branches diverge. The full suite and paired Engine Budget belong at gate.

| Seed 161 native fixture | Observed outcome |
|---|---|
| `dominion_bracelet_exiles_the_chosen_grantor_at_the_equipped_price` | Sakashima let two Bracelets equip one Gray Ogre. Both +1/+1 bonuses gave a price of eleven. Raising power to sixteen, fifteen and fourteen gave prices of zero, zero and one. Each observed offer identified its Equipment. Forks spent either ordinary blue or restricted colorless mana and left the other unit floating. Repeating either choice reproduced the receipt and successor. Automatic payment spent the restricted unit. Both branches exiled only the chosen Bracelet. The opponent kept their next turn when the paid ability was countered and delegated it when the ability resolved. |
| `construct_a_cosmic_cube_requires_the_plan_to_be_sacrificed` | Doubling Season turned the second draw trigger into two 2/1 Villains with menace and two plan counters. Both viewers saw eight counters. Removing the Plan before sacrifice or countering the separately targeted trigger preserved the opponent’s turn. After successful resolution seat zero chose which of seat one’s Mountains paid for its Gray Ogre. The bound Command reproduced the same successor, spent seat one’s mana and left seat zero’s pool and the fork root intact. |
| `player_control_card_costs_and_trigger_threshold_reach_learning_tokens`, `every_trigger_condition_and_ability_reaches_the_learning_tokens`, `each_counter_kind_reaches_a_fact_row` | The catalog exposes the grantor exile, power reduction, counter crossing and reflexive trigger. The plan counter joins its permanent through the existing counter fact rows. |

### Two-Bracelet offers through `f09d8335`

Seven focused native cases passed at the earlier review. Seed 161 exposed a gap in those checks: the offer focused the Equipment permanent but described its underlying card as the exile cost. At `9de9c111` the native description names Gray Ogre as the ability source and the visible Bracelet permanent as the cost. Semantic text uses the same two focus slots and the viewer's battlefield aliases. Academy carries the native label.

The seed-161 fixture checked both tensor joins and compared each bound activation with indexed play through exile, payment receipt and remaining mana. On the rebuilt CPython 3.13 extension, the seed-23 semantic and Academy Commands matched indexed play at targeting and after payment for either Bracelet. The fixture in `tests/academy/test_bracelet_offers.py` now selects the indexed action independently by the Ogre and Bracelet permanent IDs. This catches a shared offer-binding error even if both Commands agree.
Behavior review at `dcdb8301` closes the two-Bracelet defect. Thirteen constructed native debug cases passed across Bracelet, Cube and Bloodbending, including control tokens, private visibility and delegated-payment replay. Both seed-23 consumer cases passed, as did Academy’s original Command import and stale-revision rejection case. The working phase is satisfied. The full suite and paired 10% Engine Budget remain at gate.

## Cross-consumer acceptance

Native decisions own the rules subject and publish the chooser once. Preserve private bottom positions, sideboard privacy, forced-choice skipping, complete native and Academy offers, replay and fork isolation, stale and wrong-actor rejection, and saved model column prefixes. At gate compare both viewers, reconnect at the pending choice, and check native, Python, tensor and text projections. Keep frozen evidence at its original revision.

### Turn ending at `f2caa4fde913dfb70ccedea1d16d9ea2a1ee8b22`

A pending state-based choice already retains a native frame in the cloned and hashed game state. `FrameCompletion::EndTurn` supplies its return point and its viewer-facing description.

| Debug witness | Observed result |
|---|---|
| `chk::time_stop_exiles_the_stack_during_opposing_combat`, seed 161 | The defending seat exiled its Time Stop and the attacking seat's Bolt with Sundial's ability between them. The Ogre stopped attacking. All eight cleanup discards replayed on forks and stale Commands were rejected. The queued extra combat disappeared and the extra turn began. |
| `chk::time_stop_skips_the_end_step_but_preserves_its_delayed_trigger`, seed 161 | Unearthed Cityscape Leveler remained on the battlefield through the skipped end step and went to exile at the opponent's end step. |
| `end_turn_resumes_its_state_based_check_before_cleanup`, seed 725 | With two Azulas, a spell copy, one waiting death trigger and another creature marked for lethal damage, each legend choice replayed from the same root. The copy ceased to exist, the lethally damaged creature died, the this-turn cast grant expired before cleanup priority, and only the new trigger gained three life. |
| `tla::secret_of_bloodbending_control_during_time_stop_and_repeated_cleanup`, seed 161 | Combat control ended before the first discard. Paid turn control lasted through two cleanups, including a second Time Stop in cleanup. Humble Defector drew two cards during cleanup priority and the second discard was mandatory. |
| `cr_514_cleanup::cr_514_3a_cleanup_grants_priority_after_a_legend_choice` | A cleanup whose only work was a legend-rule choice granted priority and stayed in cleanup (CR 514.3a). Before `f2caa4fde` it did not. |
| `m12::sundial_of_the_infinite_ends_the_turn_exiling_the_stack` and `combat_counts_partners_and_committed_deaths_in_a_terminal_combat` | Sundial still exiled the underlying spell, expired the Ogre's buff and remained own-turn only. Interrupted combat still settled capture counts. |

All 20 turn-ending, cleanup and Bloodbending debug tests passed again with nextest at `f2caa4fde`. The Scryfall conformance check passed at `cc3671bd4`. A release CPython 3.13 build replayed a Python text offer to cast Time Stop at seed 23 with six Islands. Both seats observed the spell in exile and seat 1 took the next turn. The focused Python checks for Bloodbending's text/policy/Academy authority and reconnecting at a pending discard passed (three cases).

## Gate measurements

**Performance** keeps the paired same-runner 10% Engine Budget and its semantic digests. Real mulligan decisions change decision sequences even on ordinary decks. Record rule-driven digest changes under the repository baseline procedure and retain the old recording. Measure ordinary play, training stepping and search alongside bounded constructed positions in `managym/examples/bench_states.rs`. Compare size, action count and work at the decision as well as time. Add a named state where existing fixtures miss the shape. [Validation and stress-state harness](../docs/development.md#stress-states)

| Constructed shape | Checkpoint and cost to measure |
|---|---|
| Seven opening cards, mostly identical Axes versus mixed Axes, Quicksilvers and distinct ineligible cards | First and later ordered bottom selections and pregame placement. Count legal object choices, compare observation, clone and search work. |
| Fynn and a Silencer, with increasing numbers of nontoken deathtouch/artifact attackers versus nonmatching tokens | Combat damage and trigger ordering. Count sources examined and matches per damage event. |
| One or several Heralds among many unrelated permanents, including a mixed board of continuous effects | Ordinary SBA pass, blocked loss and removal of the last Herald. Compare layer queries and stabilization work. |
| Maze's End, duplicate Guildgates versus ten different Guildgates, several candidate Gates in library | Activation payment, private search and post-entry distinct-name check. Count candidates and legal choices. |
| A Bracelet, many potential equip targets of different powers and a mix of ordinary/restricted mana | Equip target enumeration and granted activation at low and high power. Compare payment work, receipts and nested choice cost. |
| Time Stop above increasing numbers of spells, copies and abilities during combat, with a cleanup discard and a waiting trigger | Measure the actual transition into cleanup, its first choice, and exact-fork cost. Check surviving stack, event and schedule digests. |

The later gate owns the complete debug suite, schema and package checks, representative Academy interaction, model compatibility and budget evidence. The focused receipts cover turn ending and opening hands before the sync at `d067b283`. That sync brought in saved live positions and native allocator changes. Seed 213 below covers private-bottom restoration. The subsequent sync at `e7a81c2b` brought in self-play from captured mid-game snapshots. Include that consumer in retained snapshot compatibility at gate and use the synced allocator for the paired budget.

## Opening-hand implementation

Ordinary reset now stops at the starting seat’s declaration. The opening workspace owns the stage, kept seats, mulligan counts and bottom cursor. Each bottom selection moves its exact object and records the resulting private library positions. The next round draws only for seats that took another mulligan. Both opening cards use the same printed battlefield-entry ability. Pregame entries carry turn zero. The first ordinary turn starts after the last opening action.

- At seed 103, both seats redrew twice. Each second selection offered six exact objects. Commands for all legal bottom and entry choices replayed on independent forks, stale Commands failed, and the original witness stayed intact. Each chooser saw its bottom positions and the other seat saw none. Both viewers saw the Axe and Quicksilver enter before turn one. Quicksilver’s turn-one activation still cost five.
- At seed 104, the starting seat kept its seven original objects. The other seat redrew down to zero without asking the starting seat again.
- The FDN equip fixture moved both keywords and +1/+1 between two Gray Ogres. The MSH fixture paid four generic mana for a normal entry’s power-up, received both counters and exhausted the activation for that object.

The CPython 3.13 release build passed scalar/vector opening parity at seed 213. Two different private bottom selections produced identical opposing viewer JSON and different owner library facts; an owner-perspective sample preserved the owner's full observation. Academy restored all seven bottom offers and the frame hash on reconnect and accepted the retained Command. The existing Bloodbending text-policy authority tests, saved-column-prefix test, direct Torch attachment projection and entry-turn tensor mutation passed. The native card shells matched their Scryfall fixtures.

### State-based terminal outcomes at `6591d386`

The release build at the earlier sync failed the 4096-byte GameState assertion. Compact opening counters alone were insufficient. OpeningHands, Playing and Finished now share one procedure field, and combat storage is allocated during combat. The release build passed the original ceiling. That revision used match-state hash v18. The instructed-victory revision below uses v19.

- At seed 213, the CPython 3.13 release extension restored snapshots at both the first private bottom selection (seven offers) and the second selection after one card was already placed (six offers). Both seat views and the exact search witness matched. Restoring the saved replay JSON and choosing the original card reproduced the same successor. Choosing another card changed only the entitled viewer’s library knowledge.
- At seed 161, `fynn_poison_and_herald_loss_protection` attacked with Fynn at eight opposing poison. The defender reached ten in both viewer observations. The unprotected defender lost; a defender with Herald survived, consumed a failed draw, then lost to poison on Herald’s removal. `herald_protects_against_a_failed_draw_in_its_last_sba_batch` kept the game live after a lethally damaged Herald protected against a failed draw in its final batch. A subsequent failed draw ended it.
- At seed 704302, a seat at zero life with a failed draw and an opposing seat at zero life drew. Both causes survived on the first seat. Semantic Commands replayed on exact forks. The Env result fixture verified poison rewards of +1 and −1 and draw rewards of zero for both seats; the simultaneous life/empty-draw seat reported `deck_empty`.
- Player counters use the existing sparse counter storage and counter fact rows. Both viewer-relative joins carry the poisoned player’s count. Text and Academy expose the same public count, and the terminal result records the pre-batch conditions after applying protection.

### Instructed outcomes and explicit wins at `77f1c4b3` and `c4c61653`

At seed 161, the Pact upkeep offered all four modes. The loss Command replayed identically on an exact fork. Without a response its controller lost. Flashing Herald in response kept the controller alive through resolution and subsequent removal. Only the three unused modes were offered at the next upkeep. A CPython 3.13 release run at seed 23 restored the pending upkeep from both a live snapshot and serialized replay. The same bound Command reached the same terminal witness on each branch and left the root unchanged.

Maze's End returned to hand before the search. Nine Gate names plus a duplicate failed when the controller declined to find. Finding the tenth name won. With ten names already on the battlefield, declining to find also won, and an opposing Herald prevented the win. The Env fixture checked both seat assignments, the instructed loss reason and winner/loser rewards for each kind of outcome. Existing simultaneous-loss reward and reason projections passed.

At `4edddb75`, seed 161, equipping Leyline Axe to Fynn and attacking unblocked dealt four combat damage and gave the defender four poison counters across both damage steps.

At `4bb40e8d`, seed 23 exposed a consumer discrepancy. Indexed and semantic Pact steps reached the same exact terminal state with reward −1, but the semantic info dictionary was empty. Three mana encoding/filter fixtures also passed at this revision.

### Env parity and card interactions through `5466dbe3`

- At `d3aec3f4`, the seed-23 Pact snapshot fixture stepped indexed action 3 and its bound Command from both live and replay-restored snapshots. All four branches returned reward −1, terminated true, truncated false, winner 1 (Opponent), and p0 loss reason instructed_loss. Their exact successors matched and the root stayed intact. The test also passed on the release extension rebuilt with Silencer.
- At `7d62f9e3`, the seed-161 Silencer fixture granted deathtouch to Silencer or a Ballistic Boulder token. Their two-damage hits put two or one poison triggers on the stack. Exiling Fynn before either resolved preserved its trigger. Six poison became ten and a loss for the nontoken hit, or eight for the token hit. Fully prevented damage, blocker-only damage and a Warstorm Surge hit produced no poison. The native Scryfall conformance test passed.
- Through `b75a5670`, at seed 161 the Pact fixtures resolved all three resource modes, countered the draw trigger before Agent of Treachery stole Pact, and observed only the three unused choices for the new controller. After four resolved modes under Herald the next upkeep had no trigger on the stack. A second Pact or a blinked original offered all four choices. The countering fixture uses a scenario-local activation with the native counter-ability effect.
- At `14d58670`, the opposing seat flashed Herald while Maze’s activation was on the stack and Maze was already in hand. Herald resolved first and prevented victory despite ten distinct Gate names.
- At seed 161, the Maze land fixture played the land, observed tapped entry, advanced to the controller's next upkeep, and confirmed the mana choice added exactly one colorless. Both Maze fixtures passed. The seed-23 Pact outcome parity fixture also passed against the installed release extension.

### Native admission and release checks

At `5d6e00bb` the native probe admitted Silencer alone among the three cards. At `0ea0c204` the rebuilt CPython 3.13 extension admitted Bracelet, Cube and Silencer together. The earlier review passed nine focused native cases and the seed-23 Pact parity fixture. Three focused Python cases subsequently passed across the paid and unpaid Bloodbending decision surfaces and Academy reconnect at a pending delegated discard.

### Maze’s End decomposition at `1af15ad4`

Maze now places the ten-name Gate condition after the entire search. Fabled Passage checks the ordinary land count only after a successful find. The condition reads effective names in both evaluation contexts. Ordinary counts keep their previous catalog token representation. The retired Maze-specific instruction has no successor token because its meaning is now distributed across the condition and the win instruction.

At seed 161, nine names plus a duplicate failed on decline, finding the tenth won, declining with ten won, and flashing Herald prevented that victory. Passage fetched a Forest tapped with two other Forests and untapped with three. The native description and catalog tests verified the threshold wording and distinct-name field. Both seat assignments received the expected rewards for an instructed win or loss. The suspended Clone replacement fork passed with its observed v19 state hash.

On the rebuilt CPython 3.13 extension, the seed-23 Pact Command matched indexed play across live and replay-restored snapshots. Both checkpoint formats passed catalog-growth binding, and the saved-column-prefix case passed. The native rename table exposed the retired Maze variant.

### Post-compression review

Maze’s End and Fabled Passage share the search and battlefield-count logic. Maze checks ten distinct effective Gate names after either search branch. Passage checks four lands only after a successful find, when it has a fetched object to untap. Review found no further simplification here.

The seed-213 scalar/vector opening-label check, ordinary cleanup legend choice and Sakashima/two-Bracelet exemption passed. Seed 725 initially failed before either fork because it built the stack during opening hands. At `72d87467`, the fixture in `managym/src/flow/resolution_tests.rs` took both native keep choices and asserted Playing before constructing the position. The directed test passed with “turn is ending” visible at the legend choice. Either choice resumed cleanup and resolved the surviving trigger for three life while the original state hash stayed unchanged.

The decomposition criterion is met. The only remaining mention of the retired Maze instruction in engine source is its catalog token retirement entry. Maze and Passage each express their timing in the card definition through the shared search and count operations. The focused seed-725 native test passed again on review. Gate retains the affected suites and performance measurement.

### Candidate measurement at 7e5f71bd

Local protocol set before sampling. Use this Linux x86_64 runner (CPU model unreported), Cargo 1.99.1, locked release build with benchmark counters. Run the curated 512-game headless reset/step workload once, and the ordinary seeded Search-64 harness once with a 240-second wall limit, retaining complete seed rows. For constructed states use seed 7, sizes 4 and 8, duplicate Gray Ogres and one Herald per seat among cycling static abilities. Measure clone, observation and the damage pass with the harness's three timing samples and a separate single allocation sample. No cross-machine timing comparison.

Follow-up attribution, chosen after seeing the initial allocation counts. Run the same three layers and sizes on herald-healthy and herald-statics together after ordinary search finishes. These roots differ only in seat zero's life (20 or 0). Compare damage allocations and work, since timing alone cannot attribute Herald's protection query.

The 512 ordinary games all finished, with 118,255 decisions in 3.584 s (32,992 steps/s), digest 17942664948341206392. Layer-board evaluation took 0.572 s over 208,412 builds. Search-64 finished 40 decisions on each seed without a playout cap hit.

| Search seed | Total s | Worst reply s | Simulations | Decision digest |
|---|---:|---:|---:|---:|
| 1 | 29.277 | 2.359 | 7,872 | 9802435900664878732 |
| 2 | 34.207 | 1.827 | 9,152 | 16681168173759266580 |
| 3 | 28.400 | 1.855 | 8,128 | 10971803445708699009 |

At the blocked damage checkpoint, size counts the variable creatures per seat. The two Herald rows use the matched follow-up timings.

| Board | Median µs, 4 → 8 | Allocations, 4 → 8 | Allocated bytes, 4 → 8 | Layer-board builds, 4 → 8 |
|---|---:|---:|---:|---:|
| Duplicate Ogres | 20.31 → 33.63 | 174 → 274 | 44,918 → 85,982 | 4 → 8 |
| Statics with Herald, starting life | 41.17 → 130.37 | 763 → 3,307 | 109,112 → 476,033 | 4 → 9 |
| Same board, seat zero at zero life | 39.09 → 122.37 | 863 → 3,308 | 120,623 → 476,055 | 5 → 9 |

At size eight on the protected board, observation used 120 allocations / 38,070 bytes, and exact fork used 26 / 15,484. In the first run their medians were 10.91 and 2.60 µs. Timing and allocation modes produced the same surface and successor digest in each experiment: c9651da83aabe75a912d82063882d4d18eedaf30a166d9e7059eeeefe560df70 for Ogres plus protected Herald, fe95622bc016a436347f87cfc401223770fb33151bfa041c8864e6df8749502c for the matched Herald boards. The release example test passed for every named recipe.

**Implementation decision.** Keep the procedure and combat representation. The expensive measured transition is crowded combat with layered characteristics. The matched size-eight boards differ by one allocation and 22 bytes. A loss-protection index would add invalidation work without addressing the observed growth. At size four the pending loss did add a board evaluation, so that inference is specific to the larger cell. No runtime deletion is justified in this phase.

A follow-up can instrument which writes invalidate the nine board evaluations inside the size-eight damage pass. If successive evaluations read identical layer inputs, remove the redundant invalidation at its writer. Any such change needs the debug layer oracle, damage/trigger order, simultaneous SBA and Herald-removal fixtures, exact fork and both constructed digests. Keep the complete legal actions, native skipping and observation capacity checks. The counts alone cannot establish which writes are redundant.

The ordinary decision digests differ from the recorded pre-mulligan reference. Gate must review those changes and run the uninstrumented paired budget on its dedicated runner. With candidate and reference examples built as in engine-budget.yml, run:

```bash
uv run --no-project --python 3.13 python scripts/engine_budget.py sample --candidate managym/target/release/examples --reference other/managym/target/release/examples
uv run --no-project --python 3.13 python scripts/engine_budget.py compare
```

No baseline update or relative speed claim follows from the local instrumented sample.

### Performance review at d972bed7


The reviewed reference is `932a1bb9`. Its 512-game workload took 123,517 steps with digest 6877553498864102521. The candidate took 118,255 steps and all three search seed digests changed. Opening choices change the paths through those games. The local instrumented release and the hosted reference also have different build and machine profiles. These receipts establish neither a speedup nor a regression. Gate must reconcile the changed paths and run the paired release Engine Budget on its ubuntu-24.04 runner. Reference adoption remains a separate reviewed operation.

The local engine measurement was based on `7e5f71bd`; the matched Herald workload is recorded in `5a43ee03`. The timing and allocation receipts agree on each constructed cohort digest. Doubling the variable creatures on the healthy Herald board multiplied damage-pass allocations by 4.33 and median time by 3.17. At size eight, adding a pending loss left board builds at nine and added 22 allocated bytes. Crowded damage merits investigation if it constrains the paired workload. Retain the current representation pending evidence identifying a redundant rebuild. Debug checks in `flow/statics.rs` compare the cached board with a fresh evaluation, and `flow/layers.rs` checks dependency pruning against pairwise evaluation. The release samples do not exercise those assertions. Gate retains the debug suite alongside the directed Herald-removal, simultaneous-loss and exact-fork evidence.

### Performance review after simulation repair at `13fead79`

Advance to gate. Four focused Pact and simulation cases passed on the CPython 3.13 extension. The seed-23 indexed action and replayed Command returned the same positive-life winner and exact successor. With a one-step cap, both winning seats, a simultaneous draw and a still-live protected game reached the summary as four distinct results. Each seat won one of four games. The CLI reads these counts from the same summary.

The repair changed the Python consumer. The native benchmark source and workloads are unchanged since the preceding review, so the constructed-state findings still apply. Crowded combat remains the measured opportunity. The ordinary candidate and historical reference follow different decision paths, leaving relative performance unresolved. Gate owns digest reconciliation, the paired 10% Engine Budget and debug invariant checks. Reference adoption requires its separate review.
