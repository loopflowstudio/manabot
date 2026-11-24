MTG-418 had a Task directive and shared set design but no scratch plan at the checkout base. The working design lives in scratch/plan.md. Bakersbane Duo, Lunar Convocation, Plasma Bolt, Michelangelo, Game Master and Unravel are the selected concrete cross-set witnesses for the five requested query families. The primary FDN worklist remains the ledger selection.

Fumigate's isolated destruction fixture uses the existing indestructible Marit Lage token while Predator Ooze's damage-history work remains pending. The integrated Predator Ooze interaction stays in the plan.

Compression keeps the v18 bindings intact. Each caller supplies its binding when constructing a batch, including an explicit None for independent followups.

For concept review, pair Cat Collector with Venom Connoisseur in the next slice. Two gain listeners alone cannot expose a resolution count inherited by a returning permanent.

Resolution keys include the captured printed face: ability index zero can belong to different printed abilities on a permanent whose face changes. The live projection shows the current face and each stack entry retains its own.

Checkpoint assembly uses the saved fact embedding width at construction. Fresh encoders use the native fact vocabulary.

For the exact-source pair, start with Panda and restore its pending trigger before testing source and target reentry separately. Immolator alone cannot exercise target enumeration.

The checkout's library-order pin addition failed to compile with an unbound pre-move object and exceeded the GameState size guard. The prerequisite repair uses the departing object and boxes the two existing pin sets together.

For Lunar Convocation, place the Food gain before the enchantment enters. This distinguishes controller history from events witnessed by the permanent in the same two-branch fixture.
For Predator Ooze, use membership per exact source and recipient. Repeated hits and source reentry distinguish this from a per-hit count or a card identity.

For the policy projection, use one v11 relation per live damage pair so both endpoints join the padded permanent table. Steel Hellkite is the next planned source-history witness.
Damage history uses a sorted vector for the typically small set of distinct hits in one turn. Insertion shifts later pairs, and lookup starts at the source’s first pair.
For Hellkite, use a pending activation after source departure to test whether eligible seats remain visible without a live source row.

Hellkite shares the sorted damage history with Ooze. The two player seats fit in two viewer-relative flags on each v11 permanent or pending activation.

Compression retains the sorted pair representation and reads the combat-player range directly. Permanent and stack rows share the same viewer-relative seat conversion.

For Lunar Insight the fixture restores a cloned pending game, following the Steel Hellkite card fixture. Felling Blow is the next planned resolution-order witness.

The Felling Blow fixture uses Gray Ogre and Steel Hellkite for the requested 2-power source and 5-toughness victim.

For Drake Hatcher, carry the amount through combat batch matching into its pending trigger. Use a restored paid activation to distinguish spending counters from creating the token.

The pending damage binding exceeded the 4096-byte GameState ceiling with trigger placement stored inline. Boxing the placement round preserves the ceiling. The countered-activation fork uses a constructed instant with the engine’s counter-ability instruction.

For Hunter, use a nontoken noncreature copying a creature as one simultaneous victim. This makes the departure and destination readings distinguishable.

Prowler’s turn-wide fact uses the active player row as its v11 join for either viewer. The turn owns the single boolean.

The Blast privacy fixture uses identical closed lists for both substitutions and ordinary forced-action skipping. Nonacting LLM privacy is checked at capture admission because only the chooser receives a decision prompt.

Bakersbane Duo uses Tolarian Terror discounted by five graveyard spells for the three-to-five crossing. Counterspell supplies the countered branch while Unravel remains its own slice.

The power batch follows attack and raid because its four paired cases exercise the new query readers at declaration, trigger placement and resolution. Bulk Up can try the existing printed flashback path while the other seven rows proceed independently.

For the power batch and later queries, add one composable `EffectValue::Property` (`Power | Toughness | ManaValue`, `Sum | Max`, live-or-LKI for single-object selectors) while preserving legacy TLA variants for pinned pilots. Unify `collect_occurrences` recursively over trigger wrappers with per-member `intervening_if` and a CR 101.2 cap of 1 on `OnceEachTurn` triggers under doublers. Wire `Condition::Compare` into `layers.rs` so static abilities can share the same vocabulary without re-entering `layer_board()`, and reuse `TargetCreaturesDealPowerDamageToLastTarget` for Bite Down.

The power batch uses a constructed five-loyalty planeswalker for Bite Down, so the same pending spell can be replayed after either target returns. Ruby's color branches start at an empty mana pool. Both source-power fixtures use the Ogre's bound address from a real cast offer before combat to check the old target identity after reentry.

For the life/threshold batch and Findings 1–2, unify event-bound numeric trigger payloads into one `triggering_amount: i32` (v11 slot 16, continuation, text, Academy) rather than adding a separate `triggering_life_lost` field; add `TriggerCondition::PutsCountersOn` with `TriggerSubject` while preserving legacy `PutsCounters` for pinned TLA pilots; add `ObjectSelector::Paid(How)` (with LKI stack-receipt projection for departed paid objects) and `StaticAbility::CostsLess { amount, condition }` while preserving legacy TLA variants; store `Player.starting_life: i32` (default 20, omitted from hash when 20) with `EffectValue::LifeAboveStarting` and `ActivatedAbilityDefinition.condition: Option<Condition>` in one `check_activation_terms` authority; route `Modification::Firebending(n)` through `view.triggered_abilities` and delete `granted_firebending_triggers`; and add `GameOutcome::Draw` with zero-choice CR 104.4b mandatory-loop detection in `tick.rs` (finite high-life drains still win).

The life batch renamed the branch-only `triggering_damage` and `EffectValue::TriggeringDamage` to `triggering_amount` / `TriggeringAmount` outright. Pinned pilots never saw either name. Only losses and combat hits bind an amount, so existing gain triggers keep their fact rows.

`Player.starting_life` reaches no tensor column: every setup starts at 20, and only a constructed fixture changes it. Ayli's threshold shows through which activations are offered.

The CR 104.4b draw detector is deferred to its own slice with a loop partner (Vito, Thorn of the Dusk Rose). Bloodthirsty Conqueror's ledger row carries `pending` for it.

For the death batch (concept review at `3da933e53b`): bisect and clear `frozen_sacrifice_162/163` at v26 before bumping the hash; complete `ObjectLki` with layered `name`, `types`, `subtypes`, `colors`, `is_legendary` and combat/tap flags (`was_attacking`, `was_tapped`, `was_in_combat`), share one `matches_permanent_facts` helper across live `matches_view` and departed `lki_matches_predicate` (replacing destination-card/printing fallbacks in `subject_matches_moved_object` and `additional_trigger_times`), and add `Condition::TriggeringMatches(CardPredicate)` (keeping legacy `TriggeringHadCounters` for pinned TLA pilots); project departed triggering-permanent LKI via `effect_execution` (`Projection::lki`) and departed `target_reference` definitions instead of adding bespoke stack columns; bind `TriggeringPlayer` via condition-aware `event_player(leaf, event, hit)` (controller for `Dies`/`LTB`/`ETB`/`Sacrificed`, owner for `PutIntoGraveyard`); and carry `chosen_modes` on `TriggeredAbilityOnStack` (leaving `inline_effects: None` for printed `TriggerModes`) so CR 603.4 `intervening_if`, `ability_index`, `StackObjectData.chosen_mode`, and v11 `stack::MODE` stay unified.

Combat damage to one player from several sources commits one `LifeChanged` per source, so "whenever an opponent loses life" (Bloodthirsty Conqueror) triggers once per attacker, each with its own amount. CR 119.9 makes gains per source, and nothing in the CR or the card rulings says losses work otherwise. The total gained is the same either way. Keep per-source loss unless a ruling says otherwise.

Kalastria Highborn's "this creature or another Vampire you control" is `Dies(you_control(Vampire))`. A Kalastria that has lost the Vampire type would not trigger on its own death; no fixture needs that yet, and a subject naming "this or matching" would cover it.

Massacre Wurm's stolen-creature control uses the opponent's Agent of Treachery, since tests cannot call `change_control`.

For the post-death concept review at `ac8add3da2` (`scratch/concept-review.html`): merge the Draw-listener, Filtered-cast, and Stack/Counter batches into one 12-card slice on hash v28; store `SpellLki` in `GameState::spell_lki` at the single stack-departure site (`zones.rs:658`) and delete `ResolutionTarget.spell` (all live/departed spell queries read `Game::spell_info`, extending `PlayerSelector::TargetController` to `Target::StackSpell` and adding `ObjectProperty::Colors`); unify `asks_triggering_spell` into recursive `ContextRead::TriggeringObject` and project departed `SpellLki` via `effect_execution` / `target_reference`; add `PlayerSelector::EachPlayer` (APNAP) and multi-player `Effect::ForPlayer` via existing `RestoreController`; add `targets: SpellTargetFilter` to `TriggerCondition::Casts` (keeping legacy `SpellTargetsOnlyThis`) and `Condition::Not(Box<Condition>)`; keep `tick.rs`'s Turn-1 draw step unchanged to preserve frozen pilot and replay step schedules; and keep Ovika gated on composite `PaymentCost` while testing departed-X mana value directly.

Draw/cast batch: Firespitter Whelp's overlapping case (a noncreature Dragon spell) has no catalog witness; pushing a subtype onto a hand card does not survive its move to the stack, so the test checks one trigger for a Dragon creature instead. Unravel's spell-copy case and An Offer's copied and stolen spells are untested; `spell_info` treats them like any other spell.

The mana planner treats a counter-removal mana ability as at most one activation per plan (`limited`). A source with ten counters and a five-counter cost that needs two activations in one payment would be under-planned; Ramos is once each turn, so no current card reaches it.

EOE ledger rows had no `implementation` field; Unravel's row now carries the FDN form (`status`, `source`, `fixture`).


For the post-draw/cast concept review at `68c767217e` (`scratch/concept-review.html`): replace `ObjectLki.name: String` with `#[serde(skip_serializing_if = "Option::is_none")] pub renamed: Option<String>` (populated only when `characteristics.name != definition.name`) and share `lki_face_printing` / `lki_mana_printing` across `statics.rs::lki_matches_predicate` and `resolution.rs::departed_property` so `compatibility_names_and_content_allocations_are_metamorphic` passes, departed `Clone`s read their copied MV/supertypes, and trigger snapshots drop a per-object `String` clone; enforce CR 201.2a (`!a.is_empty() && a == b`) in `matches_permanent_facts`; add `ObjectSelector::TargetAndSameName(TargetSpec)` for simultaneous *Maelstrom Pulse* destruction (preserving CR 201.2a on nameless face-down targets); add optional `in_zone: Option<ZoneType>` on `EffectValue::Objects` alongside `Effect::MillCards.player` so *Dread Summons* verifies actual graveyard arrival while keeping CR 701.13c in `frame.this_way`; and use 2-player `TargetSpec::PermanentOpponentControls` for *Trygon Predator* rather than plumbing `triggering_player` into `target_meets_requirement`.


For the post-v29 concept review at `d7ef237481` (`scratch/concept-review.html`): replace the branch-only `TriggerCondition::DealsNoncombatDamage { source: Actor, to: Actor }` (zero pinned-pilot exposure) with `TriggerCondition::DealsDamage { kind: DamageKind, source: TriggerSubject, to: DamageRecipientFilter }` on `EventKind::DamageDealt` (keeping `DealsCombatDamage` for pinned pilots and `Per::Batch`), generalize `event_amount` and `trigger_context` for `DamageDealt`, and narrow *Quilled Greatwurm*'s ledger blocker to its graveyard cast and distributed counter removal; delete `name_triggering_object`'s blanket `.replace("that permanent", ...)` and thread `triggering_noun` into `selector_description` for `ObjectSelector::Triggering`; add `EffectValue::Scaled { factor, value }` so *Archway Angel* emits one CR 119.9 life-gain event; make `Condition::WarpedThisTurn { who: Actor }` symmetric with `Condition::PermanentLeftThisTurn { who, nonland }` (using `Actor::AnyPlayer` for both halves of Void) with `Game::record_warped_cast(player)` as the commit boundary; use `CardPredicate.named` for *Hare Apparent* (CR 201.5c survives source departure/rename; `development-v1` enforces no 4-copy limit); and pull *Ghalta, Primal Hunger*† into the live-board slice alongside *Brass's Bounty* in `rix.rs`.

Damage-trigger and live-board slice: `Game::record_warped_cast` was not added. Nothing writes `warped_this_turn` before the casting-permissions task lands warp, and an uncalled `pub(crate)` writer would be setup nothing uses; Plasma Bolt's test sets the turn-history flag directly. That task should add the writer at its cast commit. Likewise `trigger_context` does not bind a `DamageDealt` source as the triggering object, so `triggering_noun` keeps "permanent" for `DealsDamage`; Quilled Greatwurm is the first card that needs both. The 5/5 Dragon token key is `token.dragon_5_5` (TLA's firebending 4/4 keeps `token.dragon`); its Scryfall id is the first 5/5 flying red Dragon token printing (PR2), and the Rabbit's is tCLB's.


For the post-v29-live-board concept review at `256d8ac310` (`scratch/concept-review.html`): split `ActivatedAbilityDefinition::is_mana_ability()` (CR 605.1a) and `has_dynamic_mana_output()` from finalized `mana_choices()`, switch Layer 6's `lose_nonmana_abilities` (`layers.rs:1405`) to `is_mana_ability()`, materialize dynamic mana heads via `split_first` in `finalize_characteristics`, and unblock *Elvish Archdruid*† immediately in Slice 1 using the existing `current_mana_output` pipeline (`fdn.rs:1926`); complete `DealsDamage` source binding in `flow/triggers.rs::trigger_context`; add `PowerCda::GraveyardMatchingBy` and `StaticAbility::CharacteristicPt` with shared Layer 7a and off-battlefield P/T evaluation; add `Effect::MillCards.player`, `Effect::MillUntil`, `EffectValue::Objects.in_zone`, and `ObjectSelector::TargetAndSameName` for the 12 primary graveyard/CDA/move-result rows (80/81); and in Slice 2 close 81/81 (*Mazemind Tome*) via generalized `check_state_triggers` (`TriggerCondition::CountersAtLeast`) + `ActivatedAbilityDefinition.add_counters` (`EventCause::Cost`), plus `ObjectProperty::Counters(CounterKind)` + departed source LKI projection in `project_stack_object` for *Ravenous Amulet*†, remaining feasible cost-primary rows (*Mentor of the Meek*†, *Immersturm Predator*†, *Fishing Pole*†), CR 104.4b draw, Findings 3–5, and final conformance re-recording.

Mana-split / CDA / mill slice: Scavenging Ooze reuses `Condition::TargetCardExiled(creature)`, which already reads the exiled object the target became (CR 400.7j), rather than a second `Objects { of: Moved, in_zone: Exile }` form of the same question. `in_zone` serves Dread Summons (graveyard) and Cemetery Recruitment (hand). No catalog card replaces a mill's graveyard move with exile, so Dread Summons' exile-replacement control has no constructible witness yet; the first card with "if a card would be put into a graveyard from anywhere, exile it instead" should add it. `MillUntil` emits one `CardRevealed` per revealed card before the batch move; the revealed cards then become public in the graveyard. `selected_players(EachOpponent)` remains two-player. The Clone catalog-order witness moved again (new definitions shift name-ordered ids). The `SETS` list in `cards/src/lib.rs` is now alphabetical; `catalog()` sorts by key, so order there changes nothing.

Compress after `5b8bc02e03`: `Effect::MillCards` now carries both `target: Option<TargetSpec>` and `player: Option<PlayerSelector>`; `PlayerSelector::Target` already covers the first. Folding them into one `player: PlayerSelector` changes every mill card's serialized effect and learning tokens, so it was left for a deliberate hash/token-rename step rather than taken in compress.


For the post-mana-split / pre-closure concept review at `13017dcdf0` (`scratch/concept-review.html`): unify CR 603.8 state triggers as `TriggerCondition::State(Condition)` (reusing existing `Condition::SourceCountersAtLeast` for *Mazemind Tome* while keeping legacy `NoCounters` for pinned-pilot *Dark Depths*), index state triggers on a dedicated `EventKinds::STATE_TRIGGER = Self(1 << 31)` bit in `TriggerIndex` rather than hijacking `EventKind::CountersPut`, and check `pending_triggers` + `trigger_placement` + `stack_objects` for CR 603.8 `waiting`; add `GameOutcome::Draw { mandatory_loop: bool }` under `#[serde(untagged)]` with progress-aware repeated-trigger-signature loop detection in `tick_priority()` (because *Bloodthirsty Conqueror* + life-gain drain + *Herald of Eternal Dawn* never repeats a `GameState` hash), tested with a constructed drain witness to respect the `foundation_composition` boundary plus a finite high-life win control, and fix Academy's `orientBoard` (`recording.ts:87`) so a draw never flips `won: false` to `true`; add `ObjectProperty::Counters(CounterKind)` + `ContextRead::SourceLki` projecting departed source LKI (with a plain-language counter summary) in `effect_execution::project_stack_object` for *Ravenous Amulet*†; ship *Mentor of the Meek*†, *Immersturm Predator*†, and *Fishing Pole*† (via unified `untap_permanent` emitting `GameEvent::PermanentUntapped` + `TriggerCondition::BecomesUntapped`, `ActivatedAbilityDefinition.tap_grantor`, and `Effect::RemoveCounter { from, kind, count, if_removed }`); and retain legacy serialized `Effect::MillCards.target` and `StaticAbility::CharacteristicPower` intact so the 12 existing TLA cards keep exact `test_pinned_pilot.py` compatibility.

Closure slice: Fishing Pole's "remove a bait counter. If you do, create a Fish" is `If { SourceCountersAtLeast(bait, 1), [RemoveCounters { chosen: bait }, create] }` rather than a new `Effect::RemoveCounter { if_removed }`. With no effect in the catalog that replaces or prevents counter removal, removing one happens exactly when the Pole still has one (and a departed Pole has none). A card that replaces counter removal would need the committed result instead.

Closure slice: "equipped creature" is a new `TriggerSubject::EquippedCreature` sharing `EnchantedCreature`'s attachment matching, so descriptions and learning tokens name the printed noun. The two could become one `AttachedCreature` subject if the token distinction is not wanted.

Closure slice: `CostComponent::AddCounters` sits beside `RemoveCounters` rather than one signed counter cost, because `CostComponent` is serialized in mid-payment state and renaming the existing variant would move those hashes.

Closure slice: CR 104.4b detection is local to one `tick_priority` call (automatic passes only, so `skip_trivial` must be on). A loop whose players hold non-pass actions (instants, mana abilities offered at priority) returns decisions every iteration and is never declared a draw; the Env step limit truncates it. The 64-resolution window and 10,000 cap are not CR numbers.

Closure slice: the Ravenous Amulet stack projection reuses `Projection::lki`'s existing summary text, which prints counters as `Counters([(Soul, 3)])`; frozen effect-execution fixtures (`tests/fixtures/effect-execution-v8.json`) hold that text, so a plain-language counter line would change them.

Closure slice: `tests/academy/test_public_commitment_parity.py` (3 cases) fails at this head and before it: the pinned `AUTHORITY_RECEIPT_SHA256` (`39ec37…`) does not match the untouched pre-opening receipt `conformance/authored-match-w17-v2/release-stack-ur-vs-gw-seed-0.json` (`48f7a8…`), and regenerating it from that receipt gives 191 commands, not the pinned 135. Its source receipt predates the opening-hand rules, so refreshing it means choosing the opening authority as its input; that is outside this Task and left as found.


For the post-closure full-branch concept review at `65cebaf342` (`scratch/concept-review.html`): move `MandatoryLoopWatch` onto `PriorityState` (`Option<Box<MandatoryLoopWatch>>`, cleared on empty stack/pending triggers, non-pass action, step start, and suspended/pending choice, with `can_lose(seat)` gating poison as well as life, bumping hash contract to v31) so CR 104.4b works under `skip_trivial = false` and when players pass priority while holding unrelated instants or activated abilities; thread `winner: int | None` into Python Academy's `viewer_view` (`server.py:530`) and `normalize_observation_to_hero` (`trace.py:344` / `recording.py:535`) so draws never flip `won: False` to `True` without churning the `experience-v1` schema; unify grantor-cost authority (`exile_grantor || tap_grantor`) across `mana.rs::reserve_announced_payment`, `observation.rs::action_focus`, and `action.rs::action_description` so two *Fishing Poles* on one creature are distinguishable and safely reserved; and require `trigger.inline_effects.is_none()` in `triggers.rs::state_trigger_waiting` so granted/delayed triggers at `ability_index == 0` never suppress printed state triggers.

Persistent loop-watch slice: a published non-priority decision with more than one action ends the watch, even when its options are equivalent each cycle (two identical trigger orders). Such a loop then runs to the Env step limit instead of a draw. Turn-history counts compare alike above 16 in the fingerprint; no catalog card reads a higher occurrence or threshold. `reserve_announced_payment` now reserves a tap grantor, but no catalog grantor taps for mana or waterbends, so the reservation has no directed test.

Hash v31 verification: after refreshing the Academy fixtures (`scripts/refresh_academy_fixtures.py`, no label changed), four Academy tests still fail for reasons outside this slice's paths: `test_play_modes.py::test_curated_combat_tape_is_identical_live_trace_and_inspector_fixture` and `::test_projection_states_combat_assignments_only_while_they_hold` (they call `scenario_refresh` while the opening-hand Modal is pending), `test_decision_surface.py::test_trace_record_preserves_nested_policy_evidence` (expected record lacks `mana_floats`/`mana_payment` and keeps `recording_import: None`), and `test_shared_table.py::test_watcher_branch_is_owner_only_while_pilot_advances_live_match` (`StopIteration` finding an offer). Not checked against the base build; gate should confirm whether they fail on main.


For the post-alignment full-branch concept review at `31c81eda80` (`scratch/concept-review.html`): keep hash contract v31 intact and fix four CR 104.4b loop-watch seams plus one tap-grantor reservation split before behavior review: (1) clear `PriorityState.loop_watch` in `execute_action` only on `Ok(())` so rejected/stale priority actions never mutate `witness(&game)`; (2) skip `watch_for_mandatory_loop()` in `tick_priority()` when `suspended_decision.is_some()`; (3) in `GameState::loop_fingerprint()`, mask only `"poison"` inside `players[*].counters` (dropping empty `"counters"`, keeping any future non-poison player counters), strip `"id"` on `stack_objects[*]` entries alongside `pending_triggers`/`trigger_placement`, and saturate only the count element (`entry[2]`) in `ability_triggers_this_turn` / `ability_resolutions_this_turn`; (4) set `sacrifice: true` on `tap_grantor` reservations in `mana.rs::reserve_announced_payment` and `announced_model` to match `owed_reservation` and `payment_reservation`.

Finding 17 verification (`31a2d7dbd3`, rebuilt cp313 extension): `test_authored_match_parity.py`, `test_choice_navigation.py` and `test_learn_setup.py` pass with no fixture refresh. `test_learn_evidence.py::test_same_tape_parity_and_tampered_persisted_command` fails: live/headless `ParityDivergence` at revision 1 (opening), seat 1's `viewer_state_sha256`/`viewer_observation_sha256` only, with identical state witnesses. The loop watch counts no resolution there and its owed flag enters no projection, so this predates the fix; not checked against the base build. Gate should confirm it on main alongside the four Academy failures above.

For the post-Finding-17 full-branch concept review at `b817943ea0` (`scratch/concept-review.html`): move `end_mandatory_loop()` in `tick_priority()` after `stabilize_before_priority()` and `is_game_over()` (simplifying its empty check to `self.stack_is_empty()`) so CR 603.8 state-trigger loops, non-batch `emit()` loops (`PermanentUntapped`), and SBA-death loops are counted instead of clearing `loop_watch` every cycle, and terminal SBAs take precedence over `CAP`; unify all watch/owed-flag clearing in `PriorityState::clear_loop_watch()` so real multi-option choices and non-pass actions clear `resolution_owed` alongside `loop_watch`; and keep `resolution_owed` as `#[serde(skip)]` on hash contract v31 (it is always `false` at priority and multi-option decisions, and `Clone` preserves it across forks and snapshots).

Finding 18 fixture choice: keep Lightning Bolt castable during both new trigger loops and pass every offered priority with forced decisions skipped. This bounds a regression by the number of passes even if the engine never starts its watch. The untap fixture starts with the Ogre's ordinary untap step.

## Sync onto main at `912f8e622a`

Main and this branch both added v11 permanent columns at position 52. The
merge keeps main's first-strike participant at 52 and puts the three
branch history columns after it. The legacy permanent layouts keep their
existing widths.

Both catalogs defined the same Cat, Goblin, Faerie, 5/5 Dragon, green Elf
Warrior and Zombie printings. The merge keeps main's keys for those
printings and updates the branch's card definitions to use them. Main's
attachment subject also covers the Fishing Pole untap trigger. Life-gain
listeners share the branch's ordinal-bearing trigger, with no ordinal
restriction for listeners that run on every gain.

## Decomposition

`TargetSpec::GraveyardCardYouOwnWithinSourcePower` stays a spec variant. A
`TargetRequirement.max_mana_value: Option<EffectValue>` would need target
legality to evaluate values without a resolution frame, and only
source-relative values would be readable there. I took the simpler option of
keeping one reader for three cards, and I'd revisit when a second dynamic
target bound appears.

Lumen-Class Frigate already encodes its station amount as
`EffectValue::PowerTappedToPay` in released definitions. I kept that token
and its nonnegative contribution per tapped creature. The existing receipt
and stack amount field hold everything the pending ability needs.

## Sync onto main at `610370233c`

Keep main’s effective mana value, prepared-cast and flashback stack columns at
37–39. Append MTG-418’s six history columns at 40–45 so stored column prefixes
continue to bind by name.

Quilled Greatwurm reads one committed combat total per exact source through
the shared trigger occurrence path. The plain black Rat keeps its own
`token.black_rat` key for Revenge of the Rats, and `token.rat` keeps the
Redcap Gutter-Dweller printing that cannot block.

The combined replay corpus retained all 14 main cases and 3,150 Commands,
with identical per-case counts and outcomes. The Academy authority tape
retained all 174 Commands, offer counts and winning seat. Academy's four
demo prefixes and five navigation prefixes replayed to their recorded
decision kinds.

## Post-merge concept review at `8838269f91`

For the post-merge full-branch concept review at `8838269f91` (`scratch/concept-review.html`): stay on hash contract v31 and close three post-merge mismatches before `gate`: (1) switch *Quilled Greatwurm* (`fdn.rs:3407`) from `EffectValue::X` to `EffectValue::TriggeringAmount` (keeping `DealsCombatDamage { to: None }` + `CombatDamageTotals` for CR 510.2 trample totals, stopping `triggers.rs:1010` from writing `CombatDamageTotals` into `trigger.x` / clamping at 255, deleting `bound_x`'s Greatwurm special case, and threading `triggering_amount` into `trigger_description` so simultaneous *Drake Hatcher*, *Bloodthirsty Conqueror*, *Niv-Mizzet*, *Lathril*, and *Quilled Greatwurm* triggers show their exact `i32` amount in `OrderTriggers` and stack labels); (2) add `#[serde(default, skip_serializing_if = "SpellProfile::is_front")] pub profile: SpellProfile` to `SpellLki` (zero front-face hash change), share one `recorded_spell_matches_predicate` helper across `recorded_cast_matches_predicate`, `Condition::TriggeringSpellMatches`, and `subject_matches_damage_source`, and resolve the cast profile name in `effect_execution::spell_lki`; (3) add exhaustive `TriggerCondition::inner(&self)` (including `Prepared`, no wildcard) and route `once_each_turn`, `condition_event_kinds`, `triggering_noun`, and `bound_x` through it. Also sync `docs/observation-and-model.md`'s v11 Layout table widths (`40 × 56`, `32 × 46`, `2 × 15`).

## Post-merge receipt selection and waiting-seat labels

The merged conformance README identifies the opening-hand and casting-only
receipts as historical. Refresh the active casting-history receipts and the
live Academy fixtures (which still have opening filenames). The pre-change
semantic binary reproduced all 14 recorded final witnesses.

The directed Hatcher check reached the right LLM label but Academy rejected
seat 1's frame: “decision context is not oriented to its acting viewer”.
The public stack entry already carries the exact amount. Keep the pending
decision narration with the acting seat, as the wire validators require.
