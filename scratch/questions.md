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

`managym::semantic_conformance_tests frozen_sacrifice_162` and `frozen_sacrifice_163` fail at ab34e99193, before the life-batch compression: the regenerated case hash differs from `conformance/semantic-kernel-w17-v6/evidence-opening-v1.json`. Gate should find which life-batch change moved the trace (activation terms, firebending as a granted trigger, or the heuristic's live attackers) and either restore it or record the shift.
