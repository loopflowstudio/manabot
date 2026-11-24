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

For the next batch, use the three second-draw listeners together. Their shared crossing and three different outcomes can be checked from one draw sequence.

The two-viewer text fixture uses Lunar Insight to draw three in one resolution. Wizard, Horde and Mystic supply its three distinct nonland mana values.
Start the attack batch with Linden and Chart a Course to distinguish per-attacker triggers from raid across a partial declaration and restore.

Sync onto main (dca0a901b6): state hash is now v26, combining main's v19 opening/victory fields with the branch's history fields. Main's `entered_this_turn` keeps permanent column 51; the two combat-damage flags move to 52 and 53. Incubation counters are appended after main's Poison and Plan. Two kinds of failure in the debug workspace run do not come from conflict resolution. First, four TLA activated-target tests (`disciple_*`, `north_pole_patrol_*`, `protectors_sacrifice_*`) fail because `activated_ability_targets` now calls `targets_for_source`, which leaves out the source only for `another` clauses. Main always left out an activated ability's own permanent. Second, the frozen semantic conformance cases (`evidence-opening-v1.json`) need re-recording under hash v26.

The attack and raid fixtures stay together in `managym/cards/tests/cards/fdn.rs`, including the Aurelia cases. They share the declaration helpers and all twelve ledger rows link here.
