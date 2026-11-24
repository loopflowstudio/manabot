# Planning assumptions

The checkout began without a scratch design. Use `scratch/plan.md` for the working phase and preserve the committed FDN set research. Use the task base to identify the original cohort after ledger statuses change. The interaction pairings introduced in the plan are proposed fixtures for integrated validation.

Selected Lightshell Duo first to check the existing private ordering contract through every consumer. Scout's Warning supplies the follow-up draw and triggers prowess in the same position.

The compression pass covers the delivered Lightshell Duo slice. The Python fixture keeps a fifth Island in hand so the player still has a decision after surveil resolves.

Eleven Bucket C rows are confirmed. Joraga Invocation requires whole-declaration block requirement maximization. Confiscate requires an attached continuous control effect in layer 2. Knight of Grace, Knight of Malice and Elenda, Saint of Dusk require hexproof qualified by source color or type. The permanent target check in `flow/resolution.rs` reads a single unconditional hexproof flag for opposing controllers. Taurean Mauler and Soulstone Sanctuary need every-creature-type membership. Curator of Destinies needs two retained piles with different disclosure and an opponent choice. Dryad Militant needs a battlefield source to redirect qualifying graveyard moves from any zone. Darksteel Colossus needs its own reveal-and-shuffle replacement from any zone. Run Away Together needs a controller comparison between targets at casting and resolution.

Audit each capability group before authoring it. For Arahbo and Helpful Hunter, fork at the shared trigger-order decision so both outcomes start from identical permanents and library order.

Use Foggy Swamp Visions to copy Arahbo from the graveyard without another Arahbo on the battlefield. For Caracal, create the ordinary Cat directly from the same token face Arahbo uses so a second lord does not obscure the change in power and toughness.

Combat assignments use the existing modal Command. For multiple blockers the attacker chooses an amount per blocker in declaration order, so the engine never materializes the Cartesian product of allocations.

Replay doubling first without Giant Cindermaw as the positive control for prohibited lifelink.

Use the public question “does another damage step follow?” as the damage-pass flag. The existing step identifier already names combat damage, and the flag stays false during ordinary single-step damage.

Model Fog Bank’s single prevention ability with one either-endpoint identity check on combat damage.

Twinflame Tyrant uses controller scopes and a damage transform. Add the source predicate and combat endpoint restrictions with the cards that exercise them. Wall of Stone survives a doubled Lightning Bolt, so the same test can inspect marked damage in every creature case.

The controller fixture changes the Bolt’s controller after targets are chosen. A single point of prevention makes both replacement orders distinguishable on the fork.

Authority uses an opponent-scoped entry ability so Zhao keeps its existing serialized predicate. The fixture changes the resolving spell's controller and places Authority on either seat to cross ownership and control in both creature cases.
Use Typhoid Rats as the untapped control. Both Innistrad creatures cost {B}, so the fixture needs no generic mana choice.

Use the first printed red 1/1 Goblin token face (Unglued) and file the cross-set fixture with Impact Tremors, whose triggers supply the response windows.

Fill the Batch 5a cohort with the ledger’s vanilla and keyword-only creatures. Include Primeval Bounty and the three Treasure producers. Keep modal and targeted removal spells in their planned Batch 5b.

Use the 23 spells named in Batch 5b as this review cohort. Keep unnamed siblings in the remaining audit queue. For Deadly Plot, Hungry Ghoul ordinarily enters untapped so the return instruction determines its orientation.

Use the 15 explicitly named Batch 5c permanents as the review cohort. Fork Soul-Guide Lantern before activation to compare its two sacrifice abilities from the same graveyards.

Batch 5d uses the 38 cards named in the plan. Reclamation Sage's Magic 2015 promo released July 17, 2014, one day before the core set printing, so its first-printing module is pm15.

Use two friendly Gray Ogres and one opposing Gray Ogre for the six unconditional attachment grants. The second friendly creature distinguishes a wearer-only grant from a controller-wide grant.

Batch 5e follows Scryfall's earliest non-digital printing for each set file (Opt in INV, Offer Immortality in a new ONE module). Planeswalker target legality uses a permanent retyped in the test, since no planeswalker card is defined.

Mode labels now come from `effects_description` when no idiomatic label exists. Some generated text is rough ("Draw 1 cards", Bushwhack's empty first clause, Crawling Barrens' "This object"); improving the shared description templates is left for a text pass.

Use the verified Batch 5e checkout for compression. Compare the rebuilt extension's full definition JSON and catalog token table against a62143faef to check the shared pump constructor.

Audit the remaining 57 rows in three passes. Put the 28 triggered creatures first so they can share entry, death and attack fixtures. Keep the 24 activation, characteristic, combat and choice rows together for the follow-up, and the five replacement rows together for the event audit.

Alignment 3 removes whole-combat prohibitions at `can_block` and leaves the declaration tick alone. A blocker that can block none of the current attackers (a ground creature against a lone flyer) is still asked with only `Decline` when `skip_trivial` is off. With `skip_trivial` on it is skipped. Changing the tick would also move the structured blocker offers, combat previews and the semantic corpus.

The compound attacker offer still selects attackers with `min: 0`. A submission leaving out a Juggernaut that must attack fails as stale or illegal at the legacy `DeclareAttacker` step instead of being prevented by the offer. Expressing a required candidate in `ChoiceStep::Select` is a protocol change.

Batch 5f follow-up choices:

- Campus Guide and Spinner of Souls express "you may" as `Effect::Modal` with an empty second mode (Solemn Simulacrum's form), not the zero-cost `OptionalPayment` the plan sketched. Both forms exist in the catalog; Modal is the one the "you may search" cards already use.
- `Effect::Reveal` keeps its library knowledge through a frame flag, `selected_revealed`, rather than a list of revealed objects: a `Vec` in `EffectFrame` exceeded the `GameState` size budget. Only a reveal of the `Selected` objects carries into a later top-of-library placement.
- Activated abilities now offer their own source as a target unless the clause says "another" (`TargetRequirement::another`). Deserter's Disciple, Earth Kingdom Protectors and North Pole Patrol relied on the old blanket exclusion and now state `another` through `Effect::another_target`, which changes those three definitions' serialization and tokens.
- `Revisions` gained `life` for `Condition::LifeAtLeast`; the `TrackedState` and `DerivedCache` size ceilings rose by 8 bytes for the fifth counter.
- Reveal-until with exactly one passed card reuses `place_entrant_on_bottom`, which keeps the facts below it but records no public fact for the revealed card now on the bottom (lossy, never wrong).
- The blocker compound offer also falls back to exact atomic offers when an attacker must be blocked by every able creature, since a later blocker's decline is then illegal.

Concept-review (`49cc87c7`) choices folded into Batch 5g:

- Pre-declared combat participants (`skip_trivial: true`): add symmetric `declared_blocks()` tolerance in `combat_preview.rs` (no wire-schema change; conflicting blocks for a pre-declared blocker are still refused) and feed already-committed attackers/blockers from the live frame into Academy's `draftWords()` so the confirm button never says `"No attack"` while `Juggernaut` is attacking.
- Spell-created prevention shield (`Fleeting Flight`): add a target-only `ReplacementEffect::PreventCombatDamage` variant (distinct from `Fog Bank`'s two-way `PreventCombatDamageToOrFromSelf`) and wire `Modification::PreventCombatDamage` across all 5 observability/text layers including `text_observation.py`.
- Single-passed-card `reveal_until_to_hand`: record the publicly revealed `[top]` card at `public[0]` and `own[0]` right before `place_entrant_on_bottom(*top)` so both viewers keep the exact bottom fact.


## Batch 5g choices

- A resolved shield's `ReplacementId` carries the record's timestamp in an optional `effect` field (skipped when absent, so printed identities serialize unchanged); `8068a268d7` replaced the first form that overloaded `static_ability: true`.
- Fleeting Flight's effect description reads "Prevent all combat damage that would be dealt to target creature until end of turn" (the shared `until` suffix), where the Oracle says "this turn".
- The curated decision-memory witness moved from seed 197 to 208, the only seed in 190–259 still live at step 600 with more than 250 distinct contexts; it is a fixture choice, not a property of the seed.

## Post-build concept-review (`6a572f60`) choices

- Combat damage assignment numeric encoding: reuse existing `announces_x` → `action::HAS_X` / `action::X` (no v11 layout change) so amounts ≥ 15 stay distinct instead of collapsing into `one_hot:15`'s `"above 15"` bucket, while keeping `AssignCombatDamage`'s custom description ahead of `"X = {mode}"`.
- Academy placement of `assign_combat_damage` offers once `Observation::action_focus` carries `[attacker, blocker]`: keep them as rail buttons via `view.ts:isButton` and highlight `[attacker, blocker]` via `askedObjects` rather than anchoring 2..N damage choices inside a popup menu on the attacker card.

The compression pass started at `57c18aa9d0`, which already contained Alignment 5 and its rebuild evidence. The semantic corpus refresh remains with gate.
