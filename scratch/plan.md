# MTG-421 · Ready-now Foundations

## Working progress

Of the 222 cards selected at base `0b1cc9cfef9c1a98430c825b9764a49237182696`, three already had definitions (Kellan, Planar Trailblazer, Imprisoned in the Moon and Doubling Season). This branch added Lightshell Duo, Arahbo, Helpful Hunter, Regal Caracal, Prideful Parent, Anthem of Champions and Ajani's Pridemate. Twelve cards have documented deferrals and 200 await audit.

**Arahbo and Helpful Hunter.** Both native trigger orders resolve from the same position. Foggy Swamp Visions exiled Arahbo from the graveyard. Its token copy entered alone and created exactly one 2/2 Cat. The learning-token streams differed with and without the self-entry alternative. With Scout's Warning available, both text Commands put the named source on the stack first, below the other trigger. Each replay reproduced the digest and rejected stale resubmission. Both choices were valid in v11. The draw was visible in the owner's hand and hidden from the closed opponent's observation and Academy projection.

**Regal Caracal.** The Amonkhet definition creates two white Cat tokens with printed lifelink. Alongside the ordinary Cat all three were 2/2 with lifelink. Removing Caracal returned all three to 1/1 and only the two new Cats kept lifelink. Caracal and the opposing Cat received no lifelink from its ability.

**Behavior review at `edac17e6`.** Four directed Rust tests passed in the debug test profile: the two Arahbo positions, the self-entry learning-token comparison and Caracal's removal. Five Python cases passed (the Hunter draw and four Lightshell cases).

**Prideful Parent and Anthem of Champions.** At `7b808969`, the directed position in `managym/cards/tests/cards/fdn.rs` checks Parent and its Cat before Anthem enters (2/2 and 1/1), while it is present (3/3 and 2/2), and after it leaves (2/2 and 1/1). Parent keeps vigilance and the opposing Cat stays 1/1 at all three points. Both ledger rows cite the fixture. The existing entry trigger, ordinary Cat token and controller-scoped continuous effect cover the interaction.

**Drakuseth audit complete.** Its ledger row remains pending. Target selection permits reuse across clauses and resolution cannot assign 4 and 3 damage to their respective recipients in one batch. The choice-continuations obligation below covers both gaps.

At `3a54967f`, both directed debug fixtures passed again. The target-submission fixture rejected the first permanent twice in one clause and accepted it again in the next clause. This reproduces the selection gap independently of the damage-batching finding from code inspection. Parent and Anthem satisfy their working obligation. Full consumer and conformance suites remain with gate.

### Behavior after the shared repair at `12096ae8`

**Surveil knowledge after resolution.** Milling either inspected card shifts the other to known position 0 for the chooser. Keeping both and reversing their order changes only positions 0 and 1. Hill Giant and Forest, previously placed by Accumulate Wisdom, remain known at positions 3 and 4. Four Lightshell cases passed on the release extension rebuilt from merged snapshot `8aef2be7`, covering native observations, v11, text, closed-opponent privacy and replay on independent forks.

**Focused evidence from constructed positions**

- The FDN definition uses the ordinary enters trigger with `Surveil { count: 2 }`. The chooser can mill either card followed by the other, mill just one, or keep both in either order. The debug FDN fixture draws the chosen first card with Scout's Warning, after prowess resolves to make Duo 4/5. Duo returns to 3/4 by the next turn. These outcomes agree with the published mechanics. [Surveil](https://magic.wizards.com/en/news/feature/tarkir-dragonstorm-mechanics) · [Prowess](https://magic.wizards.com/en/news/feature/magic-origins-mechanics)
- The chooser gets three initial offers and two ordering offers, with matching v11 valid-action counts and distinct object focus. Each tested Command gave the same transition and digest on independent forks. Reusing the capture was rejected with the root and committed branch intact. Swapping the private top cards, or selecting opposite cards before committing, produced identical opponent semantic JSON, Academy projections and v11 buffers.
- A separate `GameSession.current_recovery("reconnect")` probe at the constructed Lightshell root recovered both named mill offers and **Keep on top**. Submitting **Keep on top** recovered the two bound ordering offers. Choosing Hill Giant first put it on top. The Academy session returned the duplicate receipt and rejected a new Command using the old revision at each transition. The committed Python fixture directly exercises `decision_surface_view`; this session probe exercised the envelope.
- Debug tests passed for the shared surveil completion event, ordered continuation and private fork behavior. The prowess fixtures passed for Swiftspear, the Monk token and Sokka's grant, including the own-noncreature control in the Iguana/Boar fixture. The constructor and predicate produce the existing rules structures. The catalog returned two triggered program spans for Duo, and its prowess span matched Swiftspear token for token. Nine learning-token observability checks and the registered-card Scryfall conformance check passed. The two text fixtures that now share the replay helper also passed. This slice changes no payment or receipt semantics.

Snapshot library-order pins retain inspected object identities across exact forks and replay. Viewer observations expose only the positions known to that viewer.

At `2aa72769`, the rebuilt release extension passed all four Lightshell cases in `tests/semantic/test_text_observation.py` (74 deselected).

**Run Away Together remains deferred.** With two creatures controlled by one player and none by the other, the spell has no legal pair. With one creature per player, changing control of one in response makes both targets illegal. If one leaves instead, its last known controller determines whether the surviving creature returns. These are acceptance positions for the missing definition. Native `TargetRequirement` has no different-controller relation, and `resolution_targets` rechecks each target independently except for shared creature type. The shared targeting authority needs the relation at selection and resolution. [Foundations release notes](https://magic.wizards.com/en/news/feature/foundations-release-notes)

**Ajani’s Pridemate.** The directed fixture `pridemate_triggers_once_per_lifelink_source_in_a_damage_batch` in `managym/src/flow/proposed_event.rs` passed. Damage of 2 to a creature and 3 to its controller produced one gain of 5 from a single source, or gains of 2 and 3 from distinct sources. Pridemate resolved to 3/3 and 4/4 respectively. The fixture submits the damage batch directly. Full debug, pinned-pilot compatibility and browser checks remain gate evidence.

**Combat assignment at `e4784f29`.** Before the repair, a 5-power Foggy Swamp Spirit Keeper blocked by Gray Ogre gained only 2 life. The damage batch had already discarded the other 3 before lifelink ran. The passing regression `combat_assignment_deals_all_five_damage_as_one_lifelink_gain` in `managym/cards/tests/cards/tla.rs` now observes 5 damage to the sole blocker without trample. With trample it checks all four blocker/player offers (2/3 through 5/0). The shared Command helper submits 2/3, checks stale rejection and compares the independent fork witness. Combat ends at 25/17 life with one gain of 5 and one Pridemate counter. The shared assignment path in `managym/src/flow/combat_actions.rs` builds the damage batch for both cases.

**Behavior review at `5c1aa5be` — iterate.** The directed combat regression passed again. The native modal decision enumerates the inclusive damage range and focuses the public attacker and blocker. Its labels name both recipients. Continuation code 32 identifies the decision in policy relations. The independent fork witness is checked immediately after the Command, and the original scenario checks the resolved gain and counter. Tensor, LLM and Academy rendering were not exercised in this run.

**Two-attacker combat fixture delivered.** Two Foggy Swamp Spirit Keepers attack together, one with a +1/+1 counter. The fixture selects the second Pridemate trigger through a bound Command, checks stale resubmission and resolves the original and independent fork through end of combat. Both record gains of 2 then 3 and finish with two Pridemate counters. The focused native card test passed.

The decision-code fixture now takes the current object identity of each deck card. Both previously blocked library checks passed, covering the decision-code bound and per-source gain in a constructed damage batch.

## Experience and model to build

Authoring Foundations cards should be direct, predictable, and collision-free: one canonical way to write each static effect, token face, lifegain trigger, and replacement effect, backed by an honest readiness partition of all 222 rows so work never stalls mid-slice on hidden engine gaps or fragments into card-shaped enum variants.

### Canonical authoring model (and choices made for review)

1. **Capability-by-capability readiness partition of the 222 rows.** `fdn.jsonl`'s `dependencies == []` tag came from a text-only planning scan (*"No games, tests or benchmarks were run"*). The three existing definitions are reconciled. Classify the remaining rows against native selection, execution and observation. The task brief allows “minimal local primitives”. The designs below are candidates. Verify their callers and scope before admitting them.
   - **Bucket A — Composable today:** Zero engine edits needed; batch-author in capability-grouped slices of ~15–35 cards instead of 1–2 cards per slice.
   - **Bucket B — Unlocked by minimal local primitives:** Land each shared primitive once with directed counterexamples, observability guards, and learning tokens, then batch-author every FDN card it unlocks.
   - **Bucket C — Explicitly deferred:** Record as `definition pending` in `fdn.jsonl` with the exact missing authority and owning parallel track. Do not add one-off shortcuts.
   - *Choice made for review:* Admit a proposed primitive to Bucket B once the audit demonstrates a bounded native change and a directed interaction that distinguishes it. Keep larger missing authorities in Bucket C.

2. **One continuous static authoring form (`StaticAbility::Continuous`).** `managym/src/flow/layers.rs` already lowers every legacy `StaticAbility` wrapper (`PtBuff`, `KeywordGrant`, `TypeGrant`, `AttachedSetsCharacteristics`, `AttachedHas`, `TriggeredGrant`, `SetsLandType`) into one `Node`/`Modification` layer pass, while `AttachedSetsCharacteristics` cannot express color or card-type changes (`Witness Protection`, `Eaten by Piranhas`).
   - Author **every new FDN static characteristic effect** with `StaticAbility::Continuous(StaticContinuous { .. })`. Add `Modification::SetName(String)` for name-overwriting Auras (`Witness Protection`). If any FDN Equipment grants an activated ability referencing `ObjectSelector::Grantor`, populate `granted_by` for `StaticScope::Attached` in `layers.rs` (otherwise `Modification::GrantActivated` already covers ordinary grants like `Imprisoned in the Moon`).
   - *Choice made for review:* Keep existing TLA/demo cards on their legacy `StaticAbility` variants so `tests/model/test_pinned_pilot.py` and stored checkpoints remain untouched; apply `StaticAbility::Continuous` uniformly to all new FDN cards.

3. **Characteristic-qualified keys for new creature tokens.** Existing bare keys in `managym/cards/src/tokens.rs` were already claimed by non-vanilla TLA/demo faces (`token.dragon` has firebending 4; `token.soldier` is a 2/2 red firebender; `token.spirit` has a block restriction; `token.bear` is 4/4; `token.elf_warrior` is G/W), and Scryfall conformance skips tokens.
   - Reuse predefined game tokens (`token.food`, `token.treasure`, `token.clue`, `token.powerstone`) and preserve every existing key and payload untouched.
   - Give every **new** non-predefined creature token in `managym/cards/src/tokens.rs` a self-describing, characteristic-qualified key: `token.<name>_<p>_<t>_<colors>[_<ability>]` (e.g., `token.cat_1_1_w`, `token.cat_1_1_w_lifelink`, `token.knight_1_1_w`, `token.soldier_1_1_w`, `token.dragon_4_4_r_flying`, `token.elf_warrior_1_1_g`, `token.goblin_1_1_r`, `token.zombie_2_2_b`, `token.dog_1_1_w`, `token.beast_4_4_g`, `token.insect_1_1_bg_flying`, `token.faerie_1_1_u_flying`).
   - *Choice made for review:* Characteristic-qualified keys over ad-hoc adjectives (`plain_dragon`, `lifelink_cat`), so future Standard sets never collide.

4. **Per-source lifelink and life-gain triggers.** `damage.rs` reads source keywords once for lifelink and deathtouch. During batch commitment, `proposed_event.rs` totals dealt lifelink damage by source object and controller and emits one `LifeChange` proposal per total. Replacement choices on the incoming damage finish before commitment, so the accumulator can stay local. `GainsLife { who }` matches a positive committed life change for the selected actor. Use this condition for the remaining life-gain cards.

5. **Composable damage and entry replacements.** `ReplacementEffect` currently has card-shaped damage variants (`DoubleDamageToController`, `PreventDamageToController` — 0 real-card callers — plus `OpponentDamagePlusCounters`), and `StaticAbility::EntersTapped(CardPredicate)` applies to both players (cannot express `Authority of the Consuls`).
   - Add one composable damage replacement (`ReplacementEffect::ModifyDamage { source_by, source_predicate, recipient_by, combat_only, transform }`) covering `Twinflame Tyrant` (your sources → opponents/their permanents → double), `Gratuitous Violence` (your creatures → any → double), and `Fog Bank` (combat damage dealt to or by this → prevent all).
   - *Choice made for review:* Migrate `DoubleDamageToController` and `PreventDamageToController`'s unit tests in `resolution_tests.rs` to `ModifyDamage` and retire their names in `managym/src/cards/token_renames.rs` (verifying `test_pinned_pilot.py` and `record_vocabulary.rs`). Keep `OpponentDamagePlusCounters` untouched if TLA cards depend on its tokens, or verify pinned-pilot impact before touching it.
   - Scope `EntersTapped` with `who: Actor` (preserving existing TLA `Actor::AnyPlayer` behavior and pinned-pilot tokens via default/serde compatibility, or adding an opponent-scoped entry replacement if renaming `StaticAbility::EntersTapped` would alter Zhao's tokens — verify against `test_pinned_pilot.py` before editing the existing variant!).

6. **Combat constraints boundary.** Local prohibitions (`Vampire Interloper`, `Stromkirk Noble`) belong here (ensure `eligible_blockers` in `combat_actions.rs` also respects unconditional `CantBlock` cleanly). `Juggernaut`'s solo `MustAttack` can be implemented locally only if verified against tapped / summoning-sick / Pacifism counterexamples and documented as bounded by the absence of attack taxes/caps; `Joraga Invocation`'s global requirement maximization (CR 509.1c) belongs in Bucket C (whole-declaration solver).

## Which cards to deliver

Use the 517-row [FDN ledger](../.lf/wave/coverage/research/standard-2026-11/sets/fdn.jsonl) and [set notes](../.lf/wave/coverage/research/standard-2026-11/sets/fdn.md). Selection refers to **checkout base `0b1cc9cfef9c1a98430c825b9764a49237182696`**: `implementation.status == "definition pending"` and `dependencies == []` across these five primary batches:

| Primary batch | Base rows | Representative examples |
| --- | ---: | --- |
| `foundation_composition` | 136 | Helpful Hunter, Stab, Day of Judgment, Arahbo, Dragon Fodder, Impact Tremors, Ajani's Pridemate |
| `characteristics` | 38 | Anthem of Champions, Regal Caracal, Kellan (base), Imprisoned in the Moon (base), Witness Protection, Eaten by Piranhas |
| `choice_continuations` | 31 | Lightshell Duo (done), Refute, Burglar Rat, Opt, Faebloom Trick, Uncharted Voyage |
| `replacement_events` | 13 | Diregraf Ghoul, Doubling Season (base), Twinflame Tyrant, Gratuitous Violence, Fog Bank, Authority of the Consuls |
| `combat_constraints` | 4 | Vampire Interloper, Stromkirk Noble, Juggernaut, Joraga Invocation |

### Bucket C deferrals and audit candidates

Leave pending in `fdn.jsonl` with exact missing authority and owning track:
- **Global block-requirement solver (CR 509.1c):** `Joraga Invocation`.
- **Layer 2 static control change:** `Confiscate`.
- **Quality-scoped hexproof:** `Knight of Grace`, `Knight of Malice`, `Elenda, Saint of Dusk`.
- **Every creature type (all zones for Mauler, animated battlefield characteristics for Sanctuary):** `Taurean Mauler`, `Soulstone Sanctuary`.

- **Pile formation and handoff:** `Curator of Destinies`. LookAndSelect has one chooser and moves selected cards to hand. Curator needs two piles that survive a change of chooser, separate disclosure for each pile, and movement determined by the second choice. Defer to the choice-continuations track.

- **All-zone graveyard replacement:** Discard uses `move_object` and mill batches zone-move proposals. Stack departures also use the zone-move boundary, which handles flashback. In `flow/proposed_event.rs`, graveyard redirection in `normalize_proposal` is restricted to battlefield departures. Replacement discovery has no graveyard-destination arm for either Dryad Militant’s battlefield ability or Darksteel Colossus’s own reveal-and-shuffle instruction. Both belong to the replacement-events track.

- **Different controllers:** Run Away Together needs one two-creature targeting clause. In `flow/play.rs`, `remaining_legal_targets` excludes previously chosen objects but accepts another creature with the same controller. In `flow/resolution.rs`, `resolution_targets` rechecks each slot and has a group check only for shared creature type. Separate caster-relative clauses would wrongly reject a pair whose creatures exchanged controllers before resolution. Defer to choice continuations.

- **Drakuseth, Maw of Flames (choice continuations).** The native test `trigger_target_distinctness_is_enforced_per_clause_at_submission` rejects duplicates within a clause and accepts the same object again in the next clause. Retargeting also checks distinctness only within the clause. The requirement’s `another` excludes the source, which is itself a legal target here. At resolution, `DealDamage` assigns one amount to every legal recipient of its active clause. Separate clauses could assign 4 and 3, but would commit separate damage batches for one simultaneous instruction. Pending support for excluding targets chosen in an earlier clause and applying both amounts in one batch. Fork with three targets bound and remove the primary in one branch. The two surviving secondary targets must each receive 3 in the same batch.

Candidates still to verify:
- **Reveal-until-matching with random bottom:** `Spinner of Souls`.
- Any additional row identified in Step 0 whose printed text requires a substantial new authority from a parallel track (e.g., verify `Circuitous Route` basic-or-Gate union search, `Campus Guide` search-shuffle-then-top, `Goblin Oriflamme` / `Kitesail Corsair` while-attacking static, `Pirate's Cutlass` entry attach).

## Execution sequence

Every slice follows the same authoring rules: file each card once under its **first printing's set module** (`managym/cards/src/<set>.rs`) with its Scryfall printing ID, register via `cards()` and `cards/src/lib.rs`, snapshot shell + Oracle text in `managym/tests/fixtures/scryfall_cards.json`, update `fdn.jsonl` (`status: "source definition found"`, exact source path), and add directed set tests covering normal paths and counterexamples.

### Slice 2 — Per-source life gain

**Next slice.** Follow a replacement choice through damage commitment and Pridemate resolution. Use damage actually dealt as the distinguishing assertion. Full prevention yields zero damage, zero gains and zero counters. A prohibition on life gain preserves the damage but still yields zero gains and zero counters. The two-attacker ordering and recovery fixture already covers ordinary combat.

- Fully prevented damage and `PlayersCantGainLife` each yield zero gain triggers. Fork at a pending replacement choice and resolve each branch to check the gains due for its committed damage and that the root stays intact. Replay the same replacement choice on an independent fork and compare the committed gains, pending stack order after the same ordering decisions, and resulting Pridemate counters.
- Author Marauding Blight-Priest and Drogskol Reaver using the shared trigger. Exercise the Priest’s opponent life loss and Reaver’s draw. For an unblocked Reaver, verify a gain and draw in each of the first-strike and normal combat damage steps.
- Check the life-gain condition in the catalog learning tokens and its viewer text. Keep the consumer and conformance suites at gate.

### Slice 3 — Composable Replacements & Token-Multiplier Proof
1. Implement `ReplacementEffect::ModifyDamage` (migrating/retiring test-only `DoubleDamageToController` and `PreventDamageToController` cleanly with `token_renames.rs` and checking `test_pinned_pilot.py`) and opponent-scoped entry-tapped replacement (preserving Zhao's pinned tokens).
2. Add remaining FDN token faces needed for this slice (`token.goblin_1_1_r`, `token.knight_1_1_w`, `token.soldier_1_1_w`, `token.dragon_4_4_r_flying`, `token.elf_warrior_1_1_g`, `token.zombie_2_2_b`, `token.dog_1_1_w`, `token.beast_4_4_g`, `token.insect_1_1_bg_flying`, `token.faerie_1_1_u_flying`).
3. Author **Twinflame Tyrant**, **Gratuitous Violence**, **Fog Bank**, **Authority of the Consuls**, **Diregraf Ghoul**, **Dragon Fodder**, **Impact Tremors**, **Heroic Reinforcements**, and remaining ready replacement cards (`Elspeth's Smite`, `Scorching Dragonfire`, `Obliterating Bolt`, `Grappling Kraken`, `Fleeting Flight` if local).
4. Verify:
   - **Doubling Season + Impact Tremors + Dragon Fodder:** 4 independently identifiable `token.goblin_1_1_r` enter and fire 4 Tremors triggers (vs. 2 without Season).
   - **Doubling Season boundary:** effect counters vs. loyalty cost vs. resolving permanent-spell copy becoming a token.
   - **Twinflame Tyrant:** your source → opposing creature/player doubles; friendly damage and opposing source do not.
   - **Fog Bank:** prevents combat damage both dealt and received; noncombat damage still applies.
   - **Authority of the Consuls:** opponent's creature enters tapped + controller gains 1 life; controller's own creature enters untapped.

### Slice 4 — Remaining Bucket B Primitives & Characteristics / Combat / Choice Cards
1. Add small missing Bucket B primitives confirmed in Step 0:
   - `Modification::SetName(String)` for `Witness Protection` (and verify `Eaten by Piranhas` via `StaticContinuous`).
   - Small `Condition` / `CardPredicate` / `TargetSpec` / library-end additions where one clean primitive unlocks multiple FDN cards (e.g., `Condition::LifeAtLeast`, `TargetSpec` planeswalker combinations for `Hero's Downfall` / `Viashino Pyromancer` / `Boros Charm`, `CardPredicate` `not_flying` / `min_toughness` for `Seismic Rupture` / `Valorous Stance`, top-or-bottom option on `OwnerPutsIntoLibrary` for `Uncharted Voyage`, `eligible_blockers` filtering `CantBlock`).
2. Author the unlocked Bucket B cards across `characteristics`, `choice_continuations`, `replacement_events`, and `combat_constraints` (`Vampire Interloper`, `Stromkirk Noble`, and `Juggernaut` if verified sound).
3. Verify key interactions: `Felidar Savior` (0/1/2 distinct other creatures, one removed before resolution), `Faebloom Trick` (2 Faeries enter before reflexive tap target is chosen and respondable), `Refute` and `Burglar Rat` (different choosers), `Opt` (scry before draw with no priority gap), `Stab` (lethal toughness loss vs. cleanup).

### Slices 5–N — Batched Bucket A Authoring
Deliver all remaining Bucket A cards in cohesive capability slices of ~20–35 cards each:
- **Batch 5a:** Vanilla & evergreen creatures + basic token producers (`Guarded Heir`, `Dragon Trainer`, `Rite of the Dragoncaller`, `Resolute Reinforcements`, `Elfsworn Giant`, `Infestation Sage`, `Maalfeld Twins`, `Release the Dogs`, `Rampaging Baloths`, `Eager Trufflesnout`, etc.).
- **Batch 5b:** Targeted removal, counters, bounces, pumps, modal spells, and sweepers (`Stab`, `Day of Judgment`, `Abrade`, `Cancel`, `Negate`, `Essence Scatter`, `Flashfreeze`, `Disenchant`, `Mortify`, `Angelic Edict`, `Make Your Move`, `Broken Wings`, `River's Rebuke`, `Aetherize`, `Overrun`, `Giant Growth`, `Snakeskin Veil`, `Dive Down`, `Make a Stand`, `Goblin Surprise`, `Slagstorm`, `Seeker's Folly`, `Deadly Plot`, etc.).
- **Batch 5c:** Triggered & activated permanents (`Clinquant Skymage`, `Firebrand Archer`, `Guttersnipe`, `Phyrexian Arena`, `Tatyova`, `Balmor`, `Good-Fortune Unicorn`, `Youthful Valkyrie`, `Shipwreck Dowser`, `Elvish Regrower`, `Angel of Finality`, `Meteor Golem`, `Soul-Guide Lantern`, `Arcanis`, `Rogue's Passage`, etc.).
- **Batch 5d:** Remaining `characteristics` Auras/Equipment/lords (`Empyrean Eagle`, `Lyra Dawnbringer`, `Death Baron`, `Aggressive Mammoth`, `Swiftfoot Boots`, `Basilisk Collar`, `Fireshrieker`, `Goldvein Pick`, `Pacifism`, `Starlight Snare`, `Twinblade Blessing`, `Unflinching Courage`, `Untamed Hunger`, `Involuntary Employment`, `Ghitu Lavarunner`, `Billowing Shriekmass`, `Skyknight Squire`, `Inspiring Paladin`, `Quick-Draw Katana`, `Adventuring Gear`, `Crawling Barrens`, etc.) and remaining `choice_continuations` (`Micromancer`, `Macabre Waltz`, `Pilfer`, `Duress`, `Rune-Scarred Demon`, `Affectionate Indrik`, `Bushwhack`, `Reclamation Sage`, `Wary Thespian`, `Solemn Simulacrum`, `Fierce Empath`, `Vile Entomber`, `Dragon Mage`, `Icewind Elemental`, `Rune-Sealed Wall`, `Strix Lookout`, `Battle-Rattle Shaman`, etc.).

## Verification across layers

At each slice, verify rules, learning tokens, observations, and conformance:

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards fdn::
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards observability::
cargo test --manifest-path managym/Cargo.toml --test conformance_tests every_registered_real_card_matches_scryfall
```

Also run the corresponding first-printing set filters for reprints. Whenever a new engine primitive or observation-relevant mechanic lands (Slices 2, 3, 4), rebuild the native extension and exercise consumer parity across native `Observation`, Python projection, v11 named tensor rows, LLM text, Academy frames, exact fork/replay, stale-Command rejection, and closed-disclosure opponent privacy using:
- `tests/semantic/test_text_observation.py`
- `tests/env/test_observation_characteristics.py`
- `tests/model/test_effect_vocabulary.py`
- `tests/model/test_pinned_pilot.py`

Preserve the four headline integrated-validation witnesses (board, chooser, action sequence, observed outcomes, test path):
1. **Arahbo + Helpful Hunter** (lord bonus, non-recursive Cat creation, both trigger orders).
2. **Doubling Season + Impact Tremors + Dragon Fodder** (4 distinct Goblins + 4 Tremors triggers vs. 2 without Season; plus loyalty-cost vs. effect-counter distinction).
3. **Lifelink combat + Ajani's Pridemate** (2 lifelinkers → 2 triggers; 1 trampling lifelinker → 1 trigger; prevented damage → 0).
4. **Lightshell Duo under closed disclosure** (Slice 1 choice and physical-order evidence is in place; post-resolution library knowledge passes both review positions above).

## Later phases

**Decomposition.** After Slices 2–N, audit new definitions for duplicated predicates, selectors, and token/trigger helpers (e.g., shared Cat/Angel/Zombie predicates, remaining duplicated lifegain or token-creation logic, shared retained-order handling between `Opt` and `Lightshell Duo`). Move real callers, delete any redundant owner, and keep directed counterexamples.

**Performance.** Run bounded `bench_states` stress positions matching the [stress-state procedure](../docs/development.md#stress-states): repeated Cat/Goblin entries under a lord or Impact Tremors; stacked static sources and attachments (`Arahbo`, `Regal Caracal`, `Anthem of Champions`, `Imprisoned in the Moon`, `Witness Protection`); `Felidar Savior` with mixed eligible/ineligible creatures; `Twinflame Tyrant` with many simultaneous damage recipients and competing replacements. Retain the same-runner Engine Budget (`bench_steps` and `bench_search`, exact digests, 10% limit). Before closing MTG-421, verify the exact partition of all 222 base rows into delivered, previously defined, and pending (each pending row naming its exact missing authority).
