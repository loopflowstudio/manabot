# MTG-421 · Ready-now Foundations

## Working progress

**Working phase requires another implementation pass.** Slice 1 defines Lightshell Duo (`b9208b43d8`, `b522de30c4`). Review against the runtime catalog at `b522de30c4` found four of the 222 selected names from base `0b1cc9cfef9c1a98430c825b9764a49237182696`. Lightshell Duo is new. Kellan, Planar Trailblazer (`fdn.rs`), Imprisoned in the Moon (`emn.rs`) and Doubling Season (`rav.rs`) predate the batch and now have their source paths recorded in the ledger. Of the other 218, eleven have confirmed deferrals and 207 await the readiness audit. The Arahbo/Hunter, Doubling Season/Fodder/Tremors and lifelink/Pridemate interactions remain unimplemented as sets of cards.

### Behavior after the shared repair at `12096ae8`

**Surveil knowledge after resolution.** Milling either inspected card shifts the other to known position 0 for the chooser. Keeping both and reversing their order changes only positions 0 and 1. Hill Giant and Forest, previously placed by Accumulate Wisdom, remain known at positions 3 and 4. Four Lightshell cases passed on the release extension rebuilt from merged snapshot `8aef2be7`, covering native observations, v11, text, closed-opponent privacy and replay on independent forks.

**Focused evidence from constructed positions**

- The FDN definition uses the ordinary enters trigger with `Surveil { count: 2 }`. The chooser can mill either card followed by the other, mill just one, or keep both in either order. The debug FDN fixture draws the chosen first card with Scout's Warning, after prowess resolves to make Duo 4/5. Duo returns to 3/4 by the next turn. These outcomes agree with the published mechanics. [Surveil](https://magic.wizards.com/en/news/feature/tarkir-dragonstorm-mechanics) · [Prowess](https://magic.wizards.com/en/news/feature/magic-origins-mechanics)
- The chooser gets three initial offers and two ordering offers, with matching v11 valid-action counts and distinct object focus. Each tested Command gave the same transition and digest on independent forks. Reusing the capture was rejected with the root and committed branch intact. Swapping the private top cards, or selecting opposite cards before committing, produced identical opponent semantic JSON, Academy projections and v11 buffers.
- A separate `GameSession.current_recovery("reconnect")` probe at the constructed Lightshell root recovered both named mill offers and **Keep on top**. Submitting **Keep on top** recovered the two bound ordering offers. Choosing Hill Giant first put it on top. The Academy session returned the duplicate receipt and rejected a new Command using the old revision at each transition. The committed Python fixture directly exercises `decision_surface_view`; this session probe exercised the envelope.
- Debug tests passed for the shared surveil completion event, ordered continuation and private fork behavior. The prowess fixtures passed for Swiftspear, the Monk token and Sokka's grant, including the own-noncreature control in the Iguana/Boar fixture. The constructor and predicate produce the existing rules structures. The catalog returned two triggered program spans for Duo, and its prowess span matched Swiftspear token for token. Nine learning-token observability checks and the registered-card Scryfall conformance check passed. The two text fixtures that now share the replay helper also passed. This slice changes no payment or receipt semantics.

Snapshot library-order pins retain inspected object identities across exact forks and replay. Viewer observations expose only the positions known to that viewer.

The Run Away Together audit is complete. Its ledger row records the missing controller comparison at casting and resolution. The next audit starts with Drakuseth, Maw of Flames: trace whether its optional secondary targets can exclude the primary target and retain their separate damage amounts through resolution. Use that finding to classify related rows together. The next playable witness is Arahbo plus Helpful Hunter, including both trigger orders, the draw, and the resulting Cat's characteristics. The full debug suite, pinned-pilot compatibility and browser checks remain gate evidence.

## Experience and model to build

Authoring Foundations cards should be direct, predictable, and collision-free: one canonical way to write each static effect, token face, lifegain trigger, and replacement effect, backed by an honest readiness partition of all 222 rows so work never stalls mid-slice on hidden engine gaps or fragments into card-shaped enum variants.

### Canonical authoring model (and choices made for review)

1. **Capability-by-capability readiness partition of the 222 rows.** `fdn.jsonl`'s `dependencies == []` tag came from a text-only planning scan (*"No games, tests or benchmarks were run"*). The three existing definitions are reconciled. Classify the remaining 207 unaudited rows against native selection, execution and observation. The task brief allows “minimal local primitives”. Treat the designs below as proposals until the audit establishes their callers and scope:
   - **Bucket A — Composable today:** Zero engine edits needed; batch-author in capability-grouped slices of ~15–35 cards instead of 1–2 cards per slice.
   - **Bucket B — Unlocked by minimal local primitives:** Land each shared primitive once with directed counterexamples, observability guards, and learning tokens, then batch-author every FDN card it unlocks.
   - **Bucket C — Explicitly deferred:** Record as `definition pending` in `fdn.jsonl` with the exact missing authority and owning parallel track. Do not add one-off shortcuts.
   - *Choice made for review:* Include the 4 small local primitives (Bucket B) in MTG-421 rather than restricting MTG-421 to zero-engine-edit cards (which would exclude headline witnesses like `Ajani's Pridemate`, `Twinflame Tyrant`, `Authority of the Consuls`, and `Witness Protection`).

2. **One continuous static authoring form (`StaticAbility::Continuous`).** `managym/src/flow/layers.rs` already lowers every legacy `StaticAbility` wrapper (`PtBuff`, `KeywordGrant`, `TypeGrant`, `AttachedSetsCharacteristics`, `AttachedHas`, `TriggeredGrant`, `SetsLandType`) into one `Node`/`Modification` layer pass, while `AttachedSetsCharacteristics` cannot express color or card-type changes (`Witness Protection`, `Eaten by Piranhas`).
   - Author **every new FDN static characteristic effect** with `StaticAbility::Continuous(StaticContinuous { .. })`. Add `Modification::SetName(String)` for name-overwriting Auras (`Witness Protection`). If any FDN Equipment grants an activated ability referencing `ObjectSelector::Grantor`, populate `granted_by` for `StaticScope::Attached` in `layers.rs` (otherwise `Modification::GrantActivated` already covers ordinary grants like `Imprisoned in the Moon`).
   - *Choice made for review:* Keep existing TLA/demo cards on their legacy `StaticAbility` variants so `tests/model/test_pinned_pilot.py` and stored checkpoints remain untouched; apply `StaticAbility::Continuous` uniformly to all new FDN cards.

3. **Characteristic-qualified keys for new creature tokens.** Existing bare keys in `managym/cards/src/tokens.rs` were already claimed by non-vanilla TLA/demo faces (`token.dragon` has firebending 4; `token.soldier` is a 2/2 red firebender; `token.spirit` has a block restriction; `token.bear` is 4/4; `token.elf_warrior` is G/W), and Scryfall conformance skips tokens.
   - Reuse predefined game tokens (`token.food`, `token.treasure`, `token.clue`, `token.powerstone`) and preserve every existing key and payload untouched.
   - Give every **new** non-predefined creature token in `managym/cards/src/tokens.rs` a self-describing, characteristic-qualified key: `token.<name>_<p>_<t>_<colors>[_<ability>]` (e.g., `token.cat_1_1_w`, `token.cat_1_1_w_lifelink`, `token.knight_1_1_w`, `token.soldier_1_1_w`, `token.dragon_4_4_r_flying`, `token.elf_warrior_1_1_g`, `token.goblin_1_1_r`, `token.zombie_2_2_b`, `token.dog_1_1_w`, `token.beast_4_4_g`, `token.insect_1_1_bg_flying`, `token.faerie_1_1_u_flying`).
   - *Choice made for review:* Characteristic-qualified keys over ad-hoc adjectives (`plain_dragon`, `lifelink_cat`), so future Standard sets never collide.

4. **Per-source lifelink batching and `TriggerCondition::GainsLife`.** `damage.rs::commit_damage` currently calls `commit_life_change` per damage target, bypassing `ProposedEvent::LifeChange` and splitting a single lifelink+trample creature (blocker + player) into two life-gain events (violating CR 702.15b / 119.9), and `TriggerCondition` has no lifegain trigger.
   - Aggregate positive lifelink damage **per damage source** within each damage batch and route each source's gain through `ProposedEvent::LifeChange` (so one trampling lifelinker = 1 life-gain event, two lifelinkers = 2 life-gain events, double strike = 1 event per combat damage pass, and prevented damage / `PlayersCantGainLife` = 0 events).
   - Add `TriggerCondition::GainsLife { who: Actor }` wired to positive `GameEvent::LifeChanged` (and indexed in `TriggerIndex`, `observability.rs`, and LLM text).

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

Candidates still to verify:
- **Cross-target relational constraints:** `Drakuseth, Maw of Flames` (primary + up to 2 distinct other targets).
- **Reveal-until-matching with random bottom:** `Spinner of Souls`.
- Any additional row identified in Step 0 whose printed text requires a substantial new authority from a parallel track (e.g., verify `Circuitous Route` basic-or-Gate union search, `Campus Guide` search-shuffle-then-top, `Goblin Oriflamme` / `Kitesail Corsair` while-attacking static, `Pirate's Cutlass` entry attach).

## Execution sequence

Every slice follows the same authoring rules: file each card once under its **first printing's set module** (`managym/cards/src/<set>.rs`) with its Scryfall printing ID, register via `cards()` and `cards/src/lib.rs`, snapshot shell + Oracle text in `managym/tests/fixtures/scryfall_cards.json`, update `fdn.jsonl` (`status: "source definition found"`, exact source path), and add directed set tests covering normal paths and counterexamples.

### Slice 2 — Triage + Token/Static/Lifegain Model Proof (run next)
Begin with the Arahbo and Helpful Hunter position. Audit those two definitions, then exercise both trigger orders before extending the batch. Complete the cohort partition across the remaining slices.
1. **Step 0 audit & ledger sync:** Three existing definitions are reconciled and eleven deferrals documented. Audit each group against native rules before authoring it. Start the targeting group with Drakuseth, Maw of Flames. The token group can proceed independently. Record the A / B / C partition in `scratch/questions.md` and the exact missing authority for each deferral in `fdn.jsonl`.
2. **Tokens + `StaticAbility::Continuous` proof:** Add `token.cat_1_1_w` and `token.cat_1_1_w_lifelink` in `tokens.rs`. Author **Arahbo, the First Fang**, **Helpful Hunter**, **Prideful Parent**, **Regal Caracal**, and **Anthem of Champions** using `StaticAbility::Continuous`. Verify:
   - Arahbo + Helpful Hunter: Hunter draws 1, Arahbo creates 1 ordinary Cat (`token.cat_1_1_w`) that gets Arahbo's lord bonus and does *not* recursively trigger Arahbo; test both trigger orders.
   - Regal Caracal: creates 2 `token.cat_1_1_w_lifelink` tokens; remove Caracal and confirm those tokens retain lifelink while Arahbo's Cat does not.
   - Anthem of Champions: Cat entering after Anthem gets +1/+1.
3. **Per-source lifelink + `TriggerCondition::GainsLife` proof:** Implement per-source lifelink batching through `ProposedEvent::LifeChange` and `TriggerCondition::GainsLife { who }`, wired through `observability.rs`, learning tokens, and LLM text. Author **Ajani's Pridemate**, **Marauding Blight-Priest**, and **Drogskol Reaver**. Verify:
   - Two lifelink creatures dealing combat damage → 2 Pridemate triggers.
   - One lifelink+trample creature dealing damage to a blocker and player → 1 Pridemate trigger (CR 702.15b / 119.9).
   - Double-strike lifelinker (`Drogskol Reaver`) → 1 trigger in first-strike damage + 1 in normal combat damage.
   - Prevented damage or `PlayersCantGainLife` → 0 triggers.

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

**Decomposition.** After Slices 2–N, audit new definitions for duplicated predicates, selectors, and token/trigger helpers (e.g., shared Cat/Angel/Zombie predicates, shared prowess/lifegain/token-creation helpers, shared retained-order handling between `Opt` and `Lightshell Duo`). Move real callers, delete any redundant owner, and keep directed counterexamples.

**Performance.** Run bounded `bench_states` stress positions matching the [stress-state procedure](../docs/development.md#stress-states): repeated Cat/Goblin entries under a lord or Impact Tremors; stacked static sources and attachments (`Arahbo`, `Regal Caracal`, `Anthem of Champions`, `Imprisoned in the Moon`, `Witness Protection`); `Felidar Savior` with mixed eligible/ineligible creatures; `Twinflame Tyrant` with many simultaneous damage recipients and competing replacements. Retain the same-runner Engine Budget (`bench_steps` and `bench_search`, exact digests, 10% limit). Before closing MTG-421, verify the exact partition of all 222 base rows into delivered, previously defined, and pending (each pending row naming its exact missing authority).
