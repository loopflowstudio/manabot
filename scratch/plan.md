# MTG-420 · Battlefield forms, copies and persistent ability sources

> “stop doing 1 card at a time”

Implement the remaining cards in batches that share rules behavior. Run the outer review cycle once per batch. Record which rows passed. When a row can't join its batch, record the specific dependency.

The Task brief's design constraint: “Explore one-card transform, two-card meld, and hidden face-down characteristics against contrasting real cards before fixing the representation, preserving exact physical-card identity and viewer-safe observations.”

## Done (detail in git and the test files)

- Transform: Cecil and Esper Origins, `fin.rs` (Cecil fixtures shipped before `9dbe17ee`).
- Meld: Fang/Vanille/Ragnarok with forked library order, Epic Downfall, lethal damage and Airbending Lesson, `fin.rs` at `e9b30cb6`.
- Face-down: Manifest Dread, Clone Legion copies, turn-up, sampling roots seeds 0–7, `dsk.rs` at `3672c184`. Academy recovery and manifest text at `fde87fb7` (`tests/academy/test_manifest_recovery.py`, `tests/semantic/test_manifest_text.py`).
- Departed Clone, Joo Dee and Homunculus Horde from the departed appearance, `lea.rs` at `bf9f4d73`.
- Copied triggers and scheduled sacrifice: Electroduplicate (flashback, under Doubling Season), Rite, Fire Crystal, Driver tokens, `fdn.rs` `copy_sacrifice` at `cc1061b7`. The stack keeps the triggering face (`238d8ecf`) and the captured catalog program (`320a8875`).
- `CopyException` restricted to copiable characteristics, `320a8875`.
- Named Koma's Coil copies through Rite, Self-Reflection (hand and flashback), Relm's Sketching and Extravagant Replication. Ordering Commands replay from the constructed snapshot. `fdn.rs` `named_coil` at `d52156c9`.
- Compress `9e297f0c`: waiting triggers read the captured face (`pending_trigger_definition`/`pending_trigger_face`), and target legality shares `fits_trigger_clause`. The focused nextest run passed 42 cases. The 13 Python copy-text, Academy trigger-order and pinned-game checks passed. The merge `039b275b` brought only Loopflow workflow files.

## Model to build (concept review)

Record of the post-Batch 1 review (`4127aa97` → `1eee5c08`): `scratch/concept-review.html`. Batch 1 settled departure LKI matching (`CharacteristicView` over `ObjectLki.battlefield`), split exile-link lifetimes (`UntilSourceLeaves` vs `WithSource` pruned in `release_resolved_forms`), explicit target optionality and counted returns. Model items across Batches 2–4:

1. **Choices and payment of the entering spell (Batch 2A).** See the corrected model and checks below. A copy of a permanent spell keeps kicker and X.

2. **Unified clause target legality and zone-filtered selection (Batch 2 foundation).**
   - Extend `requirement.another` (`triggers.rs:250`) to exclude both a live source (`Target::Permanent`) and a departed source's graveyard/exile arrival (`departure_arrivals(source)` → `Target::GraveyardCard`), preventing **Fiendish Panda** from targeting its own card in the graveyard.
   - Pass source identity into `resolution_targets` (`resolution.rs:516`) so placement (`fits_trigger_clause`) and resolution (`resolution_targets`) enforce one target-legality contract, including `another` and `within_source_bound` (CR 608.2b).
   - Add `ObjectSelector::ZoneMatching { zone: ZoneType, owner: Actor, predicate: CardPredicate }` (keeping `ObjectSelector::Zone(ZoneType)`'s serialized shape unchanged) so **Raise the Past** and **Rise of the Dark Realms** reuse `Effect::Move`'s simultaneous `EventBatch` pipeline without one-off mass-reanimation effects.

3. **General creation-time value freezing for delayed triggers (Batch 2 or 4 cleanup of Batch 1 shape finding).**
   - Replace the single-field `matches!` in `CreateDelayedTrigger` (`resolution.rs:1964`) with one recursive `freeze_delayed_effect` helper that resolves creation-time source/frame values (`DeathCountersMinus` / `SourceCountersMinus`, `X`, `TriggerDamage`, `ExcessDamage`, `ManaSpentToCast`, `ColorsSpentToCast`) into `EffectValue::Fixed` across every delayed effect. Generalize `DeathCountersMinus` to `SourceCountersMinus { kind, subtract }` (live or departed LKI) with a `token_renames.rs` entry — or keep Familiar's serialized token name if avoiding pilot churn.

4. **One layered continuous-effect path for transformation Auras and accurate characteristic observation (Batch 3 foundation).**
   - Add `Modification::SetName(String)` (`effect.rs`) so **Witness Protection**, **Eaten by Piranhas** and **Imprisoned in the Moon** all compose via `StaticAbility::Continuous(StaticContinuous)` instead of extending `SetCharacteristics`. Lower legacy `StaticAbility::AttachedSetsCharacteristics` through `Modification::SetName` in `layers.rs` (deleting `Operation::Name` while preserving Honest Work's TLA learning tokens).
   - Fix two shared consumers exposed by `StaticContinuous` Auras:
     - `zones.rs:934`: invalidate both players' mana caches whenever a permanent with a static `LoseAllAbilities`, `LoseNonManaAbilities` or `GrantActivated` enters or leaves (not only `attached_characteristics` / `attached_grants` / `SetsLandType`).
     - `observation.rs:1572–1581` & `record_vocabulary.rs:295`: scan `Modification::SetName` for granted names; decouple observed `name` (`characteristics.name != object.name`) and replaced `creature_types` (`characteristics.creature_types_set`, set by `SetCreatureTypes` and `SetTypes`) from `Operation::Name` so Eaten by Piranhas and Kellan project their replaced creature types.

5. **Explicit battlefield form backing and resumable battlefield moves (Batch 4).**
   - Replace the overloaded `Object.printed` / `Object.front` / `Permanent.manifested` / `ZoneManager.melded_cards` split with an explicit battlefield form recording physical backing card(s), orientation and face-down state, separate from immutable `CopiableValues`. Gate `transform_permanent` (`zones.rs:978`) on backing form (fixing the CR 712.9 counterexample where a non-DFC Clone copying Cecil currently transforms) and expose one `is_face_down()` authority (replacing the 3 repeated `permanent.manifested && definition_id == ANONYMOUS` checks in `identity.rs:52`, `action.rs:454`, `selection_sampling.rs:212`).
   - Keep `move_object` / `complete_immediate_batch` strictly for non-choice moves (draws, stack counter, setup) and migrate battlefield-capable resolution moves (starting with `cast_permission::exile_linked` for `Decision::ExileLinked` at `decision.rs:2905`, which currently panics at `proposed_event.rs:971` if Azula exiles Ragnarok) onto `EffectFrame.batch`.

*Choices made for review (reversible):* (1a) capture the entering spell at trigger time for the CR 603.4 recheck; (2a) `ObjectSelector::ZoneMatching` reusing `Effect::Move` over bespoke mass-reanimation effects; (3a) `Modification::SetName` + `StaticContinuous` lowering over extending `SetCharacteristics`; (4a) general `freeze_delayed_effect` helper and `SourceCountersMinus` over the single-field `matches!`.

## Batch 2 decomposition (graveyard choices, costs, simultaneous movement)

I mapped this against `e1806481`. The cohort's rules duplicate three facts, and each needs one owner.

**A. Choices and payment of the spell that entered.** CR 707.10 copies kicker and X on the stack. A copy of a permanent spell may enter kicked without having been cast. A blink or a battlefield copy has neither the original's choices nor payment. [Zendikar Rising release notes](https://magic.wizards.com/en/news/feature/zendikar-rising-release-notes-2020-09-10)

- Carry one `SpellEntry { kicked, x, payment: Option<PaymentReceipt> }` from the resolving frame through the proposed move into the permanent. A cast for zero has an empty receipt, a spell copy has no payment, and an ordinary zone entry has no spell record. The proposal and permanent hold an optional shared pointer. An inline record measured 184 bytes per permanent; the shared record keeps the row at its 168-byte ceiling.
- The same entry answers X counters and Familiar's cast condition before commitment. Its own ETB captures kicker, X and spent mana. Resolution rechecks the condition from the trigger's captured choices even if the source left.
- Preserve the historical serialized permanent fields for existing games. The old `was_cast` field was always serialized, including false. A copied spell counting as cast was a rules error.

**B. Whether a target fits its clause, given its source.** Four sites hold copies today:
- `fits_trigger_clause` (`triggers.rs:244`) checks `another` against the live source only, plus distinctness, mana budget and source bound.
- Trigger mode availability (`triggers.rs:501`) checks `another` against the live permanent only.
- `retarget.rs:101–116` checks distinctness, budget, `another` and source bound.
- `resolution_targets` (`resolution.rs:516`) checks spec legality only. It never applies `another` or `within_source_bound`.
- *Owner:* one `target_fits_clause(requirement, target, ClauseSource { object, reference }, controller)`. It combines `target_is_legal` with `another` and `within_source_bound`. `another` excludes the live source permanent and every `departure_arrivals(source)` as `Target::GraveyardCard`/`ExiledCard`, which is how Fiendish Panda can't return itself. All four sites call it. Choice-time-only rules (distinctness, `total_mana_value_x`) stay at the two choice sites (placement and retarget) as one `choice_admits` wrapper, so placement and retarget share one function too. `resolution_targets` takes the `ClauseSource`.

**C. Every matching card in a public zone.** `ObjectSelector::Each` is battlefield-only by contract, and `Zone(ZoneType)` is the resolving player's hidden-selection provenance.
- *Owner:* `ObjectSelector::ZoneMatching { zone, owner: Actor, predicate: CardPredicate }`, resolved in `Game::select` beside `Each`. Its result feeds the existing `Effect::Move` batch, giving one simultaneous `EventBatch` with ETB triggers and `under` for control.
- Raise the Past is `Move { what: ZoneMatching { Graveyard, You, creature mv≤2 }, to: Battlefield }`. Rise of the Dark Realms is the same with `AnyPlayer` and `under: You`.
- `Zone(_)` and `Each` keep their serialized shapes. The new variant reaches `ability_description.rs`, the learning tokens and the `observability.rs` guard in the same change.

**Not built in Batch 2: model item 3 (`freeze_delayed_effect`).** Nine-Lives Familiar is still its only consumer, and no Batch 2 or 3 card creates a delayed trigger that reads a creation-time value. The single `matches!` at `resolution.rs:1970` stays until a second real consumer appears. Generalizing now would mean inventing one.

### Delete — do not maintain (deepest owner first, consumers moved in the same cut)

1. Removed `Permanent.was_cast`, `cast_x` and `cast_spent` as state fields. Their serialized names remain in the digest projection.
2. Removed the separate stack lookup from `EntersWithCounters.if_cast`. Both this replacement and `TriggerCondition::WasCast` read the presence of payment on the spell entry.
3. Moved `Condition::Kicked` out of the resolution-only group so the entering source can answer it when the trigger fires.
4. The inline `another` checks at `triggers.rs:250`, `triggers.rs:501` and `retarget.rs:109`, and the separate `within_source_bound` calls at `triggers.rs:265` and `retarget.rs:114`. All move into `target_fits_clause`.
5. No old path is deleted for C. Nothing bespoke exists to replace; the new selector is the only implementation.

### Implement slices (one shared mechanism per loop pass)

1. **A, spell entries — built.** Sun-Blessed Healer (FDN) and Nullpriest of Oblivion (ZNR) use the same kicked intervening-if and graveyard move. The two set tests fork the kicker and target Commands, reject stale submissions, spend Cleric-only mana, remove the source with Lightning Bolt, and blink it in a sibling branch. The chooser sees the captured kicker during placement and both viewers see it after removal.
   - Eleven debug checks passed across the card, rules, source-tap and conformance binaries, including Familiar's cast-only replacement, Toph's spend receipt and the spell copy's X, kicker and absent payment. Both FDN ledger rows name their directed checks.
2. **B, clause legality — next.** Build `target_fits_clause` and `ClauseSource`, then move the four sites onto it. Cards: Fiendish Panda; Driver of the Dead (`avr.rs`, existing), whose remaining restrictions get exercised.
   - Panda's death trigger never offers its own graveyard card, and its power bound reads LKI after the +1/+1 counters.
   - A retargeted or resolution-time target outside Panda's bound is illegal.
   - The existing Jailer, Wolfbat and TLA within-power cases still pass.
3. **C, zone selection.** Build `ZoneMatching`. Cards: Raise the Past and Rise of the Dark Realms, on one two-graveyard setup.
   - Check simultaneous entry: two Driver of the Dead triggers see each other. Check ownership versus control under Rise, and ETB triggers.
4. **Remaining cohort on the same setup, composed from existing parts.**
   - Heartfire Immolator: prowess, self-sacrifice cost, `SourcePower` LKI. Pump it, sacrifice it, return it before resolution; the damage uses its departed power.
   - Ordeal of Nylea: `Attacks { EnchantedCreature }`, `Sacrificed { This }`, basic-land search.
   - Feldon's Cane (`atq.rs`): exiles itself as a cost, then shuffles.
   - Any of these that needs a new rules variant gets it as its own directed variant, never as card logic.

### Review criterion (`coverage-decomposition-review`)

Healer, Nullpriest, Panda, Driver, Raise the Past, Rise of the Dark Realms, Immolator, Ordeal and Cane each compose from the shared mechanisms, with no card-specific engine branch and no duplicate state:
- one `SpellEntry` owner for the choices and payment of an entering spell;
- one `target_fits_clause` behind placement, mode availability, retarget and resolution;
- one zone selector feeding `Effect::Move`.

The review should find:
- No `was_cast`/`cast_x`/`cast_spent` and no stack re-lookup for cast provenance: `rg 'was_cast\b|cast_x|cast_spent|find_spell_on_stack_index' managym/src/flow/{zones,proposed_event,triggers}.rs` is empty apart from unrelated spell lookups.
- No inline `requirement.another` outside `target_fits_clause`.
- Every card above defined in its first-printing set file with a directed test in `managym/cards/tests/cards/<set>.rs`, passing in debug.

This concerns growing the card file, not runtime speed.

## Remaining batches

1. **Exact departures and linked returns — built (`e6f6ef27`, compress `47dec0aa`/`2955a3d8`); behavior review passed (`scratch/behavior-review.md`).**
   - Built: `ObjectLki.battlefield` (`BattlefieldFacts`) plus `attached_to` and `meld_partner`, one `CharacteristicView` matcher (`layers::matches_view` / `matches_departure`) read by departure triggers, delayed `CreatureDies`, `SourceMatches`, `SourceCountersAtLeast` and additional trigger times; `ObjectSelector::Triggering` and `This` follow each departure arrival (both meld cards); `WithSource` links survive departure and are pruned in `release_resolved_forms`; `ObjectSelector::ExiledWithSource`; explicit `exile_until_source_leaves(spec, min)` with the inference deleted; `EntryCounters` (bare kind serialization for one counter) and `EffectValue::DeathCountersMinus`, frozen to a literal when the delayed trigger is created; `EntersWithCounters.if_cast`.
   - Cards: Banishing Light (`jou`), Stasis Snare (`bfz`), Prayer of Binding (`dmu`), Kykar, Nine-Lives Familiar, Infernal Vessel, Valkyrie's Call (`fdn`), Hoarding Dragon (`m11`), Angelic Destiny (`m12`), Fake Your Own Death (`snc`), Undying Malice (`vow`); `token.flying_spirit`; `Revival` counter; `Demon` subtype. Earth Kingdom Jailer now states its `min: 0` clause.
   - Proof: 14 `fdn::departure_*` cases (debug), with Commands forked and stale ones rejected at Prayer's target, Kykar's mode, the Dragon's search and the Familiar's return; 142 cases across `observability::`, `lea::`, `fin::`, `dsk::`, `fdn::`, Jailer and Wolfbat; `canonical_replay_tests` and the LKI/exile rules filters. Ledger rows updated.
   - Gate: Jailer's learning program gained an explicit `TargetClause`. If `tests/model/test_pinned_pilot.py` fails, declare it with `scripts/pinned_pilot.py record --reason` (the extension was not rebuilt here). Full debug nextest, clippy and Engine Budget are still to run.
   - Not covered: Malice or Call paired with a blink in response (the Call case covers simultaneous deaths). Add it in the Batch 1 review if a fixture is cheap; otherwise carry it to Batch 2's integrated cases.
2. **Graveyard choices, costs and simultaneous movement.**
   - *Foundation:* A is built. B (clause legality) and C (zone selection) remain. Item 3 waits for a second consumer.
   - *Cards & fixtures:* Use one two-graveyard setup for Raise the Past, Rise of the Dark Realms, Driver of the Dead (remaining target restrictions), Fiendish Panda (including self-exclusion in graveyard), Heartfire Immolator, Sun-Blessed Healer (including removal in response to kicked ETB), Nullpriest of Oblivion, Feldon's Cane (`atq.rs`) and Ordeal of Nylea. Abyssal Harvester is blocked (see below).
3. **Effective characteristics and grants.**
   - *Foundation:* Land item 4 (`Modification::SetName`, `StaticContinuous` lowering for transformation Auras, mana-cache invalidation for static ability removal/grant in `zones.rs:934`, `record_vocabulary.rs` scan, and decoupled observed `name` / `creature_types_set` in `observation.rs:1572`).
   - *Cards & fixtures:* Use one board before and after grant, copy, attachment movement and reentry for Celestial Armor, Dropkick Bomber, Kellan (`fdn.rs`), Mild-Mannered Librarian, Soulstone Sanctuary, Crawling Barrens, Witness Protection, Eaten by Piranhas and Imprisoned in the Moon (`emn.rs`). Include death-time LKI checks for Witness Protection (Citizen) and Imprisoned in the Moon (noncreature Land), mana-cache checks on opponent's enchanted mana dork, and observation checks on Kellan and Eaten by Piranhas's replaced creature types.
4. **Representation unification, consumers and budget.** Land item 5 (battlefield form backing + CR 712.9 transform gate + `is_face_down()` authority + battlefield `move_object` audit, plus item 3 if deferred), close the open replay/sampling checks, verify consumers and measure against the engine budget.

Row identity comes from the [FDN ledger](../.lf/wave/coverage/research/standard-2026-11/sets/fdn.jsonl) (Task 05 of the [milestone](../.lf/wave/coverage/research/standard-2026-11/milestone.md)). Update each row with its delivered source and checks. Reuse first printings. Match produced tokens by their full printed characteristics: Kykar needs a white flying Spirit, and the catalog already defines Treasure and Koma's Coil.

| Rows | Behavior to exercise |
| --- | --- |
| Banishing Light, Stasis Snare, Prayer of Binding | Remove the source before its entry trigger resolves, then again after the exile resolves. Only the second has an exile to end. Move the exiled card out and back before the source leaves. The new appearance stays put. Prayer gains life with zero targets chosen (`min: 0`); Banishing Light and Stasis Snare require a target (`min: 1`). A sole target that becomes illegal loses the entire resolution. |
| Kykar; Hoarding Dragon | Choose Kykar's blink or its Spirit. The delayed return binds to the object in exile. With two Dragons that found different artifacts, only the dying Dragon recovers its linked card via `WithSource` surviving departure. Bounce and replay the same Dragon, then check the old link. |
| Nine-Lives Familiar, Infernal Vessel, Valkyrie's Call | Capture the Familiar's revival count at death (`k - 1` frozen into the delayed trigger) and spend one on its next-end-step return; cast entry gives 8 revival counters while blink gives 0. In a sibling fork, move it out of the graveyard. Vessel and Call read death-time effective creature types through unified LKI so a returned Demon Vessel or Angel creature does not re-trigger on a second death. Counters and type grants attach to the returned incarnation. Kill Call together with an eligible creature; the trigger still returns the owner's card. |
| Fake Your Own Death, Undying Malice, Celestial Armor | Track which creature received a grant and when it ends. Move the Equipment after its entry trigger resolves. The protection follows its original recipient. When Fake Your Own Death is cast on a stolen creature, its owner gets the tapped card and the trigger's controller gets the Treasure. A dying token still produces Treasure. Malice returns the card tapped with its counter. |
| Angelic Destiny, Ordeal of Nylea | Follow the enchanted creature through death (including simultaneous Aura + creature departure reading `ObjectLki.attached_to`) and the Aura through its own departure. Only the eligible Aura appearance is recovered. Resolve Ordeal's attack, counter threshold, sacrifice and optional library selection through its actual triggers. |
| Raise the Past, Rise of the Dark Realms, Driver, Fiendish Panda, Heartfire Immolator | Put qualifying and excluded cards in both graveyards. Check simultaneous entry, ownership, resulting control and ETB triggers. Driver and Panda choose legal graveyard targets when their triggers are placed, including Panda's last power and its exclusions. Pump Immolator, sacrifice it as the cost and return it before resolution. Its damage uses its departed power. |
| Sun-Blessed Healer, Nullpriest of Oblivion, Feldon's Cane | Read the actual kicker choice in the ETB intervening-if (CR 603.4). An ordinary blink supplies `None` and does not trigger. Each has its own graveyard target restrictions. Check competing restricted spends on the kicked cards. Cane exiles itself as the cost before shuffling. |
| Characteristics subset | Kellan and Librarian: incarnation-bound upgrades. Sanctuary and Barrens: persistent versus temporary animation. Dropkick Bomber: flying and a damage trigger granted to one exact recipient. Witness Protection, Eaten by Piranhas and Moon: layered form changes across a copy and through death LKI. |

Pair Malice or Valkyrie's Call with simultaneous deaths and a blink in response. Keep simultaneous reanimation and sacrificed-source power in the integrated cases.

**Blocked: Abyssal Harvester.** It needs the history/queries Task's exact graveyard arrivals this turn from any zone. `TurnState::graveyard_from_battlefield` records only battlefield departures. The FDN ledger names no release that provides this. When the query lands, locate its consumer API with `rg -n 'graveyard_from_battlefield|GraveyardThisTurn|graveyard.*this turn' managym/src/flow managym/state/src`. Then offer Harvester a milled creature, a dead creature and a card that left and reentered. Follow the selected exile appearance through the Nightmare copy and the exile of the other Nightmare tokens.

## Open checks on finished work

- Lost Days: the order choice should be visible only to its owner, and the game should replay from its record.
- Horde, Driver and Fire Crystal still need viewer projections and replay of their recorded Commands from a recorded root. Their native fork and restore checks already pass.
- The named Coil setup has no seed-to-position replay file.
- Manifest sampling: there is no pending private selection in the fixture. Constrained sampling, paired seeded searches and comparative viewer trace exports remain open.
- Audit immediate battlefield move callers by their legal source zones. Start with `cast_permission::exile_linked` (`Decision::ExileLinked`) and linked entry follow-ups that consume a move's arrivals.

## Representation review

Review the final Rust representation against these witnesses:

- **Cecil** keeps its card, object, counters, attachments and targets across transformation; a non-DFC **Clone copying Cecil** cannot transform (CR 712.9).
- **Esper Origins** gets a new exact identity at each zone move.
- **Ragnarok** is one permanent backed by two physical cards, and each card moves exactly once.
- **A token copy of Ragnarok** moves no physical card.
- **A copy of a manifested permanent** gets the public 2/2 without the hidden printing or turn-up.
- **A returned Angel / Demon / manifested land / mooned creature dying** matches its last-known effective battlefield characteristics, never its printed graveyard card.

Native owners: `managym/state/src/{game_object,permanent,ability,effect,zone}.rs`, `managym/src/flow/{identity,zones,layers,statics,trigger,triggers,proposed_event,resolution,cast_permission}.rs`.

- Keep `CardId`, `ObjectId`, `PermanentId` and viewer IDs distinct.
- Capture the departing form and effective `ObjectLki` facts before `Object::mint_appearance`. Bind every link to the exact object its move produced.
- Keep `ExileLink` (with separate `UntilSourceLeaves` vs `WithSource` lifetimes), `DelayedTrigger` and `AbilitySource` as separate types for their separate lifetimes.
- Preserve the two-arrival batch through replacement and rollback.
- `CopiableValues` feeds departed-source reads and token creation. The content pack decides when a departure needs a snapshot.
- Keep counters, animation, equipment and transient grants out of the copied payload.
- A stacked grant still resolves after its grantor leaves or the source changes face. A new appearance starts with only its own grants.
- Invalidate the layer, mana and trigger indexes on form, grant, control and zone changes.
- Planeswalkers, emblems and castable spell-copy links stay with their milestone owners.

After the cohort, review Clone, Electroduplicate, Hoarding Dragon, Valkyrie's Call, Manifest Dread and Ragnarok together for shared movement, snapshot, ability, LKI and observation owners. Remove superseded fields and callers, and keep the directed regressions.

## Consumer checks

- One native viewer projection (`flow/knowledge.rs`, `agent/{observation,observation_json,observation_encoder,shared_effects,record_vocabulary}.rs`, `src/cards/tokens.rs`, bindings, `decision_context.rs`, `manabot/semantic/text_observation.py`) carries form, copy exceptions, grants, expiry and permitted knowledge. Legal actions, focus, history and relations use the same permanent reference. Hidden keys stay out of opponent joins, copy provenance, offer labels and tokens. Add the matching `observability.rs` entries. Preserve the v9/v10 layout and frozen evidence.
- Face-down comparison: build two roots that differ only in which hidden card a slot holds. At the opponent's decision, compare native and Python observations, v11 buffers, LLM text, Academy reconnect, viewer history and replay export. Check controller inspection separately, along with reveal on departure, `possible_worlds.rs` and search, and closed disclosure.
- Submit actual Commands at the private selection, kicked payment, meld order and trigger order. Fork while each is pending, reject the stale Command, and check that each consequence commits once. Replay each branch from its recorded root.
- For each group, retain the setup, seed, chooser, offered Commands, the branch taken, the outcome, viewer rows, text and tokens, the replay witness, the test path and the commit.

Focused checks from `managym/`: `cargo test -p managym-cards --test cards fdn::` and the matching set filters, `cargo test --test rules_tests cr_400`, `cargo test --test canonical_replay_tests`, the possible-world and search tests, and the learning-token test in `observability.rs`. After rebuilding the extension, run `uv run --extra dev pytest tests/llm/test_llm_native_game.py tests/env/test_observation.py tests/model/test_effect_vocabulary.py tests/model/test_pinned_pilot.py tests/academy/test_choice_navigation.py -q`. The full debug suite, format, clippy and package gates run at gate.

## Measurement

Add bounded states to `managym/examples/bench_states.rs`, varying the permanent count independently of source variety:

- Many Coils with Electroduplicate and kicked Rite, with variants that add distinct exceptions and Kellan grants.
- Many Banishing Lights with one source leaving. Mixed Familiar, Vessel and Call deaths. Healer kicked and unkicked.
- Several manifests over distinct printings, plus one Fang/Vanille pair and several namesakes.

Gate whole-game step throughput and Search-64 at the same-runner 10% limit. Attribute any digest change to a rules change ([engine budget](../docs/development.md#engine-performance-budget)). The 4096-byte inline `GameState` bound holds because trigger placement is boxed (`fde87fb7`). Its fork and throughput cost still needs measuring.
