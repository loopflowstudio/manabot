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

Record of the full review: `scratch/concept-review.html`. Before authoring Batch 1's cards, fix the four structural mismatches in the shared lifetime model so each batch exercises the real representation instead of card-specific workarounds:

1. **One effective-characteristic matcher for live and departed permanents (lead slice, before Batch 1 cards).**
   - *Problem:* `subject_matches_moved_object` (`triggers.rs`) and `delayed_triggers_for_event` (`CreatureDies`) currently test `&self.state[moved_object]` (the reset printed graveyard card) with only `animated_creature: bool` patched in, ignoring layered subtypes, card types, power, keywords, colors and copy/manifest faces. `Condition::SourceMatches` only checks live permanents; `TriggerSubject::EnchantedCreature` checks only live Auras (`ObjectLki` lacks `attached_to`); `ObjectSelector::Triggering` only returns live permanents (cannot follow a dead creature to its graveyard card).
   - *Model:* Complete `ObjectLki` (`managym/state/src/game_object.rs`) with compact effective battlefield facts — `types: CardTypes`, `subtypes: Vec<String>`, `colors: Colors`, `is_legendary: bool`, `name: String` (when set/distinct), `mana_value: u8`, `attached_to: Option<ObjectId>` — alongside its existing `power`, `toughness`, `keywords`, `counters`, `abilities_lost`, `granted_abilities`, `copiable`. Introduce one borrowed `CharacteristicView<'a>` matcher shared by `layers::matches_view` (live permanents) and departure readers (`subject_matches_moved_object`, `DelayedTriggerEvent::CreatureDies`, `Condition::SourceMatches`, `TriggerSubject::EnchantedCreature`, `additional_trigger_times`). Keep `ObjectLki` in `managym-state` free of `flow` dependencies and avoid cloning derived ability vectors.
   - *Successor following:* Extend `ObjectSelector::Triggering` (`resolution.rs`) to follow `departed_permanent_lki(id)` → `zones.successor(departed)` in graveyard/exile (matching `ObjectSelector::This`), while battlefield-only verbs continue narrowing through `selected_permanents`. Replace the bespoke successor lookup in `MayExileTriggeringWithSource` with the shared helper.
   - *First proof:* A creature returned by Valkyrie's Call (Angel) or Infernal Vessel (Demon) dies a second time and does **not** re-trigger; a manifested Forest dies as a creature; an Imprisoned-in-the-Moon creature dies as a noncreature Land; simultaneous Aura + enchanted-creature death still fires Angelic Destiny.

2. **Separate exile-link lifetimes and explicit target optionality (lead slice, before Batch 1 cards).**
   - *Problem:* `handle_leaves_battlefield` (`zones.rs`) deletes *all* `exile_links` matching `departing_object`, destroying `LinkedExile::WithSource` links (`returns: false`) before Hoarding Dragon's dies trigger even stacks. `ObjectSelector` cannot select cards exiled with the source, and `effect_target_requirements` (`ability.rs`) silently forces every `LinkedExile::UntilSourceLeaves` effect to `min: 0` (making mandatory Banishing Light / Stasis Snare optional unless wrapped).
   - *Model:* Only `UntilSourceLeaves` links end and queue immediate return followups in `handle_leaves_battlefield`. `WithSource` links survive source departure while the exiled appearance remains in exile and any live permanent, pending trigger, stack ability or suspended frame still references `link.source`, pruned alongside `ObjectLkiLog::retain_forms`. Add `ObjectSelector::ExiledWithSource` so Hoarding Dragon and future CR 607.2a abilities compose with `Effect::Move` / `SelectCards`. Make target optionality explicit via `TargetClause` (`min: 0` for Prayer of Binding, `min: 1` for Banishing Light and Stasis Snare) rather than inferring `min: 0` from `UntilSourceLeaves`. Keep `Effect::ExileLinked`'s existing serialized name for learning-token stability (its runtime `ExileCastGrant` ownership stays with cast permissions).

3. **Multi-counter zone entry and frozen delayed-trigger values (Batch 1).**
   - Generalize `Effect::Move` entry counters from `with_counter: Option<CounterKind>` (hardcoded `1`) to support a count (e.g. `with_counters: Option<(CounterKind, EffectValue)>` or a count paired with `CounterKind`, preserving existing serialization for count 1 / absent).
   - Allow `Effect::CreateDelayedTrigger` to freeze creation-time numeric quantities (e.g. resolving `Effect::Move` counter counts or storing a captured `i32` on `DelayedTrigger`) so Nine-Lives Familiar's end-step return uses `death_revival_counters - 1` captured at death rather than reading `0` from the graveyard. Also support counter-kind checks on `ObjectLki` (`SourceCountersAtLeast` reading LKI after departure) for Familiar's intervening-if.

4. **Unified cast provenance on `Permanent` and ETB triggers (Batch 2).**
   - Replace the partial `was_cast` / `cast_x` / `cast_spent` fields on `Permanent` (`permanent.rs`) with `cast: Option<CastReceipt>` carrying `{ kicked, x, spent }` (`None` for tokens, blinks and reanimations), and thread `kicked` onto `PendingTrigger` and `TriggeredAbilityOnStack`.
   - Allow `Condition::Kicked` (and `WasCast`) to evaluate in an entering permanent's intervening-if (CR 603.4) and cast-entry replacements (Nine-Lives Familiar, Sun-Blessed Healer, Nullpriest of Oblivion).

5. **Explicit battlefield form backing and resumable battlefield moves (Batch 4).**
   - Replace the overloaded `Object.printed` / `Object.front` / `Permanent.manifested` / `ZoneManager.melded_cards` split with an explicit battlefield form recording physical backing card(s), orientation and face-down state, separate from immutable `CopiableValues`. Gate `transform_permanent` (`zones.rs`) on backing form (fixing the CR 712.9 counterexample where a non-DFC Clone copying Cecil currently transforms) and expose one `is_face_down()` authority so callers never repeat `permanent.manifested && definition_id == ANONYMOUS`.
   - Keep `move_object` / `complete_immediate_batch` strictly for non-choice moves (draws, stack counter, setup) and migrate battlefield-capable resolution moves (starting with `cast_permission::exile_linked` for `Decision::ExileLinked`, which currently panics if Azula exiles Ragnarok) onto `EffectFrame.batch`.

*Choices made for review (reversible):* (1a) compact effective fields on `ObjectLki` + shared borrowed `CharacteristicView` over storing `Arc<Characteristics>` across crate boundaries; (2a) retain `WithSource` links across departure and prune on liveness rather than copying link IDs onto stack objects; (3a) `Option<CastReceipt>` on `Permanent` over a fourth loose `cast_kicked` bool; (4a) retain `Effect::ExileLinked`'s serialized variant name to avoid learning-token churn.

## Remaining batches

1. **Exact departures and linked returns — built, awaiting review.**
   - Items 1–3 landed: `ObjectLki.battlefield` (`BattlefieldFacts`) plus `attached_to` and `meld_partner`, one `CharacteristicView` matcher (`layers::matches_view` / `matches_departure`) read by departure triggers, delayed `CreatureDies`, `SourceMatches`, `SourceCountersAtLeast` and additional trigger times; `ObjectSelector::Triggering` and `This` follow each departure arrival (both meld cards); `WithSource` links survive departure and are pruned in `release_resolved_forms`; `ObjectSelector::ExiledWithSource`; explicit `exile_until_source_leaves(spec, min)` with the inference deleted; `EntryCounters` (bare kind serialization for one counter) and `EffectValue::DeathCountersMinus`, frozen to a literal when the delayed trigger is created; `EntersWithCounters.if_cast`.
   - Cards: Banishing Light (`jou`), Stasis Snare (`bfz`), Prayer of Binding (`dmu`), Kykar, Nine-Lives Familiar, Infernal Vessel, Valkyrie's Call (`fdn`), Hoarding Dragon (`m11`), Angelic Destiny (`m12`), Fake Your Own Death (`snc`), Undying Malice (`vow`); `token.flying_spirit`; `Revival` counter; `Demon` subtype. Earth Kingdom Jailer now states its `min: 0` clause.
   - Proof: 14 `fdn::departure_*` cases (debug), with Commands forked and stale ones rejected at Prayer's target, Kykar's mode, the Dragon's search and the Familiar's return; 142 cases across `observability::`, `lea::`, `fin::`, `dsk::`, `fdn::`, Jailer and Wolfbat; `canonical_replay_tests` and the LKI/exile rules filters. Ledger rows updated.
   - Gate: Jailer's learning program gained an explicit `TargetClause`. If `tests/model/test_pinned_pilot.py` fails, declare it with `scripts/pinned_pilot.py record --reason` (the extension was not rebuilt here). Full debug nextest, clippy and Engine Budget are still to run.
   - Not covered: Malice or Call paired with a blink in response (the Call case covers simultaneous deaths).
2. **Graveyard choices, costs and simultaneous movement.**
   - *Foundation:* Land item 4 (`Option<CastReceipt>` carrying `kicked`, `x`, `spent` into `Permanent` and ETB triggers/intervening-if).
   - *Cards & fixtures:* Use one two-graveyard setup for Raise the Past, Rise of the Dark Realms, Driver of the Dead (remaining target restrictions), Fiendish Panda, Heartfire Immolator, Sun-Blessed Healer, Nullpriest of Oblivion, Feldon's Cane (`atq.rs`) and Ordeal of Nylea. Abyssal Harvester is blocked (see below).
3. **Effective characteristics and grants.** Use one board before and after grant, copy, attachment movement and reentry for Celestial Armor, Dropkick Bomber, Kellan (`fdn.rs`), Mild-Mannered Librarian, Soulstone Sanctuary, Crawling Barrens, Witness Protection, Eaten by Piranhas and Imprisoned in the Moon (`emn.rs`). Include death-time LKI checks for Witness Protection (Citizen) and Imprisoned in the Moon (noncreature Land).
4. **Representation unification, consumers and budget.** Land item 5 (battlefield form backing + CR 712.9 transform gate + `is_face_down()` authority + battlefield `move_object` audit), close the open replay/sampling checks, verify consumers and measure against the engine budget.

Row identity comes from the [FDN ledger](../.lf/wave/coverage/research/standard-2026-11/sets/fdn.jsonl) (Task 05 of the [milestone](../.lf/wave/coverage/research/standard-2026-11/milestone.md)). Update each row with its delivered source and checks. Reuse first printings. Match produced tokens by their full printed characteristics: Kykar needs a white flying Spirit, and the catalog already defines Treasure and Koma's Coil.

| Rows | Behavior to exercise |
| --- | --- |
| Banishing Light, Stasis Snare, Prayer of Binding | Remove the source before its entry trigger resolves, then again after the exile resolves. Only the second has an exile to end. Move the exiled card out and back before the source leaves. The new appearance stays put. Prayer gains life with zero targets chosen (`min: 0`); Banishing Light and Stasis Snare require a target (`min: 1`). A sole target that becomes illegal loses the entire resolution. |
| Kykar; Hoarding Dragon | Choose Kykar's blink or its Spirit. The delayed return binds to the object in exile. With two Dragons that found different artifacts, only the dying Dragon recovers its linked card via `WithSource` surviving departure. Bounce and replay the same Dragon, then check the old link. |
| Nine-Lives Familiar, Infernal Vessel, Valkyrie's Call | Capture the Familiar's revival count at death (`k - 1` frozen into the delayed trigger) and spend one on its next-end-step return; cast entry gives 8 revival counters while blink gives 0. In a sibling fork, move it out of the graveyard. Vessel and Call read death-time effective creature types through unified LKI so a returned Demon Vessel or Angel creature does not re-trigger on a second death. Counters and type grants attach to the returned incarnation. Kill Call together with an eligible creature; the trigger still returns the owner's card. |
| Fake Your Own Death, Undying Malice, Celestial Armor | Track which creature received a grant and when it ends. Move the Equipment after its entry trigger resolves. The protection follows its original recipient. When Fake Your Own Death is cast on a stolen creature, its owner gets the tapped card and the trigger's controller gets the Treasure. A dying token still produces Treasure. Malice returns the card tapped with its counter. |
| Angelic Destiny, Ordeal of Nylea | Follow the enchanted creature through death (including simultaneous Aura + creature departure reading `ObjectLki.attached_to`) and the Aura through its own departure. Only the eligible Aura appearance is recovered. Resolve Ordeal's attack, counter threshold, sacrifice and optional library selection through its actual triggers. |
| Raise the Past, Rise of the Dark Realms, Driver, Fiendish Panda, Heartfire Immolator | Put qualifying and excluded cards in both graveyards. Check simultaneous entry, ownership, resulting control and ETB triggers. Driver and Panda choose legal graveyard targets when their triggers are placed, including Panda's last power and its exclusions. Pump Immolator, sacrifice it as the cost and return it before resolution. Its damage uses its departed power. |
| Sun-Blessed Healer, Nullpriest of Oblivion, Feldon's Cane | Read the actual kicked `CastReceipt` in the ETB intervening-if (CR 603.4). An ordinary blink supplies `None` and does not trigger. Each has its own graveyard target restrictions. Check competing restricted spends on the kicked cards. Cane exiles itself as the cost before shuffling. |
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
