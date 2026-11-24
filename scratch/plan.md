# MTG-420 · Battlefield forms, copies and persistent ability sources

> “Stop doing 1 card per `coverage-working` loop: group remaining in-scope ledger rows into cohesive batches by shared mechanism/test setup so you finish in a few substantial passes.”

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
- Compress `9e297f0c`: waiting triggers read the captured face. The focused nextest run passed 42 cases. The 13 Python copy-text, Academy trigger-order and pinned-game checks passed. The merge `039b275b` brought only Loopflow workflow files.
- Batch 1 exact departures and linked returns (`e6f6ef27`, `47dec0aa`, `2955a3d8`). Effective `ObjectLki` uses the same matcher as live characteristics. `WithSource` exile links survive departure, and delayed returns bind to exact arrivals. The card list and directed evidence are in [the behavior review](behavior-review.md). The implementation run passed 14 departure cases, 142 focused card and observability cases, and the canonical replay and LKI/exile filters.

## Batch 2 · Graveyard choices, costs and simultaneous movement · done

Built and reviewed (cohort decomposition review at `c7e3270240`: met). Detail lives in git and the named tests.

- Spell entries (`d61564193a`): `SpellEntry` (`state/src/permanent.rs`) owns kicker, X and payment on the permanent; `Arc` keeps `Permanent` at 168 bytes. Healer (`fdn.rs`) and Nullpriest (`znr.rs`) share `check_kicked_return`.
- Source-relative targets (`7e1fa0b92e`, `24ac46fe41`, `f4b34e568f`, `705db6bab6`): placement, mode availability, retargeting, resolution and activation offers share `target_fits_clause`/`legal_clause_targets` with the exact `AbilitySource`; `TargetRequirement::another`. Witnesses: Panda, Driver, Diaochan (`ptk.rs`), Koma (`khm.rs`), Disciple/Patrol/Protectors.
- Public-zone movement (`8d76e65987`, `c1c8a2f490`): `ObjectSelector::ZoneMatching`; Raise the Past (`fdn.rs`), Rise of the Dark Realms (`m14.rs`); `Game::object_mana_value` is the one mana-value reader.
- Cohort finish: `EffectFrame.attached` with Ordeal of Nylea (`8dd3afd0ef`, `ths.rs`), Heartfire Immolator (`c8ecfe85b3`, `m21.rs`), Valkyrie's Call (`96982d4030`), clause gaps (`261b034b2e`), modal trigger owner `chosen_modes`/`modes_chosen` (`7914e67f5f`).
- Last debug evidence (after `3702646188`): 880 cards-crate tests, `rules_tests` 192 (2 ignored), `canonical_replay_tests` 6, 26 `flow::target_authority_tests`. Merges `361823be59` and `4c10520b67` landed after parts of this evidence: rerun vector/recording consumers and replay at gate.

Left from Batch 2, by trigger rather than schedule:

- `activated_ability_targets` (`play.rs`) applies distinctness and the shared-type filter itself instead of calling `can_add_clause_target`. Fold it in when an activation first declares `total_mana_value_x` or a spell needs the partner lookahead.
- Defer general delayed-effect value freezing until a second card needs it (`CreateDelayedTrigger` freezes `DeathCountersMinus` for Nine-Lives Familiar; proposal in `scratch/concept-review.html`).
- **Abyssal Harvester depends on MTG-418**, which must provide the native query for exact graveyard arrivals this turn from any zone (`TurnState::graveyard_from_battlefield` records only battlefield departures). No PR or release yet. Once it reaches `main`: `lf sync`, then `rg -n 'graveyard_from_battlefield|GraveyardThisTurn|graveyard.*this turn' managym/src/flow managym/state/src`. Offer Harvester a milled creature, a dead creature and a card that left and reentered; follow the selected exile appearance through the Nightmare copy and the exile of the other Nightmare tokens.

## Batch 3 · Effective characteristics, grants, combat-damage triggers and stack trigger projections · done

All slices are built (`d93b4c19f6` through the activation-limit commit after `ab832ec7d5`). Detail lives in git and the named tests.

- Stack trigger projections: `StackObjectData.chosen_mode`/`x` for every kind; `trigger_description` reads the effects `stacked_trigger_effects`/`pending_trigger_effects` return; `resolution::catalog_join_misses` is the one rule for when stacked and pending frames carry a program (`fdn::check_both_trigger_orders`, `tla::bumi_two_chosen_modes…`).
- Layered characteristics: `Modification::SetName`, `AllCreatureTypes` as `CreatureTypes::All` (Soulstone Sanctuary), `SetTypes` keeping corresponding subtypes (CR 205.3d; Witness Protection, Eaten by Piranhas), `name_set`/`creature_types_set` projected separately. A later `AllCreatureTypes` clears `creature_types_set`; mid-pass `All` counts only for a creature through `Characteristics::has_all_creature_types` (`fdn::all_creature_types_overrides_a_set_list_and_needs_a_creature`).
- CR 302.6: `Characteristics::restricted_by_summoning_sickness` is the one creature/haste reading.
- `ZoneEntry.attached_to`; layered CR 704.5m/n/p attachment SBAs, and an Aura that is a creature is illegally attached (CR 303.4d; `rules::cr_704_sba::cr_704_5m_n_p…`).
- `TriggerCondition::SourceDealsCombatDamage` (Dropkick Bomber). `GameState.mana_cache` deleted.
- Activation limits: `ActivatedAbilityDefinition::once_per_object()` (`once_only || exhaust || power_up`) and `is_use_limited()` are the one owner; `Permanent.spent_abilities` (serialized as `exhausted`) holds per-object spent indices. `PermanentData.exhausted_abilities` counts only exhaust abilities; `used_abilities` lists every spent index. Mild-Mannered Librarian (`j22.rs`, `j22::mild_mannered_librarian_becomes_a_werewolf_only_once`); `Werewolf` added to the subtype table.
- Debug evidence after the activation-limit commit: all 895 cards tests; `rules_tests` + `canonical_replay_tests` + `scenario_tests` 206 (2 skipped); lib `mana|observation|layers|sba` 98 of 99 (the known base failure, `questions.md`).
- Decomposition review, second pass (2026-11-24, after the activation-limit commit): **met**. Every use-limit read goes through `once_per_object`/`is_use_limited` (`Permanent::activation_used`/`record_activation`, the three `mana.rs` sites, the mana test); `statics.rs` reads `power_up` only for its cost reduction. Soulstone Sanctuary and Dropkick Bomber reach observation, text and tokens through one layer value and one matcher; the set-list/All projection, mid-pass reads and creature-Aura SBA each have a directed test that fails without the fix (the Forest witness gets +1/+1 from Allied Teamwork without the mid-pass gate).
- Decomposition review, third pass (2026-11-24, at `a855e140aa`, after compress `77a073e5bf`): **met**. Followed Mild-Mannered Librarian (offer `action.rs`, activation `play.rs`, mana `mana.rs:1807/1819/2953/3132`, observation `used_abilities`/`exhausted_abilities`) and Soulstone Sanctuary (layer `AllCreatureTypes` clears `creature_types_set`; mid-pass `has_subtype`, `buff`, `matches_view` all read `has_all_creature_types`; post-pass readers see `finish_object`'s normalized value). No card-key dispatch in `flow/` or `agent/`; the removed `Game::record_activation` wrapper has no callers left. Debug at HEAD: cards 895/895; `rules_tests` + `canonical_replay_tests` + `scenario_tests` 206 (2 skipped); lib `mana|observation|layers|sba` 98/99 (known base failure). Kindred creature types are a deferred gap with no catalog witness (`questions.md`).
- Folded elsewhere: the death/LKI witnesses are the representation review's "mooned creature dying"; CR 603.6a Aura-entry suppression waits for a catalog witness (`questions.md`).


Use first printings and record each delivered definition and directed check in the [FDN ledger](../.lf/wave/coverage/research/standard-2026-11/sets/fdn.jsonl) where it applies. Match tokens by their full printed characteristics. Scope comes from Task 05 of the [milestone](../.lf/wave/coverage/research/standard-2026-11/milestone.md).

## Batch 4 · Representation, consumers and budget

Next: slice 3 (close two post-slice-2 concept-review mismatches, then representation review, consumer checks, Engine Budget). Slices 1 and 2 are built and compressed; slice 2's directed witnesses fail without their fixes. Start slice 3 with the two concrete mismatches surfaced in [the concept review](concept-review.html) (downstream viewer-safe identity/LKI provenance + `decision_capture` completeness, and `selection_hidden_allocations` reading `retained_base()` printings), then run the two-root manifest privacy comparison (opponent observation, v11 buffers, text, Academy offers/context, viewer history) on the slice-2 combat-trade, bounce and targeting fixtures.

Build in three slices (rationale and counterexamples in [the concept review](concept-review.html)):

1. **Boxed `BattlefieldForm`, backing-owned transformability, canonical hash and face-down privacy · done.**
   - `Object.form: Option<Box<BattlefieldForm>>` (`state/src/game_object.rs`) replaces `copy`/`front`/`printed`: `backing` (the printing behind a manifest, meld result or entry copy), raw `TransformingFaces` (front, back, `back_up`), `face_down` and `copy`. `Object` is now ≤ 40 bytes (`game.rs` ceiling tightened from 96); `Permanent` unchanged. `ZoneManager.melded_cards` untouched.
   - `ContentPack::transforming_faces` is the one transformability authority (form faces, else the catalog DFC row of an object with no form); `object_row().back_face`, `copiable_values().faces` (replacing `front`) and `battlefield_entry_face` read it. `ContentPack::back_face` deleted. `Object` reads the form through `copy()`, `backing()`, `form_faces()` (held faces only, never the authority) and `is_face_down()`. `become_copy_of` drops faces (single printing backing).
   - `Game::turn_over(object, FormTransition::{Transform, TurnFaceUp})` (`zones.rs`) is the one transition helper; `Object::show_transforming_face` applies `copy.exceptions` once to the raw face and keeps `copy.of`. `Object::is_face_down` replaces the three `manifested && ANONYMOUS` checks; `Permanent.manifested` documents only its public-id alias role.
   - `CanonicalCard` adds optional `backing_definition_id` and `front_definition_id` (omitted on ordinary objects).
   - `source_focus` reads `observed_object_id`; `Observation::event_data(game, event)` names a face-down `DamageDealt`/`AbilityTriggered`/`AbilityCountered` source by its permanent's `object_event_ref`.
   - Witnesses: `fin::clone_of_either_cecil_face_is_single_faced`, `fin::copied_cecil_tokens_transform_with_their_exceptions_once` (Electroduplicate copying Ember Island's nonlegendary Cecil Hero: five exceptions, once each, over three transforms; MV 1 on the back face), `fin::ragnarok_token_copy_departs_without_ordering_or_moving_either_card` (existing), `dsk::swapped_hidden_manifests_hash_apart_and_look_alike`, `dsk::a_face_down_attacker_deals_damage_without_naming_its_card` (fails without the masking: `(1, 16)` vs `(4, 170)`).
   - Taught: a legendary copy under the same controller dies to the legend rule, so copy witnesses use the opponent's Cecil or a nonlegendary copy. Ordinary-card hashes are unchanged; transformed objects now also hash their front id, and token copies keep `copy.of` across transforms.
   - Decomposition review (2026-11-24, at `2abd082655`; no engine change since `fb8bcba24a`): **met**. Followed Cecil (Clone of either face, Electroduplicate token: `copiable_values`/`object_row`/`battlefield_entry_face`/`turn_over` all read `ContentPack::transforming_faces`; `back_face_row` is its only catalog reader) and a manifest (`turn_face_down` → `turn_over(TurnFaceUp)` → `show` applies `copy.exceptions` once; `mint_appearance` restores the backing). `Object.copy`/`front`/`printed` and `ContentPack::back_face` are gone; no card-key dispatch in `flow/`, `agent/`, `content_pack.rs` or `state/`. Debug `fin::`+`dsk::` 27/27 at HEAD. Its residuals (unused `Object::token_copy`, departed DFC faces, split face-down predicates) closed in slice 2.
   - Debug at this slice: cards 899/899; `rules_tests` + `canonical_replay_tests` + `scenario_tests` 206 (2 skipped); lib filters `mana|observation|layers|sba|card_capture|hash|identity|zones` 106/107 (known base failure) and `possible_worlds|search|selection_sampling|resolution|target_authority|decision` 163/163. Gate: Python observation/trigger-order/pinned-pilot on a rebuilt extension; Engine Budget digests (serialized `CopiableValues.faces`, hash fields).
2. **Slice-1 edge closures, resumable `ExileLinked` and joint manifest sampling · done** (`bdb99f6636`..`92eede10a8`; compress `bbe51bda2f` folds the foretell set-aside into one pass and keeps manifest-free deals rng-free).
   - `ContentPack::catalog_faces` is shared by `transforming_faces` (no form) and `departed_copiable_values` (no snapshot); a departed front-face Cecil's token copy transforms, a departed Clone of Cecil's does not (`flow::resolution_tests::departed_copy_source_keeps_catalog_faces_only_on_a_transforming_card`, synthetic `CreateTokenCopy { of: This }`: no catalog card copies a departed DFC, since Electroduplicate and Ember Island fizzle without their target). `Object::token_copy` deleted; `cr_400_identity` builds tokens with `Object::from_definition`.
   - `Game::hides_physical_card` (live `manifested` or face-down permanent, or a departure whose LKI is `ANONYMOUS`) is the one predicate for `observed_object_id` and `Observation::event_data`; a face-down `CardMoved` departure into Hand/Library names no source. Witnesses `dsk::a_face_down_creature_trading_in_combat_names_only_its_permanent`, `dsk::a_face_down_permanent_bounced_to_hand_names_no_card`.
   - `EventBinding::ExileLinked { actor, grantee, source, grant }`: the selection proposes a batch; `run_frame` passes the binding's actor to `meld_order_choice`; `bind_event_results` calls `grant_exile_linked` per arrival (each its own mana value). The program projection serializes the binding with `source: None` and names the source through `object("linked source")`; `ExileLinked` and the fields `actor`/`grantee`/`source` are in `record_vocabulary.rs`. Witness `fin::azula_ragnarok_exile_is_ordered_by_the_exiling_owner_and_grants_both_cards` (fails when the controller orders); Hama and Azula controls pass.
   - `selection_hidden_allocations` enumerates `(foretold, manifest)` name fills with weight `ways · F! · M!`; `prepare_selection_hidden_slots` returns the drawn manifests; `draw_manifests` (unconstrained materialize and `determinize`) sets the first `F` foretell cards aside and draws `M` uniformly from the rest with the deal's rng; `fill_manifest_slots` only installs printings. Witnesses `hidden_selection_manifest_fill_leaves_the_committed_library_cards` (weight 2 not 1; `materialize_index` and `determinize` seeds 0–7) and `foretold_and_manifest_slots_fill_without_prefix_bias` (400 seeds; 0.68 before).
   - Taught: setting the foretold cards aside and then taking the rest's first cards is still biased (the later foretell card sits late in the shuffle), so the manifest draw needs its own rng draw. `determinize` keeps rejection sampling for constrained roots (already correct in distribution).
   - Debug at `92eede10a8`: cards 902/902; `rules_tests` + `canonical_replay_tests` + `scenario_tests` 206 (2 ignored); lib filters `possible_worlds|search|selection_sampling|resolution|target_authority|decision|mana|observation|layers|sba|card_capture|hash|identity|zones` 242/243 (known base failure); `cargo check --all-targets` clean. Gate: Python observation/trigger-order/pinned-pilot on a rebuilt extension; Engine Budget digests (manifest draws now consume rng when `M > 0`; ExileLinked program tokens).
3. **Consumer identity/LKI closures, manifest printing matcher, representation review, consumer checks and Engine Budget:**
   - **Close downstream viewer-safe identity, LKI provenance and capture completeness** (`decision_context.rs`, `continuation.rs`, `effects.rs`, `decision_capture.rs`, `identity.rs`):
     - `decision_object` routes `display_id` through `observed_object_id(object)` and aliases `entity_id` to `observed_object_id(object).0` when `hides_physical_card(object)` (leaving ordinary non-manifest `entity_id = object.slot() as u32` bitwise unchanged); `presentation_subjects` skips `"object:<slot>"` when `hides_physical_card(*object)`.
     - Add `Game::observed_source_definition(object) -> Option<CardDefId>` (`object_lki` → live `definition_id` → `source_base`, always `.catalog_id()`, never `ANONYMOUS` or a bounced manifest backing) and a viewer-safe event name helper for `decision_context.rs::recent_facts` (`PermanentSacrificed`, `AbilityCountered`, returning `"a face-down permanent"` when live `is_face_down()` or departed `ANONYMOUS` LKI). Use them in `effects.rs:56,82`, `continuation.rs:489` (fallback only when `source.definition.is_none() && !self.hides_physical_card(source_object)`), and `decision_context.rs:241-268`.
     - In `decision_capture.rs`, guard `stack.source_card_registry_key` and `effect.source_registry_key` with `u32::try_from(...)`, and collect `permanent.copy` (`registry_key` and `exceptions[*].source_registry_key`) via `u32::try_from(...)` into `capture.definitions`.
     - Witnesses: (a) two-root manifest comparison across `decision_offer_subjects`, `decision_context`, `presentation_subjects`, `capture_decision` (targeting a face-down 2/2, a granted stack ability/effect from a face-down permanent, and a countered ability after bouncing the manifest to hand); (b) `capture_decision` + `render_decision` on a transformed token copy of `Cecil, Dark Knight` under closed decklists with no other front-face Cecil visible.
   - **Read `retained_base()` printings in `selection_hidden_allocations` and `inspection_slot_matches`** (`selection_sampling.rs`):
     - Evaluate hypothetical `Hand`/`Library` slots against `self.state[object].retained_base()` (`ObjectKind::Card`, printed mana value, CDA power when `predicate.reads_power()`, matching `recorded_object_matches_predicate`) instead of `object_matches_predicate` (which hits `permanent_matches_predicate` on a live face-down manifest), and compare `retained_base().0` in `inspection_slot_matches`.
     - Witnesses: sole-copy manifested `Forest` under `"select 2 creatures"` library selection (impossible world gets weight 0, never false-positive `InconsistentWorld`) and under `"select 2 lands"` library selection (positive weight, materializes on attempt 1).
   - **Representation review, consumer checks and Engine Budget:** Finish the remaining representation review witnesses (Cecil, Clone of either Cecil face, departed Cecil and the Ragnarok token copy are witnessed in slices 1–2), two-root manifest privacy checks across native and Python consumers (observation, v11 buffers, LLM text, Academy reconnect, viewer history, combat damage/trade, bounce-to-hand, triggers, constrained sampling, paired seeded searches and comparative viewer trace exports), replay/fork checks, pinned pilot and Engine Budget measurements below.

## Open checks on finished work

- Lost Days: the order choice should be visible only to its owner, and the game should replay from its record.
- Horde, Driver and Fire Crystal still need viewer projections and replay of their recorded Commands from a recorded root. Their native fork and restore checks already pass.
- The named Coil setup has no seed-to-position replay file.
- Manifest sampling: there is no pending private selection in the fixture. Constrained sampling, paired seeded searches and comparative viewer trace exports remain open.
- Batch 1's constructed fixtures still need viewer projections, text, tokens and Command replay from a recorded root.
- At gate, on the rebuilt extension: Python trigger-order and observation tests and Engine Budget digests. Serialized bytes changed for modal trigger stacks (`chosen_modes`), triggered stack rows' `stack::X`/`stack::MODE`, `summoning_sick` on noncreature and printed-haste permanents, and removing `mana_cache` from `CanonicalMatchState`, `AllCreatureTypes` in `MODIFICATION_NAMES`, `once_only` on activated abilities and `exhausted_abilities` no longer counting power-up spends, and `creature_types`/`temp_subtypes` projections for replaced creature types. `observation_json::…historical_captures_move_battlefield_and_spell_rows_onto_their_records` also fails at base `25ba418a2a` (`questions.md`).
- Run the pinned pilot after rebuilding. Earth Kingdom Jailer's explicit optional `TargetClause` changed its learning program. Check any mismatch before recording an intentional break with `uv run python scripts/pinned_pilot.py record --reason "<why>"`.

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
