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

## Batch 3 · Effective characteristics, grants, combat-damage triggers and stack trigger projections · built

All slices are built (`d93b4c19f6`–`c6acc3f944`). Detail lives in git and the named tests.

- Stack trigger projections: `StackObjectData.chosen_mode`/`x` for every kind; `trigger_description` reads the effects `stacked_trigger_effects`/`pending_trigger_effects` return; `resolution::catalog_join_misses` is the one rule for when stacked and pending frames carry a program (`fdn::check_both_trigger_orders`, `tla::bumi_two_chosen_modes…`).
- Layered characteristics: `Modification::SetName` (Operation::Name deleted), `AllCreatureTypes` as `CreatureTypes::All` (Soulstone Sanctuary), `SetTypes` keeping corresponding subtypes (CR 205.3d; Witness Protection `snc.rs`, Eaten by Piranhas `lci.rs`), `name_set`/`creature_types_set` projected separately with replaced types kept out of `added_subtypes`. All creature types reaches the policy through its shared-effect row.
- CR 302.6: `Permanent::new` always sets `summoning_sick`; `Characteristics::restricted_by_summoning_sickness` is the one creature/haste reading (`permanent_can_tap`, `PermanentData.is_summoning_sick`, the heuristic's attack pool).
- `ZoneEntry.attached_to` attaches entering Auras before `CardMoved`; layered CR 704.5m/n/p attachment SBAs (`rules::cr_704_sba::cr_704_5m_n_p…`).
- `TriggerCondition::SourceDealsCombatDamage` with `#[serde(skip)] combat` on damage events (Dropkick Bomber).
- Dead `GameState.mana_cache` deleted with every manual invalidation; `Game::producible_mana` is public.
- Witnesses: Celestial Armor, Crawling Barrens, Soulstone Sanctuary attacking, Eaten by Piranhas on a hasty creature, Clone under Witness Protection.
- Last debug evidence (after `318154d567`): all 893 cards tests; `rules_tests` + `canonical_replay_tests` + `scenario_tests` 205 (2 ignored); `cr_704` 12 after the SBA change; lib 346 of 347 (the known base failure, `questions.md`). Not rerun after the compress `c6acc3f944` beyond `cr_704`.

Left in Batch 3:

- **Mild-Mannered Librarian & unified activation limits**: add `once_only: bool` (`#[serde(skip_serializing_if = "Not::not")]`, preserving TLA/MSH tokens) alongside two single-owner methods on `ActivatedAbilityDefinition`: `once_per_object()` (`once_only || exhaust || power_up`) and `is_use_limited()` (`once_per_object() || once_each_turn`). Route `Permanent::activation_used` / `record_activation`, `Game::record_activation`, and all three `mana.rs` sites (`1819`, `2952`, `3132-3133` — which currently omit `power_up`) through them. Keep one `Vec<usize>` on `Permanent` for per-object spent indices (staying within the 168-byte ceiling), and derive `PermanentData.exhausted_abilities` by counting only spent indices with `ability.exhaust` so non-exhaust once-only and `power_up` abilities stop printing as `"exhaust abilities used"`, while `PermanentData.used_abilities` (`FactKind::UsedAbility`) continues projecting every spent ability index. Author Mild-Mannered Librarian and its directed check.
- **Batch 3 layer/projection & CR 303.4d SBA fixes** (fold into the same slice before decomposition review):
  - Clear `view.creature_types_set = false` when `Modification::AllCreatureTypes` applies (and guard `observation.rs:1583` with `!characteristics.creature_types.is_all()`), so `SetCreatureTypes` followed by `AllCreatureTypes` never emits `PermanentData.creature_types = Some([])` (`"creature types set to none"`).
  - Gate intermediate-layer `AllCreatureTypes` reads (`layers.rs:781, 1263`) on `view.is_creature() && view.creature_types.is_all()` (CR 205.3d).
  - Return `true` in `aura_attachment_illegal` (`sba.rs:386`) when `view.is_creature()` (CR 303.4d: an Aura that is also a creature cannot enchant anything), with directed tests for all three cases.
- **Decomposition review of the whole batch**, tracing layered characteristics, activation limits and trigger projection through their consumers (observation, text, tokens, heuristic).
  - First pass (2026-11-24, at `42a239fdec`): **not met**. No `mana_cache`/`affects_mana`/`invalidate_mana_cache`/`cached_producible_mana` identifiers remain. Soulstone Sanctuary reaches observation, text and tokens through one layer value (`layers.rs:1075` → `finish_object` → shared-effect row, `ability_description.rs:1376`), and Dropkick Bomber through one matcher (`triggers.rs:1787-1791`) with template text and the observability guard. The three planned defects are still present: `layers.rs:1075` leaves `creature_types_set` true after `SetCreatureTypes` then `AllCreatureTypes`, so `observation.rs:1583` projects `Some([])`. `layers.rs:781` and `1263` read `is_all()` mid-pass without `view.is_creature()`. `aura_attachment_illegal` (`sba.rs:386`) keeps a creature Aura attached. The activation-limit predicate is restated at five sites: `permanent.rs:251-261`, `play.rs:1356`, and `mana.rs:1819`, `2952` and `3132-3133`, and the three `mana.rs` sites omit `power_up`. The Librarian slice's `once_per_object`/`is_use_limited` owner removes that duplication. These findings came from reading the code: no suites were rerun in this pass. Rerun the review after the slice lands.
- Folded elsewhere: the death/LKI witnesses (Witness Protection's Citizen, Imprisoned in the Moon's noncreature Land) are the representation review's "mooned creature dying"; CR 603.6a Aura-entry suppression waits for a catalog witness (`questions.md`).

Next slice: Mild-Mannered Librarian with unified activation limits plus the `AllCreatureTypes` and CR 303.4d SBA fixes, then the batch's decomposition review, then Batch 4's form representation.

Use first printings and record each delivered definition and directed check in the [FDN ledger](../.lf/wave/coverage/research/standard-2026-11/sets/fdn.jsonl) where it applies. Match tokens by their full printed characteristics. Scope comes from Task 05 of the [milestone](../.lf/wave/coverage/research/standard-2026-11/milestone.md).

## Batch 4 · Representation, consumers and budget

- Replace the split across `Object.copy`, `Object.printed`, `Object.front`, `Permanent.manifested` and `ZoneManager.melded_cards` by moving rare form and copy state behind `Object.form: Option<Box<BattlefieldForm>>` (shrinking `Object` from its 96-byte ceiling to ~40 bytes with zero allocation for ordinary cards, while keeping `Permanent` within 168 bytes). Express physical backing (single card, melded pair, token, transforming token), orientation/face-down state and `Permanent`'s stable manifest public-id alias separately from materialized `CopiableValues`. Stop copying `source.front` in `Object::become_copy_of` (`game_object.rs:644`): a non-transforming copy of a transformed DFC (Clone copying transformed Cecil) keeps only the shown face's copiable values (MV 0 per CR 712.8e, cannot transform per CR 712.9), while a transforming token copy (CR 707.8a, e.g. Ember Island Production on Avatar Kuruk or Electroduplicate on Cecil) retains both faces and can transform. Keep raw backing face(s) separate from exceptioned `PrintedCharacteristics` views so every face transition applies `CopyEffect.exceptions` once from the raw face and preserves `CopyEffect.of` (fixing `turn_front_face_up` / `mint_appearance` double-applying `GrantTriggered` and `GrantStatic` onto an already-exceptioned `Object.front` under CR 707.8a/707.9). Route both `transform_permanent` (`zones.rs:937`) and `FrameCompletion::TurnFaceUp` (`decision.rs:1413`) through one battlefield form-transition helper that enforces CR 712.9/712.10 backing rules, and share one `is_face_down()` across `identity.rs`, `action.rs` and `selection_sampling.rs`.
- Fix the canonical-hash collision on hidden/overridden battlefield forms in `hash.rs` (`CanonicalCard`): omit nothing that distinguishes two authoritative states — serialize optional `backing_definition_id` when the card's retained physical printing differs from `definition_id` (manifested cards, melded primary card, entry copies) and `front_definition_id` when transformed, omitting both on ordinary cards so ordinary-game hashes stay untouched while two swapped manifested permanents can never collide in `GameState::deterministic_hash()`.
- Route `Decision::ExileLinked` / `cast_permission::exile_linked` (confirmed sole remaining battlefield-capable immediate `move_object` caller; `payment.rs:1071` already uses `EffectFrame.batch`) through `EffectFrame.batch`, using the same event results that Airbending Lesson consumes. The player performing the exile chooses relative order, even when the ability controller receives the cast grants. In each ordering fork, keep Ragnarok assembled and its grants unissued while the order is pending. After the ordering Command, grant Azula's controller access to both exact exile arrivals before offering the next graveyard selection. Reject the spent Command without advancing the frame or adding grants. Keep the Hama and ordinary Azula checks as single-arrival controls.
- Finish the consumer checks and measure the engine budget below.

## Open checks on finished work

- Lost Days: the order choice should be visible only to its owner, and the game should replay from its record.
- Horde, Driver and Fire Crystal still need viewer projections and replay of their recorded Commands from a recorded root. Their native fork and restore checks already pass.
- The named Coil setup has no seed-to-position replay file.
- Manifest sampling: there is no pending private selection in the fixture. Constrained sampling, paired seeded searches and comparative viewer trace exports remain open.
- Batch 1's constructed fixtures still need viewer projections, text, tokens and Command replay from a recorded root.
- At gate, on the rebuilt extension: Python trigger-order and observation tests and Engine Budget digests. Serialized bytes changed for modal trigger stacks (`chosen_modes`), triggered stack rows' `stack::X`/`stack::MODE`, `summoning_sick` on noncreature and printed-haste permanents, and removing `mana_cache` from `CanonicalMatchState`, `AllCreatureTypes` in `MODIFICATION_NAMES`, and `creature_types`/`temp_subtypes` projections for replaced creature types. `observation_json::…historical_captures_move_battlefield_and_spell_rows_onto_their_records` also fails at base `25ba418a2a` (`questions.md`).
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
