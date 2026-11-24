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

## Batch 2 · Graveyard choices, costs and simultaneous movement

### A. Spell entries · built

`SpellEntry` owns kicker, X and optional payment through the proposed move and on the resulting permanent. A cast for zero carries an empty receipt. A permanent spell copy keeps kicker and X with no payment. A blink or battlefield copy has no spell entry. This supersedes the reviewed `CastReceipt` model because casting choices can survive without a cast. `Arc<SpellEntry>` keeps the permanent at its 168-byte ceiling, down from the 184-byte inline attempt. The digest projection retains `was_cast`, `cast_x` and `cast_spent`. The copied spell's cast flag is corrected. See `managym/state/src/permanent.rs` and `managym/src/flow/decision.rs` at `d61564193a`.

The entry answers X counters and Familiar's cast-only condition before commitment. The entering permanent's own trigger captures kicker, X and spent mana. Its intervening-if recheck can still read kicker after the source leaves. Healer (`fdn.rs`) and Nullpriest (`znr.rs`) share `check_kicked_return` in `managym/cards/tests/cards.rs`. It forks kicker and target Commands, rejects stale submissions, exercises Cleric-only mana, removes the source in response and blinks it in a sibling branch. The chooser sees the captured kicker during placement and both viewers see it after removal.

Eleven directed debug checks passed, including Familiar's cast-only replacement, Toph's spend receipt and a copied permanent spell's X, kicker and absent payment. Compression at `12c51346a1` and `7cd57bfb32` passed 81 focused Rust tests and 14 trigger-order Academy and semantic-text tests with the rebuilt CPython 3.13 extension. Both trigger orders at seed 7 produced identical state digests and both viewers' observation JSON before and after compression. The departure fixture distinguishes Gray Ogre's printing from its Mountain face at departure and its later copy. The merge of `b9b09bc6d9` added observation extents and changed policy preparation without edits in the rules flow or state crates. Those checks predate the merge and did not include the pinned pilot.

### B. Source-relative target legality · built at `7e1fa0b92e`

Trigger placement, mode availability, retargeting and resolution share `target_fits_clause`. It checks the base target spec, excludes the source for “another” and reads the power bound from the exact live or departed `AbilitySource`. `departure_arrivals` identifies each exact graveyard arrival until its next move. Distinctness and the announced-X budget apply together when choosing or replacing a target. At `7e1fa0b92e` activated-ability target offers kept their own path (`activated_ability_targets`), which excluded the source from every clause and ignored `another` and the source power bound. That was a rules defect, not a compatibility choice: the opponent choosing Diaochan's second target cannot choose Diaochan (CR 115.1, no “another”), and the equipped creature cannot target itself with Trusty Boomerang's granted tap ability. 2B′ below closed that gap. The source power bound remains a `TargetSpec` twin (`GraveyardCardYouOwnWithinSourcePower`, from TLA `4c8324a595`) rather than a clause field; `target_fits_clause` is its only reader, and its serialized tokens predate this Task, so it stays.

Fiendish Panda's life-gain condition reads committed life increases. Simultaneous lifelink damage gains life once per source. With two attacking Sires, one blocked by two Ogres, Panda receives two counters for fourteen life. A Clone that dies as a 4-power Panda can return Driver. The physical Clone meets the ordinary card predicate, so “another” alone excludes its graveyard arrival. Move that card through the hand and back to the graveyard in a sibling retargeting branch and it becomes eligible. Driver's trigger on the same board can return only Llanowar Elves. A constructed modal trigger on Hill Giant offers the 3-mana Ogre and omits the mode requiring a 4-mana card. Savor in response leaves the source alive at 1 power and invalidates the Ogre at resolution.

Twenty-one focused debug checks passed, including Panda, Driver, Kykar, Jailer, Wolfbat, Scavengers, Sozin's X budget, station, Mayhem, trigger tokens and Scryfall shell conformance. The rebuilt CPython 3.13 extension admitted Panda and Driver. The Python catalog check found the life-gain condition in Panda's first ability program and the death condition and power bound in its second. Full suites and the pinned pilot remain at gate.

At `24ac46fe41`, retargeting and resolution read the same stack clauses. The departed Gray Ogre fixture checks the full replacement offer after its physical card returns as a copy. All 67 focused Rust tests passed. Four stack and continuation text tests passed against the rebuilt CPython 3.13 extension. On both seed-23 Deadly Precision target branches, the state digest, both viewers’ observation JSON and the acting decision capture matched the pre-compression build at payment, on the stack and after the two priority passes.

The merge `361823be59` brought the native snapshot self-play workers, recording updates and their bindings. It left the target and movement authorities and card fixtures intact. The recorded Batch 2B verification predates this merge. Run the consumer and replay checks against the merged bindings at gate.

### B′. Activated targets through the shared clause · built at `f4b34e568f`

`activated_ability_targets` reads `legal_clause_targets` with the activation's exact `AbilitySource` (`activation_ability_source`: the live permanent's object, or the card's object in its activation zone), so offer, choice and resolution apply one clause rule. The blanket source exclusion is gone; distinctness within the clause and the shared-creature-type partner rule stay. Three TLA activations whose text says “another” now declare it on a `TargetClause`: Deserter's Disciple, North Pole Patrol and Earth Kingdom Protectors (the `}: …another target` grep missed Protectors' sacrifice cost; its existing test caught it). Directed checks: Diaochan's opponent is offered Diaochan (`ptk.rs`), Koma's tap mode offers Koma (`khm.rs`), and the Disciple, Patrol and Protectors tests still omit their sources. All 873 card-crate tests passed in debug. The pinned pilot passed against the rebuilt CPython 3.13 extension without a recorded break. Engine rule suites stay at gate.

Compression `705db6bab6` added `TargetRequirement::another(spec)` (equal to `{ another: true, ..one(spec) }`) and replaced the spelled-out clauses in `fdn`, `kld`, `ncc` and `tla`; At `f03864e4cd` all 873 card-crate tests passed in debug after it, including the Diaochan, Koma, Disciple, Patrol and Protectors witnesses; every card whose text says “another target” declares `another`. The merge `4c10520b67` changed vector-env slots, recording and rollout collection, not `flow/` or cards. Rerun the vector/recording consumer checks and the pinned pilot at gate.

### C. Simultaneous public-zone movement · next

Add `ObjectSelector::ZoneMatching { zone, owner: Actor, predicate: CardPredicate }` beside `Each` in `Game::select` for non-battlefield card zones (owner-scoped physical cards, distinct from `Each`'s controller-scoped battlefield permanents and `Zone`'s hidden-zone sampling provenance). Use `Actor::includes` in deterministic seat order, `GameState::zone_cards` for physical cards and `object_matches_predicate` for the filter. Include the selector in `Effect::SelectCards`'s `cards_only` guard. `Effect::Move` already commits its selections as one `EventBatch`, so both cards can use that path. Raise the Past selects the caster's creatures with mana value at most two. Rise of the Dark Realms selects all players' creatures and uses the existing `under: None` default to return them under the caster's control. Preserve `Zone(_)` and `Each` serialization. Cover the new selector in `ability_description.rs`, learning tokens and `observability.rs`.

Unify mana-value reads in the same pass:
- Delete duplicate `Game::permanent_mana_value` (`resolution.rs:2405`) and filter `ObjectSelector::Each` through `Game::object_mana_value`, which already uses the front cost for transformed DFCs (CR 712.8e) and sums both cards' costs for physical melds (CR 712.8g).
- In `Game::object_matches_predicate` (`statics.rs:511`), pass `self.object_mana_value(object_id)` into `predicate.matches_printing(...)` so `SpellMatching` predicates include announced `X` on stack spells (CR 202.3e).
- In `Game::entry_matches_predicate` (`statics.rs:588`), pass mana value `0` when `entry.manifested` is true (CR 708.2) instead of reading the hidden card's printed cost.

Use Day of Black Sun at X=0 to expose the duplicate. A manifested creature has mana value zero and should be destroyed. Transformed Cecil and a physical Ragnarok should keep their abilities and survive. Extend the existing Day of Black Sun check or use the constructed form setup alongside it.

Use the two-graveyard setup for both spells. Check that entering Momo, Friendly Flier sees an entering Spectral Sailor. Use a Sailor owned by the opponent for the Rise fork to check ownership and control. Driver stays a death-trigger witness in B.

### Finish and review the cohort

- Heartfire Immolator gets prowess and a self-sacrifice cost. `EffectValue::SourcePower` already reads `battlefield_power` for the exact source, as the target clause does. Pump it, sacrifice it, return it before resolution and check damage from the sacrificed appearance's power.
- Ordeal of Nylea follows the enchanted creature's attack, counter threshold, sacrifice and optional basic-land search through native triggers. In the same pass, separate `EffectFrame.attached: Option<ObjectId>` (`skip_serializing_if = "Option::is_none"`) from `EffectFrame.context_target` (`decision.rs:193`, `resolution.rs:390, 594`): keep `context_target` strictly for triggering event objects (`ObjectSelector::Triggering` and untargeted `primary_target`), populate `attached` from `ActivatedAbilityOnStack.attached` for activations, and let `frame_attached` fall back to live `attached_permanent` or `object_lki(source).attached_to` for triggers so `ObjectSelector::Attached` never confuses an event target with an enchanted permanent and survives Aura/Equipment departure. Feldon's Cane is already defined in `atq.rs` with its directed self-exile and shuffle check (`atq::feldons_cane_is_exiled_as_a_cost_and_shuffles_the_graveyard_into_the_library`); record its verified check in `fdn.jsonl`.
- Pair Undying Malice or Valkyrie's Call with simultaneous deaths and a blink in response, as carried from Batch 1.
- Close the two remaining target-clause gaps alongside the Batch 2 finish: apply the shared `targets_share_creature_type` check in `retarget_candidates` (`retarget.rs:52`) so retargeting or copying Secret Tunnel cannot produce an illegal target pair, and check `>= requirement.min` (instead of `>= 1`) in `activation_targets_available` (`play.rs:1422`) and `legal_trigger_modes` (`triggers.rs:484`) so multi-target clauses match `legal_spell_modes`.
- Separate chosen-mode printed triggers from `inline_effects` (recorded in `scratch/concept-review.html`): carry `chosen_modes: Vec<usize>` on `TriggeredAbilityOnStack` (`skip_serializing_if = "Vec::is_empty"` so non-modal bytes stay unchanged) and `modes_chosen: bool` on `PendingTrigger` (keeping 0-mode `up_to` declines distinct from unchosen modes), leaving `inline_effects` strictly for non-printed triggers (delayed, reflexive, granted, firebending). Share one `stacked_trigger_effects(trigger)` helper across `resolve_triggered_ability`, `stack_target_requirements` and `ability_description`, while keeping `stacked_trigger_program` strictly for viewer-unrecoverable programs. This preserves `intervening_if` (CR 603.4), `ability_resolutions_this_turn`, `sba.rs` Saga chapter detection, real trigger descriptions (instead of `"delayed ability"`) and `ContinuationFrame` mode/program projection for modal triggers.
- Review the whole batch together. Each card composes from shared rules and has a directed debug case in its first-printing set's test file. Check the `SpellEntry` owner, shared clause predicate, `attached` vs `context_target` split, trigger-effect owner and single `Effect::Move` path.

Defer general delayed-effect value freezing until a second card needs it. `CreateDelayedTrigger` freezes `DeathCountersMinus` for Nine-Lives Familiar. The post-Batch 1 concept review in `scratch/concept-review.html` records the proposed generalization if another consumer appears.

**Abyssal Harvester depends on MTG-418.** It needs the native query for exact graveyard arrivals this turn from any zone. `TurnState::graveyard_from_battlefield` records only battlefield departures. MTG-418 has no published PR or release for the query. Once it reaches `main`, run `lf sync` and locate the query with `rg -n 'graveyard_from_battlefield|GraveyardThisTurn|graveyard.*this turn' managym/src/flow managym/state/src`. Offer Harvester a milled creature, a dead creature and a card that left and reentered. Follow the selected exile appearance through the Nightmare copy and the exile of the other Nightmare tokens. Continue the rest of this batch independently.

## Batch 3 · Effective characteristics and grants

- Add `Modification::SetName(String)` so Witness Protection, Eaten by Piranhas and Imprisoned in the Moon (already defined in `emn.rs`) use `StaticAbility::Continuous`. Lower `StaticAbility::AttachedSetsCharacteristics` through `Modification::SetName` in `layers.rs` and delete `Operation::Name`. Preserve Honest Work's TLA learning tokens.
- Add Layer 4 `Modification::AllCreatureTypes` and `all_creature_types: bool` on `Characteristics` and `BattlefieldFacts` for Soulstone Sanctuary (CR 205.3m) rather than enumerating 100+ creature subtype strings into `Vec<String>` and overflowing observation fact capacities. Handle `all_creature_types` in `CharacteristicView::has_subtype` (including `not_subtypes` via `is_creature_type`), in `targets_share_creature_type` / `require_shared_creature_type` (an all-types creature shares a type with any creature that has at least one creature type or `all_creature_types`), and in compact observation/token/text projection. Later-timestamp `SetCreatureTypes` / `SetTypes` in Layer 4 clears `all_creature_types`; Layer 6 `LoseAllAbilities` does not.
- Replace the ad-hoc `zones.rs:930` mana-cache check with one `static_affects_mana` helper covering `AttachedSetsCharacteristics`, `AttachedHas`, `SetsLandType`, `GrantsAnyColorManaAbility` and `StaticAbility::Continuous` with `LoseAllAbilities`, `LoseNonManaAbilities`, `GrantActivated`, `SetTypes` or `AddSubtypes`. Check the departing permanent's pre-move effective statics **before** `mint_appearance` (`zones.rs:830` already restores a departing copy or transformed permanent to its printed card before line 929 runs) as well as the entering permanent's post-form statics. Also invalidate both players' mana caches when an Equipment moves (`attach_source_to` in `resolution.rs:2505`) or detaches during SBAs (`sba.rs:272`). Check the opponent's enchanted mana creature (Imprisoned in the Moon already exposes the `StaticContinuous` gap, and a Clone of it leaving exposes the post-`mint_appearance` gap).
- Scan `Modification::SetName` for granted names in `record_vocabulary.rs`. Split `characteristics_set` into `name_set` (from `Modification::SetName`) and `creature_types_set` (from `SetCreatureTypes` and `SetTypes`) so `observation.rs` projects changed `name` and replaced `creature_types` independently. Check the replaced types for Kellan (already defined in `fdn.rs` and currently omitting observed `creature_types`) and Eaten by Piranhas.
- Use one board across grant, copy, attachment movement and reentry. Check that Celestial Armor's protection stays with its original recipient when the Equipment moves. Check Dropkick Bomber's granted flying and damage trigger, Kellan and Mild-Mannered Librarian's incarnation-bound upgrades, and Soulstone Sanctuary and Crawling Barrens' persistent and temporary animation.
- Check transformation Auras across a copy and at death. Witness Protection contributes Citizen LKI. Imprisoned in the Moon contributes noncreature Land LKI.

Use first printings and record each delivered definition and directed check in the [FDN ledger](../.lf/wave/coverage/research/standard-2026-11/sets/fdn.jsonl) where it applies. Match tokens by their full printed characteristics. Scope comes from Task 05 of the [milestone](../.lf/wave/coverage/research/standard-2026-11/milestone.md).

## Batch 4 · Representation, consumers and budget

- Replace the split across `Object.printed`, `Object.front`, `Permanent.manifested` and `ZoneManager.melded_cards` with explicit battlefield form backing (single card, melded pair, token, transforming token), orientation/face-down state and stable manifest public-id alias separate from immutable `CopiableValues`, keeping `Permanent` within its 168-byte ceiling. Stop copying `source.front` in `Object::become_copy_of` (`game_object.rs:644`): a non-transforming copy of a transformed DFC (Clone copying transformed Cecil) keeps only the shown face's copiable values (MV 0 per CR 712.8e, cannot transform per CR 712.9), while a transforming token copy (CR 707.8a, e.g. Ember Island Production on Avatar Kuruk) retains both faces and can transform. Route both `transform_permanent` (`zones.rs:975`) and `FrameCompletion::TurnFaceUp` (`decision.rs:1404`) through one battlefield form-transition helper that enforces CR 712.9/712.10 backing rules and invalidates both players' mana caches (fixing stale mana caches after turning a manifested mana creature face up; `TrackedState::DerefMut` already bumps layer/trigger revisions). Share one `is_face_down()` across `identity.rs`, `action.rs` and `selection_sampling.rs`.
- Route `Decision::ExileLinked` / `cast_permission::exile_linked` (confirmed sole remaining battlefield-capable immediate `move_object` caller; `payment.rs:1071` already uses `EffectFrame.batch`) through `EffectFrame.batch`, using the same event results that Airbending Lesson consumes. The player performing the exile chooses relative order, even when the ability controller receives the cast grants. In each ordering fork, keep Ragnarok assembled and its grants unissued while the order is pending. After the ordering Command, grant Azula's controller access to both exact exile arrivals before offering the next graveyard selection. Reject the spent Command without advancing the frame or adding grants. Keep the Hama and ordinary Azula checks as single-arrival controls.
- Finish the consumer checks and measure the engine budget below.

## Open checks on finished work

- Lost Days: the order choice should be visible only to its owner, and the game should replay from its record.
- Horde, Driver and Fire Crystal still need viewer projections and replay of their recorded Commands from a recorded root. Their native fork and restore checks already pass.
- The named Coil setup has no seed-to-position replay file.
- Manifest sampling: there is no pending private selection in the fixture. Constrained sampling, paired seeded searches and comparative viewer trace exports remain open.
- Batch 1's constructed fixtures still need viewer projections, text, tokens and Command replay from a recorded root.
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
