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
- Compress `9e297f0c`: waiting triggers read the captured face (`pending_trigger_definition`/`pending_trigger_face`), and target legality shares `fits_trigger_clause`. The focused nextest run passed 42 cases. The 13 Python copy-text, Academy trigger-order and pinned-game checks passed. The merge `039b275b` brought only Loopflow workflow files.
- Batch 1 exact departures and linked returns (`e6f6ef27`, `47dec0aa`, `2955a3d8`). Effective `ObjectLki` uses the same matcher as live characteristics. `WithSource` exile links survive departure, and delayed returns bind to exact arrivals. The card list and directed evidence are in [the behavior review](behavior-review.md). The implementation run passed 14 departure cases, 142 focused card and observability cases, and the canonical replay and LKI/exile filters.

## Batch 2 · Graveyard choices, costs and simultaneous movement

### A. Spell entries · built

`SpellEntry` owns kicker, X and optional payment through the proposed move and on the resulting permanent. A cast for zero carries an empty receipt. A permanent spell copy keeps kicker and X with no payment. A blink or battlefield copy has no spell entry. This supersedes the reviewed `CastReceipt` model because casting choices can survive without a cast. `Arc<SpellEntry>` keeps the permanent at its 168-byte ceiling, down from the 184-byte inline attempt. The digest projection retains `was_cast`, `cast_x` and `cast_spent`. The copied spell's cast flag is corrected. See `managym/state/src/permanent.rs` and `managym/src/flow/decision.rs` at `d61564193a`.

The entry answers X counters and Familiar's cast-only condition before commitment. The entering permanent's own trigger captures kicker, X and spent mana. Its intervening-if recheck can still read kicker after the source leaves. Healer (`fdn.rs`) and Nullpriest (`znr.rs`) share `check_kicked_return` in `managym/cards/tests/cards.rs`. It forks kicker and target Commands, rejects stale submissions, exercises Cleric-only mana, removes the source in response and blinks it in a sibling branch. The chooser sees the captured kicker during placement and both viewers see it after removal.

Eleven directed debug checks passed, including Familiar's cast-only replacement, Toph's spend receipt and a copied permanent spell's X, kicker and absent payment. Compression at `12c51346a1` and `7cd57bfb32` passed 81 focused Rust tests and 14 trigger-order Academy and semantic-text tests with the rebuilt CPython 3.13 extension. Both trigger orders at seed 7 produced identical state digests and both viewers' observation JSON before and after compression. The departure fixture distinguishes Gray Ogre's printing from its Mountain face at departure and its later copy. The merge of `b9b09bc6d9` added observation extents and changed policy preparation without edits in the rules flow or state crates. Those checks predate the merge and did not include the pinned pilot.

### B. Source-relative target legality · built at `7e1fa0b92e`

Trigger placement, mode availability, retargeting and resolution share `target_fits_clause`. It checks the base target spec, excludes the source for “another” and reads the power bound from the exact live or departed source. `departure_arrivals` identifies each exact graveyard arrival until its next move. Distinctness and the announced-X budget apply together when choosing or replacing a target.

Fiendish Panda's life-gain condition reads committed life increases. Simultaneous lifelink damage gains life once per source. With two attacking Sires, one blocked by two Ogres, Panda receives two counters for fourteen life. A Clone that dies as a 4-power Panda can return Driver. The physical Clone meets the ordinary card predicate, so “another” alone excludes its graveyard arrival. Move that card through the hand and back to the graveyard in a sibling retargeting branch and it becomes eligible. Driver's trigger on the same board can return only Llanowar Elves. A constructed modal trigger on Hill Giant offers the 3-mana Ogre and omits the mode requiring a 4-mana card. Savor in response leaves the source alive at 1 power and invalidates the Ogre at resolution.

Twenty-one focused debug checks passed, including Panda, Driver, Kykar, Jailer, Wolfbat, Scavengers, Sozin's X budget, station, Mayhem, trigger tokens and Scryfall shell conformance. The rebuilt CPython 3.13 extension admitted Panda and Driver. The Python catalog check found the life-gain condition in Panda's first ability program and the death condition and power bound in its second. The initial probe looked in the characteristics span before following the catalog's program rows. Full suites and the pinned pilot remain at gate.

### C. Simultaneous public-zone movement · next

Add `ObjectSelector::ZoneMatching { zone, owner: Actor, predicate: CardPredicate }` beside `Each` in `Game::select`, selecting physical cards only (`zone_cards` / `card_of`, never transient tokens) in deterministic player order (`Actor::AnyPlayer` → `[PlayerId(0), PlayerId(1)]`) and joining `Effect::SelectCards`'s `cards_only` guard. Feed its matches to `Effect::Move` as one `EventBatch`. Raise the Past selects the caster's creatures with mana value at most two. Rise of the Dark Realms selects all players' creatures and puts them under the caster's control (`under: None`). Preserve `Zone(_)` and `Each` serialization. Cover the new selector in `ability_description.rs`, learning tokens and `observability.rs`.

Delete the duplicate `Game::permanent_mana_value` (`resolution.rs:2410`) and call `i32::from(self.object_mana_value(permanent.object))` in `ObjectSelector::Each { max_mana_value }` so transformed DFCs (CR 712.8e) and melded permanents (CR 712.8g) do not read as mana value 0.

Use Day of Black Sun at X=0 to expose the duplicate. A manifested creature has mana value zero and should be destroyed. Transformed Cecil and a physical Ragnarok should keep their abilities and survive. Extend the existing Day of Black Sun check or use the constructed form setup alongside it.

Use the two-graveyard setup for both spells. Check that entering Momo, Friendly Flier sees an entering Spectral Sailor. Use a Sailor owned by the opponent for the Rise fork to check ownership and control. Driver stays a death-trigger witness in B.

### Finish and review the cohort

- Heartfire Immolator gets prowess and a self-sacrifice cost. Pump it, sacrifice it, return it before resolution and check damage from the sacrificed appearance's power.
- Ordeal of Nylea follows the enchanted creature's attack, counter threshold, sacrifice and optional basic-land search through native triggers. Feldon's Cane (`atq.rs`) exiles itself as a cost, then shuffles.
- Pair Undying Malice or Valkyrie's Call with simultaneous deaths and a blink in response, as carried from Batch 1.
- Review the whole batch together. Each card composes from shared rules and has a directed debug case in its first-printing set's test file. Check the `SpellEntry` owner, shared clause predicate and single `Effect::Move` path.

Defer general delayed-effect value freezing until a second card needs it. `CreateDelayedTrigger` freezes `DeathCountersMinus` for Nine-Lives Familiar. The post-Batch 1 concept review in `scratch/concept-review.html` records the proposed generalization if another consumer appears.

**Abyssal Harvester depends on MTG-418.** It needs the native query for exact graveyard arrivals this turn from any zone. `TurnState::graveyard_from_battlefield` records only battlefield departures. MTG-418 has no published PR or release for the query. Once it reaches `main`, run `lf sync` and locate the query with `rg -n 'graveyard_from_battlefield|GraveyardThisTurn|graveyard.*this turn' managym/src/flow managym/state/src`. Offer Harvester a milled creature, a dead creature and a card that left and reentered. Follow the selected exile appearance through the Nightmare copy and the exile of the other Nightmare tokens. Continue the rest of this batch independently.

## Batch 3 · Effective characteristics and grants

- Add `Modification::SetName(String)` so Witness Protection, Eaten by Piranhas and Imprisoned in the Moon (`emn.rs`) use `StaticAbility::Continuous`. Lower `StaticAbility::AttachedSetsCharacteristics` through `Modification::SetName` in `layers.rs` and delete `Operation::Name`. Preserve Honest Work's TLA learning tokens.
- Replace the ad-hoc `zones.rs:930` mana-cache check with one `static_affects_mana` helper covering `AttachedSetsCharacteristics`, `AttachedHas`, `SetsLandType`, `GrantsAnyColorManaAbility` and `StaticAbility::Continuous` with `LoseAllAbilities`, `LoseNonManaAbilities`, `GrantActivated`, `SetTypes` or `AddSubtypes`. Also invalidate both players' mana caches when an Equipment moves (`attach_source_to` in `resolution.rs:2505`) or detaches during SBAs (`sba.rs:272`). Check the opponent's enchanted mana creature.
- Scan `Modification::SetName` for granted names in `record_vocabulary.rs`. Split `characteristics_set` into `name_set` (from `Modification::SetName`) and `creature_types_set` (from `SetCreatureTypes` and `SetTypes`) so `observation.rs` projects changed `name` and replaced `creature_types` independently. Check the replaced types for Kellan and Eaten by Piranhas.
- Use one board across grant, copy, attachment movement and reentry. Check that Celestial Armor's protection stays with its original recipient when the Equipment moves. Check Dropkick Bomber's granted flying and damage trigger, Kellan and Mild-Mannered Librarian's incarnation-bound upgrades, and Soulstone Sanctuary and Crawling Barrens' persistent and temporary animation.
- Check transformation Auras across a copy and at death. Witness Protection contributes Citizen LKI. Imprisoned in the Moon contributes noncreature Land LKI.

Use first printings and record each delivered definition and directed check in the [FDN ledger](../.lf/wave/coverage/research/standard-2026-11/sets/fdn.jsonl) where it applies. Match tokens by their full printed characteristics. Scope comes from Task 05 of the [milestone](../.lf/wave/coverage/research/standard-2026-11/milestone.md).

## Batch 4 · Representation, consumers and budget

- Replace the split across `Object.printed`, `Object.front`, `Permanent.manifested` and `ZoneManager.melded_cards` with explicit battlefield form backing (physical card(s)/token, orientation/face-down state and stable manifest public-id alias) separate from immutable `CopiableValues`, keeping `Permanent` within its 168-byte ceiling. Route both `transform_permanent` (`zones.rs:975`) and `FrameCompletion::TurnFaceUp` (`decision.rs:1404`) through one battlefield form-transition helper that validates backing form (CR 712.9 non-DFC Clone cannot transform; CR 712.10 face-down permanent cannot transform; CR 707.8a transforming token copy can) and invalidates both players' mana caches and layer/trigger revisions (fixing stale mana caches after turning a manifested mana creature face up). Share one `is_face_down()` across `identity.rs`, `action.rs` and `selection_sampling.rs`.
- Audit immediate `move_object` callers by their legal source zones. Route Azula's battlefield selection and the shared `Decision::ExileLinked` continuation through `EffectFrame.batch`. The player performing the exile chooses relative order, even when the ability controller receives the cast grants. In each ordering fork, keep Ragnarok assembled and its grants unissued while the order is pending. After the ordering Command, grant Azula's controller access to both exact exile arrivals before offering the next graveyard selection. Reject the spent Command without advancing the frame or adding grants. Keep the Hama and ordinary Azula checks as single-arrival controls. Use the same event results that Airbending Lesson consumes and remove the linked-exile path that grants against only the first arrival.
- Finish the consumer checks and measure the engine budget below.

## Decomposition review at `284c69636a`

**Iterate.** Batch 2A meets the shared-owner criterion. The phase still has rules defects and planned deletions in 2B–4.

- Sun-Blessed Healer and Nullpriest of Oblivion use the same `Condition::Kicked` and return effect with different graveyard predicates. `FrameCompletion::Spell` gives the move one `SpellEntry`. The entering trigger captures its kicker, X and spend, and the stacked occurrence retains them through departure. `check_kicked_return` exercises both consumers through payment, target Commands, removal, blink and viewer projection. Toph reads the payment through the entering trigger. Nine-Lives Familiar reads its presence during the entry replacement. The CR 400 permanent-spell copy case distinguishes kicker and X from actual payment, and a subsequent return clears the entry. The old `ZoneEntry.x` field, parallel commit argument and permanent `was_cast`/`cast_x`/`cast_spent` fields are gone. Historical digest keys are derived by the serializer. The existing directed tests now assert against the shared entry, with no separate card-specific runtime path.
- Clone Legion copying a manifest and the departed Clone's Horde trigger both reach `copy_source` and `TokenSource::Copy`. The manifest fixture checks two generations of public 2/2 copies. The Horde fixture compares the live source and the exact departed source after the same physical card returns with a different form. This supports the shared copy payload. Physical backing still spans `Object.front`, `Permanent.manifested` and `ZoneManager.melded_cards`. Transform and turn-up still write through separate paths, and turn-up leaves mana caches intact. Batch 4's deletion and mana-creature turn-up check remain required.
- Airbending Lesson already retains the batch, presents both meld orders and binds both cast permissions before the draw. Azula's `SelectCard` response instead calls `exile_linked` and immediately resumes the frame. The immediate batch supplies no actor to `meld_order_choice`, which suppresses the exile ordering offer. The helper then reduces the results to the first arrival, leaving Azula without permission for the second card. The current Azula fixture uses Gray Ogre and cannot catch either failure. The Batch 4 acceptance check above now names this interaction.
- Pending trigger placement, mode availability, retargeting and resolution still disagree on source-relative legality. `legal_trigger_modes` omits the power bound, `resolution_targets` omits the bound and `another`, and the placement and retarget predicates compare `another` only against a live permanent. Sandbender Scavengers and Kykar exercise distinct existing parts of the contract. Keep their checks and complete the exact graveyard-source and live-source response cases in 2B. Day of Black Sun still reads `Each` through the duplicate mana-value helper. The characteristic and mana-cache deletion cuts in Batch 3 are also open.

Evidence reused from `7cd57bfb32`: 81 focused Rust cases and 14 Academy and semantic-text cases, including the retained spell-entry checks. The rules flow, state, cards and native fixture files are unchanged between that revision and the reviewed head. This review ran no new suites. Rerun the focused target, form and linked-exile cases against the corrected contract before advancing. Replace any dependent performance results after semantic changes. Gate owns the affected full suites and the pinned pilot.

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
