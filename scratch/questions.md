No scratch design existed at the checkout base. The working plan uses the accepted Task 02 milestone and exact set-ledger selectors. Monica Rambeau // Photon, Living Light is the chosen modal casting probe. Its broader battlefield-form work stays with the forms task.

The first implementation slice uses Zul Ashur and a pair of Hungry Ghouls. This isolates exact-object permission and ordinary graveyard casting before adding Muldrotha's quota choices to the same position.

The compression pass covers the implemented Zul Ashur slice and its existing casting consumers. The later overlap fixtures cover multiple permission choices.

Concept review chooses direct native offers for consequential permission and profile choices. The alternative adds a permission-selection decision to search. Revisit against measured action width and search work.

Exact-object grants receive local keys when granted. Cleanup can reorder the surviving records without changing the meaning of a selected offer. An unambiguous priority action omits the route from its Command.

Singleton grants keep their existing wire shape for retained captures. The public Python list covers any number of grants. The cold-payment fixture uses a normally played Gray Ogre because a replay file needs a recoverable action history.

The next slice starts Smaug // Spew Flame in hand to exercise a second printed program before adding Bilbo’s competing exile replacement.

Concept review moves the Bilbo replacement fork ahead of batch authoring. Two successful casts ending in exile give a stronger test of grant ownership than comparing resolution with a counter alone.

The pending hand fixture uses twelve Mountains to cover both Spew Flame and the later Smaug cast in one turn. An opposing Counterspell branches from the same announced Adventure.

Use separate constructed and ordinary-play Smaug roots. The ordinary-play root supplies the recoverable history for cold replay.

The ordinary-play replay witness uses two Ogre targets and Lightning Bolt in each deck so the target choice and paid spell are both observable with native trivial skipping enabled.

Use Twinmaw's top-library versus hand offers as the first Omen witness. That comparison can expose a profile-filtering error before the shuffle implementation exists.

Use Pyromancer’s Goggles for the copied Omen witness. The copy can kill the sole target, so the physical card’s graveyard destination isolates the copy’s shuffle.

Use Scavenger Regent as the second Omen because its X cost exercises announcement and free casting through the same selected profile.

Keep the existing v11 stack mana-value column's printed-cost reading for saved models. Expose the native on-stack mana value separately and use Exude Toxin at two X values to distinguish the readings.

Exude Toxin’s cold replay uses Swamps against Mountains and Lightning Bolt. The opponent can respond after payment. The public stack record retains the number of printed X symbols so a captured observation can reconstruct the printed cost and effective value together.
Use Twinmaw and a second nonland card for the Etali fixture so the same suspended parent exercises profile, target and cast-order choices.

Use two prepared sources in the first priority fixture. A single Seed Suture cannot expose grouping by program name. Preserve physical-card Command bindings while carrying the exact offered object into announcement.

Use Regent’s existing Omen as the prepared-inset control. Changing only the inset kind isolates the hand and Cascade eligibility rule.

Use an unrelated exiled spell copy as the negative cleanup control in the two-source fixture.

For the two-copy fixture, both branches cast Counterspell through a native Command and then cast the surviving Seed Suture targeting the Angel. This checks that the surviving offer still executes after the counter.

The transfer review selects a fixed-player source-conditioned grant as the comparison. Transfer holds both object identities constant while distinguishing a change of caster from a loss of access.

The ordinary-play transfer replay uses Agent of Treachery to take the Angel and Spectral Sailor to keep a response available after payment.

Project the caster rule on visible permissions so a policy can distinguish the effects of a control change before choosing it. Use the existing transfer fixture as the paired control.

Use a castable Lightning Bolt to issue a public target address for the Angel before it leaves. Use a controlled non-Pilot creature as the positive target for All Aboard.

Use the pending Waypoint activation as the first fork for repeat preparation. Compare exact links and copy movements before adding a stored reverse index.

Use an ordinary-play Voxmancer alongside the constructed Angel for the Waypoint replay. Its upkeep can renew its spell between activations.

Use blink for the Emeritus source-loss branch. Resolving the original trigger with the same card back in play distinguishes source appearance from physical card identity.

Group the outstanding rows by preparation, printed spell programs, and zone permissions. Each group shares the native fixtures listed in the plan.

The six history-dependent preparation rows wait on MTG-418 instead of adding turn-history counters here. The brief says to defer such cards. Leech Collector counts as one: "first life gain this turn" differs from a once-each-turn trigger when life was gained before it entered.

Concept review (whole casting/permission area) makes four reversible choices recorded in `scratch/concept-review.html` and `scratch/plan.md`:
- Replace branch-added `action::{ADVENTURE, OMEN}` flags with one `action::SPELL_PROFILE` category (size 4: Front, Adventure, Omen, BackFace); zero pinned-pilot break because columns 25-29 are new on this branch.
- Model Omniscience as `PlayRoute::FreeFromHand` (deduplicated across copies) rather than overloading `kicked: bool`, so free casts still allow Kicker/Bargain (CR 118.9d) and compose with Harbinger's `AdditionalCost::FlashSurcharge`.
- Store exiled-card counters (Tinybones stash counters) in sparse `GameState.object_counters: BTreeMap<ObjectId, Counters>` instead of adding a `Counters` field to every `Object`, keeping normal object size/clone cost unchanged and preserving CR 400.7 staleness.
- Reuse `CardDefinition.back_face` with `modal_back: bool` and `SpellProfile::BackFace` (resolving with `ZoneEntry { transformed: true }`) rather than adding a fourth `InsetSpell::ModalBack` variant.

Preserve the compact priority projection's existing front-face AnyTarget ordering. Recorded Command preconditions use its positional candidate addresses. The complete compound API binds targets from its own action, including the selected face and payment route.

Use the original M11, M13 and Time Spiral printing keys for the Foundations reprints. Their game fixtures stay with the Foundations permission cohort.

Keep the prior opening authority and parity receipts by recording the casting catalog in new files. Their definition ordinals come from the catalog at `d12eb496b3`. Comparing the retained and current Commands by printed name also resolves the source of each CARD_REVEALED event.

Use Heartwood Crafter and Codie for the next casting-provenance slice. An exile cast of a linked prepared spell exercises both the spending restriction and Codie's copy. Use the existing Burst Lightning fixture for manual kicker, with one red unit spent on the generic cost.

Post-build concept review (`a5c3c68beb` vs `82c5d1d1b7`) records five reversible choices in `scratch/concept-review.html` and `scratch/plan.md`:
- Unify priority and resolution `CastChoices` construction behind one shared builder where `resolving: true` for ALL `CastDuringResolution` casts (free and paid) and `free: true` strictly means "without paying its mana cost" (fixing Harbinger of the Tides under Yue/Etali/Cascade and giving `from_prepared` one owner).
- Keep single-route `Action.route = None` on the wire for retained Command compatibility, and add `Game::resolved_play_route` so solitary `GraveyardType` (Muldrotha) plays always project `quota_type`, Muldrotha focus, and route description.
- Decouple `PlayGrant.face_down` and `PlayGrant.condition: Option<Condition>` from `PlayGrant.foretold`, replacing the 5 visibility calls to `is_foretold` with `is_face_down_in_exile` / `can_view_face_down_exile` while keeping `is_foretold` strictly for Foretell cost and determinization (unblocking Gandalf's Flameshape).
- Publish Tinybones's static exile-counter permission on `LayerBoard.play_permissions.exile_counter_plays` as `PlayRoute::ExileCounter(CounterKind)`, backed by sparse `GameState.object_counters`.
- Keep `Action::SelectCard` for front-face `CastDuringResolution` offers and `Action::CastSpell` for non-front profiles to preserve retained replay captures.

## 2026-11-23 implement pass

- Codie's type line is "Artifact Creature — Book Construct". `Book` is filed as an artifact subtype in `state/src/subtype.rs`. That is an assumption: if the CR lists Book as a creature type, move it (one line).
- Stack spell rows (`SpellData`) carry neither `kicked`, `flashback` nor the new `from_prepared`. Codie's trigger reads `from_prepared`, so the policy cannot see whether a spell on the stack is prepared. That gap predates this pass; it is left for a shared observation change rather than one column.

## 2026-11-23 concept review (`71148883e2` vs `88ce96bd6b`)

Four reversible choices recorded in `scratch/concept-review.html` and `scratch/plan.md`:
- Route non-legacy pooled mana restrictions (starting with Heartwood Crafter's `ManaRestriction::ExceptSpellsFromHand`) through existing `PlayerData.pooled_mana_classes` -> `FactKind::EffectProgram(POOLED_MANA)` instead of appending 6 hardcoded `turn_state` columns and a bespoke `RestrictedManaData` boolean. Keeps the 4 legacy column restrictions, `test_restricted_mana_input.py`, and the pinned pilot untouched while following the open-vocabulary fact rule.
- Drive `selection_sampling.rs::unseen_objects` from `is_face_down_in_exile && !can_view_face_down_exile(obj, viewer)` and pair `fill_foretold_slots` with unrestricted `fill_face_down_exile_slots` in `search.rs` and `possible_worlds.rs`, fixing an open-decklist privacy leak for Gandalf's Flameshape.
- Consolidate all static play permissions (`graveyard_type_sources`, `exile_counter_plays`, `extra_land_plays`) onto `LayerBoard.play_permissions`, deduplicate `exile_playable_objects`, add `PlayRoute::PrintedGraveyard` for Quilled Greatwurm (so counter removal never fires on Muldrotha/Zul routes), and check `cast.spell(spell).types.is_land()` in `cast_spell_action`.
- Project `SpellOnStack.{from_prepared, flashback}` to `StackObjectData`, v11 `stack::{FROM_PREPARED, FLASHBACK}`, `text_observation.py`, and `experience_projection.py`, and replace the growing positional-tuple ladder in `permission_facts` with a named extension record for post-TLA grant terms (preserving Hama's legacy token sequence).

Slice 1 choices: the stack's `flashback` flag projects `SpellOnStack.flashback` (stack-exit exile from a flashback route); resolution-cast `exile_instead_of_graveyard` stays unprojected. The pooled-mana text keeps its existing "only for …" wording, so Heartwood's class reads "only for anything but spells from hand". Sanar's activation gate is a general `ActivatedAbilityDefinition.condition` checked with the other activation terms, not a Sanar-specific flag.

## 2026-11-23 post-Slice-1 concept review (`9b48f4abe3` vs `88ce96bd6b`)

Four reversible choices recorded in `scratch/concept-review.html` and `scratch/plan.md`:
- Record `RecordedCast { object: ObjectId, profile: SpellProfile }` in `TurnState.spells_cast` (instead of bare `ObjectId`) and resolve the cast profile in `recorded_cast_matches_predicate` once departed — fixes a live rules bug where casting and resolving an Adventure Sorcery (Smaug's Spew Flame) or Omen Instant (Twinmaw's Charring Bite) reverts `retained_base` to the Creature front face and locks Sanar's Treasure activation back out (and breaks `first_on_own_turn`).
- Resample hidden face-down exile (`is_face_down_in_exile && !can_view_face_down_exile(obj, viewer)`) for **both** seats (including own seat in `shuffle_known_library` and `shares_information` opponent), add `face_down_exile: u32` (omitted when 0) to `HiddenHandConstraints` / `PossibleWorldSpace` / `selection_sampling`, distinguish foretold vs. generic face-down exile in `text_observation.py:941`, and carry `prepared` + `condition` on `CastPermissionData` / `permission_facts` / `inspect.ts` — closes the turn-3 Flameshape owner-search leak and possible-world slot mismatch.
- Render concise `prepared` / `flashback` badges on `StackEntryView` in Academy's `Rail.svelte`, and drop redundant `"only "` before `"anything but ..."` exclusions in `text_observation.py:865`, `mana.ts:11`, and `Plate.svelte:41` (`"for anything but spells from hand"`).
- Retain `(source_id, view)` in `effective_spell_cost` and add `EffectValue::AttachedCreaturePower` reading `self.attached_creature(source_id)` -> `layer_view(c).power.max(0)` for Glamdring; explicitly allow all profiles in `permits_spell` for `PlayRoute::ExileCounter(_)` and keep spell//land MDFC back-face land plays open under `HandOnly`.

## 2026-11-24 implement pass (Slice 2 items 1–3, 5–8; Slice 3 item 2)

Reversible choices taken:
- A discard is the engine's existing hand → graveyard move (`KeywordAction::Discard` reads that `CardMoved`, as `EventBinding::Moved` already did). A discard a replacement redirects out of the graveyard would not trigger "whenever an opponent discards"; no current card does that.
- Tinybones's permission is one variant, `StaticAbility::PlayUnownedExiledWithCounter(CounterKind)`, with "during your turn", "cards you don't own" and "mana of any type" fixed to its text rather than three fields no second card uses yet.
- Counters off the battlefield are carried into exile only (`ZoneEntry::for_zone`). Graveyard or library counters would be the same map when a card needs them.
- `SpellCostReduction.generic` became an `EffectValue` with a serializer that writes a fixed amount as the bare number, so the five existing reductions keep their learning tokens and the pinned pilot is unaffected.
- Claim Territory with only one Forest found puts it onto the battlefield tapped (the inner selection requires one). Check the Oracle ruling if this matters.
- `lea::clone_copy_survives_suspended_entry_and_keeps_its_printed_base` pins a state hash that moves whenever a definition sorting before "Wolf Cove Villager" is added (four refreshes in this pass alone). Pinning a hash that excludes catalog ordinals, or asserting the copied values directly, would stop that churn.

## 2026-11-24 post-Slice-2 concept review (`8e7306cbc2` vs `88ce96bd6b`)

Four reversible choices recorded in `scratch/concept-review.html` and `scratch/plan.md`:
- Replace sequential `fill_foretold_slots` + `fill_face_down_exile_slots` with one atomic `refill_hidden_exile_slots(owner, viewer, pool)` (snapshot all hidden slots up front, assign pairwise-distinct replacements foretold-first, update `zones.exile` and `grants` in one simultaneous map), gate `knowledge.rs:180`'s `shuffle_known_library` fast path on `hidden_face_down_exile.is_empty()`, fix `experience_projection.py:481` to check `card.face_down` (not `permission.foretold`), and validate `hidden_foretold_count` + `hidden_face_down_exile_count` in both `HiddenHandConstraints::for_viewer` and `PossibleWorldSpace::validate_source` (plus `selection_sampling.rs`). Prevents duplicate-exile/merged-grant corruption when Foretell and Flameshape coexist and closes three privacy/UI leaks.
- Make `effective_spell_cost` (`play.rs:544-584`) the single CR 601.2h owner of `any_mana` across `PlayRoute::ExileCounter(_)`, `PlayGrant.any_mana`, and `StaticAbility::SpendAnyManaToCast`, reading raw `grant_base_cost` in `casting_mana_cost` while keeping `grant_cast_cost`'s existing observation wire shape — fixes live Azula + Glamdring (generic reductions eating colored pips) and Azula + colored Kicker/Replicate bugs without breaking retained TLA captures.
- Add `#[serde(skip_serializing_if = "SpellProfile::is_front")] profile: SpellProfile` to `Action::PlayLand`, validate the selected profile in `play_land` (entering `transformed: profile == SpellProfile::BackFace`), and guard `PlayRoute::FreeFromHand` against lands in `permits_spell` — fixes the live offer-vs-execute contradiction on `modal_back` land faces with zero wire change for front-face lands.
- Complete `academy/ui/src/lib/board/inspect.ts::permissionTerms` to render all `CastPermissionData` terms (mirroring `text_observation.py::_cast_permission`) alongside `prepared` and `condition`, and replace `permission_facts`'s 4-branch tuple ladder with a named extension record for post-TLA fields while preserving Hama's legacy token sequence.

## 2026-11-24 implement pass (next slice items 1, 3–5)

Reversible choices taken:
- Flameshape says "Look at the top two cards … exile them face down", so its caster has seen them. Following the concept review, the engine hides them from the caster too while no Wizard is controlled (CR 406.3: face-down exile is examined only as instructions allow; the play permission is the only instruction). If the caster should keep knowing them, that is a single-card knowledge fact, not visibility.
- `CastPermissionData.condition` is the condition's rules text (`condition_text`, e.g. "you control a Wizard permanent"), because `Condition` is not deserializable and observations round-trip through JSON. The policy reads it as a Text token; `record_vocabulary::play_grant_conditions` registers every condition a printed spell (front or inset, top-level effects) puts on its grants. A condition nested deeper in a future card would need that walker extended; the vocabulary guard would fail first.
- `POSSIBLE_WORLD_SPACE_VERSION` stays 4. `face_down_exile` is omitted from the identity and projection when zero, so every existing space keeps its identity; a space with hidden Flameshape cards is new. Its weight multiplies every world by the common factor `P(r - F, D)`.
- Academy has no face-down badge in any Svelte view; `Board.svelte` shows `permissionTerms`, which now renders every term. The planned `Inspect.svelte` badge has no existing place to go.
- `PlayLand.profile` (next slice item 2) is not built: no spell//land card exists, so the path cannot run. Building it needs a real spell//land MDFC or a rules fixture (`src/cards/fixtures.rs`), either of which shifts catalog ids again.
- `tests/academy/test_play_modes.py::test_curated_combat_tape_is_identical_live_trace_and_inspector_fixture` fails here: the new game stops at the opening Keep/Mulligan decision and `scenario_refresh` refuses it. CI does not run `test_play_modes.py`; not investigated further.

## 2026-11-24 post-Slice-3 concept review (`307465b649` vs `88ce96bd6b`)

Four reversible choices recorded in `scratch/concept-review.html` and `scratch/plan.md`:
- Give `Action::PlayLand` an omitted-when-Front `profile: SpellProfile`, remove front-face `land` pre-filters in `play_routes` (using `!land || modal_back` for `FreeFromHand`/`LibraryTop` so plain lands stay on lone `Hand`, and dropping `|| (land && kind != Land)` in Muldrotha's `GraveyardType` loop), validate `permits_spell` inside `play_land`, and prove both Spell//Land and Land//Creature MDFCs via rules fixture(s) in `managym/src/cards/fixtures.rs` while replacing the churning Clone state-hash pin in `lea.rs`.
- Include `...(player.inspected ?? [])` in `Board.svelte`'s `cards` and `textOf` (matching `view.ts:165`) and resolve `CardState` from `item.placed.anchor.id` on Academy play-tray hover (`previewCard` with `previewName` fallback), so hovering playable exiled/graveyard/library/inspected cards actually displays `permissionTerms` and `counterTerms`.
- Pass `library_top_departure: true` from `exile_top_cards_to_play` through `exile_with_play_grant` (preserving shifted `known_library` positions under Redcap/Flameshape, matching `draw_cards`), read `casting_zone` in `SpellsFromOutsideHandCostLess`, and walk `objects(&json)` in `record_vocabulary::play_grant_conditions`.
- Build Quilled Greatwurm (`PlayRoute::PrintedGraveyard` + distributed `CostComponent::RemoveCounters` + creature combat-damage amount) and finish the remaining Tinybones/Gandalf/MDFC interaction evidence inside MTG-419, and formally split the 12 non-casting shared-rules rows into separate follow-up Tasks at their named boundaries (graveyard-zone triggers, batch-collapsed triggers, surveil/mill-result queries & graveyard exile costs, APNAP wheel & delayed mana triggers, and entry-chosen creature type / control exchange / Amass).

## 2026-11-24 implement pass (Slice 4)

Reversible choices taken:
- `Action::PlayLand.profile` serializes under the key `adventure`, the key `CastSpell` and `SpellOnStack` already use for the profile, so the two play actions carry one wire name. A rename to `profile` everywhere would be one coordinated wire change.
- The MDFC rules fixtures are invented (`Tidal Errand // Tidal Haven`, `Kelp Hollow // Kelp Warden`). Their lands enter untapped and have no other text, unlike most printed MDFC lands.
- Muldrotha now offers every permanent type of a land (an artifact land gets both its Artifact and its Land route), following its ruling that any permanent-type permission may be used.
  - Behavior review (2026-11-24) reverses this. The cited ruling does not exist. CR 305.9 lets a land of another type be played only as a land, and Muldrotha's other permissions are for casting. Slice 5 item 1 restores Land-only routes for land faces (D4).
- `Condition` and `CounterKind` derive `Deserialize` so the vocabulary walk reads conditions from definition JSON. `CastPermissionData.condition` stays rules text.
- The Scryfall fixture's Heartwood Crafter and Sanar rows had `"colors": null` (double-faced/prepare layout), which broke `conformance_tests` on this branch. They now hold the front face's colors, written by hand: `refresh_card_fixture.py` probably writes null for these layouts and would need the same fix.


## 2026-11-24 post-Slice-4 concept review (`8e89bd5646` vs `88ce96bd6b`)

Four reversible choices recorded in `scratch/concept-review.html` and `scratch/plan.md`:
- Add `play_lands: bool` (default `false` = cast spells only, omitted when `false`) to `PlayGrant`, `CastPermissionData`, and `PlayGrantExtension`, set `true` only in `exile_top_cards_to_play` (Redcap, Flameshape, Loot, Zuko, Roku) and `fdn::granted_foreign_land_enters_under_its_players_control`, and unify `PlayRoute::Grant` in `permits_spell` to require `!land || grant.play_lands` — closes the live Airbend-on-Earthbent-land and cast-only-grant-on-MDFC-land-face bugs and aligns `"may play"` vs. `"may cast"` across LLM text and Academy without disturbing retained TLA cast-only captures or the pinned pilot.
- Add `ActionOption.play_route: Option<u8>` (omitted from JSON when `None`) and append categorical column 29 `action::PLAY_ROUTE` (`Category { size: 10 }`) to `action::COLUMNS`, separating overlapping same-MV routes (`ExileCounter` vs. `Grant` on Tinybones today; `PrintedGraveyard` vs. `Grant` on Quilled Greatwurm in Slice 5; `GrantedFlashback` vs. `Grant`) in the v11 policy tensor while keeping saved prefixes and `test_pinned_pilot` intact.
- Scope route disambiguation to `(object, profile)` via `permitted_play_routes` (before affordability) and pass `profile` into `selected_play_route` / `resolved_play_route` so lone-route MDFC land plays stay `route: None` under Omniscience, and add `pending_choice.is_none()` / `priority.holder == player` guards to `play_land` matching `cast_spell_action`.
- Align Slice 5's Quilled Greatwurm design to its verified Oracle text before coding: `CostComponent::RemoveCounters { remaining, kind: Option<CounterKind>, predicate }` with `kind: None` (any counter kind, prompting a mode pick when a chosen creature holds multiple counter kinds) and a per-creature aggregated `YourTurn` combat-damage trigger (CR 510.2: Trample splits and multi-blocks trigger once with the creature's total combat damage).

## 2026-11-24 implement pass (Slice 5 items 1–3)

- The LLM text changed wording for cast-only grants ("may be cast …") without bumping `representation_version` (`manabot-text-10`), as earlier slices changed permission terms. Bump it if retained captures are compared by text.
- A cast-only grant on a land shows no cost term in LLM text (the land has no casting cost), only "may be cast from exile by …" and its other terms.
- `play_route` codes number the routes 1..8 in `PlayRoute` declaration order; `Category { size: 10 }` leaves one spare code after `PrintedGraveyard` (9).
- D5 is tested by refusal in `permits_spell` (no catalog library-top predicate matches a land face) plus the existing Vizier/Kelp Hollow assertion; no fixture with a land-matching library predicate was added.


## 2026-11-24 post-Slice-5 (items 1–3) concept review (`d46d360241` vs `88ce96bd6b`)

Four reversible choices recorded in `scratch/concept-review.html` and `scratch/plan.md`:
- Guard `Game::resolved_play_route` (`play.rs:434`) on `self.state.suspended_decision.is_some()` returning `None` (and guard `action.rs:54`'s `preparation_source` arm on `suspended_decision.is_none()`), keeping non-front `CastDuringResolution` offers (`Action::CastSpell { route: None, profile }`) clean of overlapping priority route codes, focus, and descriptions rather than introducing a separate wire action variant.
- Compute `CastPermissionData.cost` (`observation.rs:1758`) via `Game::grant_card_cast_cost(grant, object)` — using the front face unless it is a land with a castable `modal_back`, in which case use the back face — so Land//Spell MDFCs (`Kelp Hollow // Kelp Warden`) project their Airbend/Flameshape cast costs across policy tokens, LLM text, and Academy while pure lands remain `None`.
- Update Academy's announcing-spell stack card lookup (`academy/ui/src/lib/board/view.ts:257`) from `frame.projection.agent.hand` to `inspectableCards([agent, opponent])` so spells announced from graveyard, exile, or library top keep their full card projection in the stack rail during targeting and payment.
- Align Slice 5's Quilled Greatwurm architecture with existing engine concepts before coding: generalize `TriggerCondition::DealsCombatDamage.to: Option<Actor>` (preserving existing learning tokens via transparent `TokenWriter::serialize_some`), reuse `EffectValue::X` with per-occurrence `(context, x)` on `FiredTrigger` fed from one aggregated combat-damage step record, place the printed graveyard permission on `StaticAbility::CastFromGraveyardRemoveCounters` (like `Flashback`/`Foretell`), and use one `CostComponent::RemoveCounters { remaining, kind, predicate, choosing_from }` with `commit_counter_change` and stable `kind as usize` mode codes.

Slice 5 item 5 choices (2026-11-24):
- `StaticAbility::CastFromGraveyardRemovingCounters(u8)` carries only the count. The plan sketched `kind: Option<CounterKind>` and a creature predicate; no card needs either, so they wait for one.
- `CostComponent::RemoveCounters { remaining, choosing_from }` removes counters through `commit_counter_change` (no event, no receipt contribution: a counter is not an object payment). The kind mode is the `CounterKind` discriminant.
- The per-creature combat-damage totals are a trigger-only event (`GameEvent::CombatDamageTotals`, `Game::emit_trigger_event`): pending events only, never the hashed ledger, observation window or policy history, which already record each `DamageDealt`. Emitting it through `emit` would have changed the state hash and policy history of every game with combat damage. Revisit if history (MTG-418) wants a per-step damage total.
- `trigger_multiplicity` now reads through `ActiveIf` (`event_condition`), so a gated per-object trigger counts each matching creature. The one existing gated case (`dft` saddled attack on `This`) is unchanged.
- The Glamdring evidence casts Gleam of Death (the stashed Glamdring's Adventure) because Glamdring's reduction applies to instants and sorceries; Glamdring's own {2} artifact cast needs no reduction.
- `test_player_control.py::test_delegated_search_keeps_viewer_and_resource_ownership[1-*]` fails here: with caster 1 the controlled seat has 16 Islands, so Islandcycling's {2} asks for a manual tap instead of finishing itself, and the loop finds no pass. Nothing in Slice 5 touches mana payment; gate should compare with main.

## 2026-11-24 post-Slice-5 whole-branch concept review (`bb2fb68c0d` vs `88ce96bd6b`)

Three reversible choices recorded in `scratch/concept-review.html` and `scratch/plan.md`:
- Remove `frame.mode = Some(usize::from(*remaining))` for `CostComponent::RemoveCounters` (`continuation.rs:724-727`) and project `"counters remaining"` (reusing `EXECUTION_TEXT[77]`, matching `Decision::DistributeCounters`) plus `"{remaining} counter(s) left to remove."` through `effect_execution::project_pending_payment`. Fixes `DecisionContextView.chosen_mode: Some(6)` / LLM `"Chosen mode: 7."` / missing remaining-counter progress with zero new vocabulary or columns, while leaving pre-existing multi-tap `CostComponent::Tap` (`continuation.rs:714-723`) untouched to avoid churning unrelated activation fixtures.
- Update `action::PLAY_ROUTE` in `observation_layout.rs:488` to list `9 printed graveyard` and `1..9`, matching `play_route_code` and `docs/observation-and-model.md`.
- Check `self.state.suspended_decision.is_some()` before `if route.is_some()` in `Game::resolved_play_route` (`play.rs:453`) and require `route: None` on `Action::CastSpell` in `Decision::CastDuringResolution` (`decision.rs:2905`) to reject forged priority routes during resolution.

## 2026-11-24 implement pass (pre-gate items 1–3)

- `"counters remaining"` carries the bare `u8` for a removal cost (DistributeCounters carries `(kind, remaining)`); the summary pluralizes ("1 counter left to remove.").
- D6: the stack entry names its event object in the native description (`" — event: Gray Ogre (object 7)"`, the same `Game::trigger_event_label` the ChooseTrigger offer already used), so LLM text and the Academy label read it without a new `StackObjectData`/`StackEntryView` field or protocol change. The policy input already joined the context through the kind-3 continuation frame (`frame.context`, `frame.x`); only `StackObjectData.x` was missing for the stack row. If Academy later wants to draw an arrow to the event creature, add a structured reference then.
- X wording: an entering permanent's own trigger reads "X (N)" (CR 107.3m: its spell's X); every other trigger reads "that many (N)" (counters put, per-creature combat damage). The choice is by `TriggerCondition` (`enters_itself`), applied to the rendered effects text.
- `tests/verify/test_effect_execution_input.py` failed on the branch before this pass (relations 49 → 50 since `be71091782` appended `relation::SPELL_PROFILE`); it now compares the recorded 49 columns as a prefix.
