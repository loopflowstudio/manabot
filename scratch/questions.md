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
