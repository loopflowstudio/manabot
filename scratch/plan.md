# MTG-419 · Shared casting profiles and play permissions

Task steering, 2026-11-24: “Continue through straightforward cards without returning to review between cards. Realign and review the cohort together. Split only at a concrete shared-rules dependency or genuinely distinct implementation boundary; record that reason and the remaining rows.”

The Task brief requires “one shared casting-profile and play-permission model” and says “Preserve complete legal choices, observations, learning tokens, LLM text, replay/forks, and the 10% Engine Budget.” It also says to “Leave scoped event-history and contextual-query extensions to the history/queries task (defer cards strictly requiring new turn-history counters).” That task is MTG-418, which is open and not merged.

## Shared casting model

A play chooses an exact object, a printed profile and a legal route. The
profile supplies the spell's characteristics, targets and printed costs. The
route supplies permission, timing and any alternative price. Additional costs
apply to the selected combination.

Built on this branch:

- **One casting authority (`CastChoices`).** Priority, resolution and `payment_preview` build it through `Game::cast_choices(player, object, profile, CastAuthority)` (`flow/play.rs`); `Decision::cast_authority()` binds `CastDuringResolution` offers. Flags are orthogonal: `resolving` for every resolution cast (CR 608.2g), `free` only for "without paying its mana cost" (CR 118.9), `flash_surcharge` only when `!resolving` and no other instant timing holds, `flashback` for stack-exit exile (CR 702.34a), `from_prepared` from `PlayGrant.prepared`. `SpellOnStack.{from_prepared, flashback}` reach the stack; `copy_stack_object` clears both. `StackObjectData`, v11 `stack::FROM_PREPARED` (38) / `stack::FLASHBACK` (39), LLM text and Academy `StackEntryView` project them.
- **Printed profiles (`SpellProfile`).** `ContentPack::spell_definition` resolves `Front`, `Adventure`, `Omen` or `BackFace`. Offers evaluate eligibility, announcement, payment and targeting against `cast.spell(...)`. A modal back enters back face up (`ZoneEntry { transformed: true }`). v11 action and relation rows carry one `SPELL_PROFILE` category (size 4).
- **Play routes (`PlayRoute`)**: `Hand`, `FreeFromHand`, `LibraryTop`, `PrintedFlashback`, `GrantedFlashback`, `Grant(usize)`, `GraveyardType`. Static permissions on `LayerBoard.play_permissions` (`StaticPlayPermissions`, `flow/statics.rs`): `look_at_top`, `free_from_hand`, `opponents_hand_only`, `graveyard_flashback`, `library_spells`, `flash_spells`. `LayerBoard.additional_land_plays_each_player` is still a separate field, and `play_routes` still scans the battlefield for Muldrotha.
- **Single-route wire compatibility.** `Action::{CastSpell, PlayLand}` keeps `route: None` for a lone route; `Game::resolved_play_route(action)` resolves it for observation, focus and description. Front-face `CastDuringResolution` offers stay `Action::SelectCard`; other profiles use `Action::CastSpell`.
- **Exact grants (`PlayGrant`, `flow/cast_permission.rs`)**: caster/liveness, timing, cost/payment, `foretold`, `on_adventure`, `flashback`, `prepared`. Face-down exile visibility still goes through `is_foretold`.
- **Zone-aware spending.** `SpendingContext::Cast { spell, from }`, `from` from `Game::casting_zone` (the origin zone while costs are paid, CR 601.2h). `ManaRestriction::ExceptSpellsFromHand`; every exclusion orders after narrower classes (`SpendOrder::Specific(usize::MAX)`). The four column restrictions stay in `PlayerData.restricted_mana`; any other restriction goes to `pooled_mana_classes` → `POOLED_MANA` facts (total still in `lesson_mana`).
- **Activation conditions.** `ActivatedAbilityDefinition.condition` ("Activate only if", checked in `check_activation_terms` and the payment probe); `Condition::CastThisTurn(CardPredicate)` reads `TurnState.spells_cast` — **defect found by post-Slice-1 concept review:** `TurnState.spells_cast` currently stores bare `ObjectId`, so once an Adventure, Omen, or Modal BackFace leaves the stack its profile reverts to the card's front face (locking Sanar back out after Spew Flame or Charring Bite resolves).
- Omniscience replaces the mana cost but allows Kicker/Bargain (CR 118.9d) and mandatory additional costs; `AdditionalCost::FlashSurcharge` is skipped during resolution.

Planned (Slice 1 alignment + Slices 2–3; none of this exists yet):

- **Profile-faithful turn cast history:** `TurnState.spells_cast: [Vec<RecordedCast>; 2]` with `RecordedCast { object: ObjectId, profile: SpellProfile }` and `recorded_cast_matches_predicate` resolving `profile` against the retained front printing once departed.
- **Consumer parity (Academy & text):** Render `from_prepared` and `flashback` tags in Academy's stack rail (`Rail.svelte`); drop redundant `"only "` before `"anything but ..."` mana exclusions in `text_observation.py:865`, `mana.ts:11`, and `Plate.svelte:41`.
- **`StaticPlayPermissions` consolidation:** Own `graveyard_type_sources`, `exile_counter_plays { kind, own_turn_only, any_mana }`, and per-seat `extra_land_plays` (absorbing `additional_land_plays_each_player`); one deduplicated `exile_playable_objects(player)` over exact grants and counter-marked exile; `permits_spell` explicitly allows all profiles for `PlayRoute::ExileCounter(_)`; `HandOnly` checks non-land profiles so spell//land MDFCs keep back-face land plays.
- **Exile counters & printed graveyard cast:** `PlayRoute::ExileCounter(CounterKind)` and `PlayRoute::PrintedGraveyard`; sparse `GameState.object_counters: BTreeMap<ObjectId, Counters>`.
- **Complete two-seat face-down exile privacy:** `PlayGrant.face_down` and `condition: Option<Condition>`; `is_face_down_in_exile` / `can_view_face_down_exile(object, viewer)` replace visibility uses of `is_foretold`; hidden-exile resampling runs for **both** seats (`shuffle_known_library` for own seat / `shares_information` opponent as well as unshared opponent determinization) so turn-2/3 Flameshape without a Wizard never leaks to its owner's search; `HiddenHandConstraints.face_down_exile` (omitted when 0) keeps `PossibleWorldSpace`, `validate_source`, and `selection_sampling` exact; `CastPermissionData` projects `prepared` and `condition` (omitted when absent/false) to policy facts (`permission_facts` named extension record, Hama's tokens unchanged), LLM text (non-foretold wording for `card.face_down`), and Academy `inspect.ts`.
- **Source-aware dynamic cost reduction:** Keep `(source_id, view)` in `effective_spell_cost` and add `EffectValue::AttachedCreaturePower` evaluated via `self.attached_creature(source_id)` → `layer_view(c).power.max(0)` (Glamdring).

## Delivered

The per-set ledgers link each implemented row to its definition and native
fixture. The shared authority first shipped on this branch with Zul Ashur,
Muldrotha, Vizier, Etali, Smaug, Bilbo, Twinmaw and Scavenger Regent. The
first 44 preparation rows include linked copies, source transfers, preparation
during resolution and Emeritus of Truce.

- Yue offers Spew Flame from hand and Solstice Revelations offers Charring Bite from exile (`f3d84a7abe`, `be71091782`).
- Monica Rambeau // Photon uses one front/back definition pair. Photon pays its printed cost, enters back face up and returns to Monica on leaving.
- Foundations covers 11 of 17 rows: Omniscience (`FreeFromHand` + manual Kicker on Burst Lightning, deduplicated across copies), High Fae Trickster (timing), Harbinger of the Tides (surcharge + required sacrifices on free cast), Sphinx of Forgotten Lore (exact-object `flashback` grant alongside Think Twice's printed route), Reassembling Skeleton, and Suspicious Shambler.
- Harmonized Trio, Joined Researchers, and Codie, Ravenous Codex bring preparation to 47 of 62 rows (`bf5e38d587`).
- HOB covers 12 of 17 Adventures plus Bilbo's competing exile replacement. TDM covers 12 of 13 Omens (Feral Deathgorger same-graveyard targeting, Till and Tend stacked extra land plays).
- Shared casting authority alignment (`c5c415307f`, `cb30b69432`): `Game::cast_choices` with `CastAuthority` (R1 Harbinger resolution fix), `Game::resolved_play_route` (R2 lone route observation/focus/description fix), front-face `SelectCard` resolution `payment_preview` (R3 fix), and `SpellOnStack.from_prepared` (cleared on stack copies).

Behavior review at `a805327074` (debug build, constructed fixtures): R1–R3 from the `e0d3c07b74` review are fixed and no defect remains open. Evidence: `harbinger_cast_by_a_resolving_effect_owes_no_flash_surcharge` (Etali, `{2}` floating, no `PayOrNot`, nothing spent), `lone_graveyard_routes_keep_their_permission_in_the_observation` (lone Muldrotha quota/focus/description; lone Zul grant `route_mana_value: 2`), and `bilbo_previews_front_face_and_adventure_graveyard_casts_alike`. A reverted probe confirmed a lone Muldrotha graveyard *land* play keeps `quota_type` Land, both focus ids and its description while the Command keeps `route: None`. Codie passes (original `from_prepared`, copy and ordinary cast not). 38 casting fixtures, `preparation_resolution` (5) and the Validation Python controls (99 passed; the optional U1952 pilot skipped) pass on a rebuilt extension. Slice 1 (`badf9179d3`..`56046282a5`) then closed the stack-visibility gap and built Heartwood Crafter // Soul Tether (`fra.rs`) and Sanar, Unfinished Genius // Wild Idea (`sos.rs`), plus the Python text test `test_a_prepared_spell_is_told_apart_from_its_copy`; Academy fixtures refreshed (labels unchanged).

Behavior review of Slice 1 at `592331aa12` (debug build, constructed fixtures, extension rebuilt from HEAD):

- **Met.** Stack provenance reaches every shared consumer: `StackObjectData.{from_prepared, flashback}`, v11 stack columns 38/39, LLM text and Academy's `decision_surface_view`. Both viewers see the Codie copy as `false` and the prepared original as `true`. Heartwood's `{C}` pays Soul Tether from exile and is refused for a Gray Ogre from hand. A reverted probe with `CRR` plus the restricted `{C}` showed the competing spends work: auto spends the restricted `{C}` first and leaves ordinary `{C}` floating, and manual offers `{R}`, ordinary `{C}` and the restricted `{C}` as separate spends. The restriction is a `POOLED_MANA` fact program (tokens come from the card's serialized definition) with zero `unrepresentable_state`. Sanar's condition holds while a matching spell is on the stack. It also holds after an instant resolves, and after its own prepared Wild Idea copy resolves (probe). Activation and the payment probe both check the condition. Wild Idea offers only instants and sorceries and reveals what it finds.
- **D1 (open, Slice 2 item 1). Cast history loses the profile.** A reverted probe in `sos.rs` used Sanar and Smaug (`cast_profile(.., SpellProfile::Adventure)`), targeting an opposing Ogre. `offers_ability(sanar, 0)` is true while Spew Flame is on the stack and false once it resolves. After the stack appearance departs, `recorded_object_matches_predicate` reads `source_base`, which is the Creature front. `SpellsCostLess.first_on_own_turn` reads the same history. Fix: the planned `RecordedCast { object, profile }`. Use this probe as its regression, plus an Omen (Charring Bite) control.
- **D2 (fixed here).** `lea::clone_copy_survives_suspended_entry_and_keeps_its_printed_base` pinned a state hash from `f3d84a7abe`. Catalog additions since then (Codie included, before Slice 1) moved its definition ordinals. The pin is refreshed to `v19:5ab0103d…`, and the rest of the test passes. Full debug `managym-cards --test cards`: 897 tests pass. Plus the refreshed test, the earlier failure was this pin alone.
- **Consumer nits (open, Slice 2 item 1).** Text renders "only for anything but spells from hand", and the native spend label reads "(anything but spells from hand only)". Academy's `Rail.svelte` does not yet show the `from_prepared`/`flashback` tags. The data reaches `StackEntryView`.
- **Controls.** Pass: the plan's five Rust controls, `heartwood`/`sanar`/`codie`/`harbinger`/`lone_graveyard`/`bilbo`, `managym --lib` (355) and the Validation Python controls (100 passed, U1952 pilot skipped, `test_pinned_pilot` shows no break). Slice 1 is satisfied except D1, which Slice 2 item 1 owns. Gate owns Clippy, the full workspace, Engine Budget and `conformance/`.


## Execution plan for remaining cohorts

Nineteen MTG-419 rows remain (5 in Slice 2, 2 in Slice 3, plus 12 non-casting shared-rules rows). Six further preparation rows belong to MTG-418. Post-Slice-1 concept review (`scratch/concept-review.html`) uncovered one live cast-history bug in built Slice 1 code, closed the Slice 1 consumer gaps, and tightened Slice 2's face-down exile privacy, route legality, and Glamdring reduction design.

### Slice 2 — Profile-faithful cast history, consumer parity, static permissions consolidation, counter-marked exile, two-seat private exile & dynamic pricing (fixes + 5 cards: Tinybones, Gandalf, Glamdring, Loot, Redcap)

1. **Profile-faithful `TurnState.spells_cast` & Slice 1 consumer parity (do first — proves the core invariant immediately):**
   - Replace `TurnState.spells_cast: [Vec<ObjectId>; 2]` with `[Vec<RecordedCast>; 2]` where `RecordedCast { object: ObjectId, profile: SpellProfile }` (`turn.rs`, `play.rs`, `game.rs`, `triggers.rs`, `statics.rs`, `state/hash.rs`).
   - Add `Game::recorded_cast_matches_predicate(cast, predicate)` resolving `profile` against the retained front printing when `live(cast.object)` is `None` (keeping `Front` on `ObjectKind::SpellCopy` using its own retained inset printing). Update both `Condition::CastThisTurn` (`statics.rs:911`) and `SpellsCostLess.first_on_own_turn` (`play.rs:542`).
   - Add a regression fixture proving Sanar's Treasure activation remains legal after casting and resolving an Adventure Sorcery (Smaug's Spew Flame) or Omen Instant (Twinmaw's Charring Bite).
   - Render concise `prepared` / `flashback` badges on `StackEntryView` items in `academy/ui/src/lib/board/Rail.svelte` (so Codie's copy and original are visually distinct in Academy).
   - Drop redundant `"only "` before exclusions starting with `"anything but "` in `manabot/semantic/text_observation.py:865`, `academy/ui/src/lib/mana.ts:11`, and `academy/ui/src/lib/board/Plate.svelte:41` (`"for anything but spells from hand"` / `"for anything but nonartifact spells"`, leaving positive restrictions as `"only for Lesson spells"`).
2. **Consolidate `StaticPlayPermissions` & per-profile legality:**
   - Move Muldrotha's `graveyard_type_sources`, Tinybones's `exile_counter_plays`, and per-seat `extra_land_plays` (merging `AdditionalLandPlay` + `AdditionalLandPlayEachPlayer` and removing `LayerBoard.additional_land_plays_each_player`) onto `LayerBoard.play_permissions` in `statics.rs`. Remove the per-card battlefield scan from `play_routes`.
   - Deduplicate exiled priority candidates in one `Game::exile_playable_objects(player)` iterator (covering `exile_granted_objects` + `exile_counter_objects` across both players' exile zones).
   - Move `cast_spell_action`'s `is_land()` rejection (`play.rs:610`) after `cast_choices` so it checks `cast.spell(spell).types.is_land()`; in `play_routes:245`, guard `casts_only_from_hand` with `!land && !self.state[object].modal_back` and enforce `!self.casts_only_from_hand(player)` for non-hand spell casts (`!types.is_land()`), so a spell//land MDFC's back-face land play is never wrongly blocked.
   - In `permits_spell` (`play.rs:334`), explicitly allow `PlayRoute::ExileCounter(_) => true` so stashed Adventures, Omens, and Modal BackFaces are castable under Tinybones.
3. **Tinybones, Bauble Burglar (FDN):** Add sparse `GameState.object_counters: BTreeMap<ObjectId, Counters>` (projected to `CardData.counters`, v11 `FactKind::Counter`, LLM text, Academy), `PlayRoute::ExileCounter(CounterKind)` (applying `pay_with_any_mana()` in `casting_mana_cost` when the static permission grants it), and discard-event trigger. Deliver the full required interaction-evidence fixture (including overlapping grant + stash counter deduplication, casting a stashed Adventure/Omen, Tinybones #1 → #2, foreign land control, `HandOnly`, and High Fae Trickster non-bypass).
4. **Gandalf, Goblins' Bane // Flameshape (HOB):**
   - Decouple `PlayGrant.face_down: bool` and `PlayGrant.condition: Option<Condition>` from `PlayGrant.foretold`.
   - Replace the 5 visibility calls to `is_foretold` (`zones.rs`, `continuation.rs`, `observation.rs`, `policy_history.rs`) with `is_face_down_in_exile` and `can_view_face_down_exile(object, viewer)` (enforcing both `shares_information` and live `condition`).
   - Update `selection_sampling.rs::unseen_objects` to include every exiled card of `owner` where `is_face_down_in_exile(obj) && !can_view_face_down_exile(obj, viewer)` — **for both `owner == viewer` and `owner != viewer`**.
   - Share one hidden-exile resampling path across **both** seats: in `search.rs`, `possible_worlds.rs`, and `knowledge.rs::shuffle_known_library` (own seat and `shares_information` opponent), pool `hidden_face_down_exile(owner, viewer)` with the unknown library and run `fill_foretold_slots` (foretell-bearers only) followed by `fill_face_down_exile_slots` (any remaining unseen card).
   - Add `face_down_exile: u32` (non-foretold hidden exile count, `#[serde(skip_serializing_if = "crate::state::is_zero")]`) to `HiddenHandConstraints` / `PossibleWorldSpace`; check it in `validate_source` and subtract `foretold + face_down_exile` from library slots in `selection_sampling.rs`.
   - Project `prepared` and `condition` on `CastPermissionData` (omitted when false/absent); update `permission_facts` (`observation_encoder.rs:1996`) so singleton conditional/prepared grants are emitted via a named extension record (keeping Hama's legacy token sequence byte-stable); update `text_observation.py:941` to distinguish foretold face-down exile from generic face-down exile (`"face down in exile (hidden from players who cannot look at it)"`) and render `condition` / `prepared` in `text_observation.py` and `academy/ui/src/lib/board/inspect.ts`.
   - Test Flameshape cast on turn 3 with no Wizard on board (hidden from owner, in owner's `unseen_counts`, resampled by `determinize(owner)`!), Wizard persistence across Gandalf dying with another Wizard, losing all Wizards (hidden from owner, counted in unseen, unplayable), regaining a Wizard (visible & playable again), opponent `unseen_counts` privacy, and `possible_worlds` / `determinize` resampling.
5. **Glamdring, Foe-hammer // Gleam of Death (HOB):** Retain `(source_id, view)` in `effective_spell_cost` (`play.rs:521`), add `EffectValue::AttachedCreaturePower` evaluated via a source-aware helper reading `self.attached_creature(source_id)` → `layer_view(c).power.max(0)` (0 when unequipped), and wire `SpellsCostLess` to support dynamic `EffectValue` reductions; Gleam of Death mills six and returns instants/sorceries before Adventure exile.
6. **Loot, Exuberant Explorer (FDN):** Wire `StaticAbility::AdditionalLandPlay` into `StaticPlayPermissions.extra_land_plays` + private look-six selection bounded by resolution-time land count.
7. **Redcap Gutter-Dweller (FDN):** Commit optional another-creature sacrifice before +1/+1 counter and `exile_top_cards_to_play(ThisTurn)`; create printed nonblocking Rat tokens.
8. Update `fdn.jsonl` and `hob.jsonl` ledgers.

### Slice 3 — Graveyard counter-payment, TDM completion & non-casting split boundary

1. **Quilled Greatwurm (FDN):** Add `PlayRoute::PrintedGraveyard` (so distributed `CostComponent::RemoveCounters` payment in `payment.rs` attaches strictly to Quilled Greatwurm's own graveyard permission and never interferes with Muldrotha or Zul Ashur on the same card) + combat-damage counter trigger. Include an overlap fixture with Muldrotha.
2. **Bloomvine Regent // Claim Territory (TDM — closes TDM 13/13):** Split-destination library search revealing both found cards before choosing which enters tapped and which goes to hand.
3. **Remaining 12 non-casting shared-rules rows (concrete split boundary):** None of these 12 rows exercises new casting-profile or play-permission concepts; batch only the small clean ones that fit within the 10% Engine Budget, and split cleanly at the named shared-rules boundary for any remaining family (recording the reason and rows per the 2026-11-24 steering):

| Cohort | Cards | Shared rule needed (outside casting/permission core) |
|---|---|---|
| Foundations (2) | Flamewake Phoenix, Gate Colossus | Graveyard-zone triggers (beginning-of-combat / landfall) retaining exact graveyard appearance through optional return. |
| Preparation (2) | Spiritcall Enthusiast, Kirol, History Buff | Batch-collapsed triggers ("one or more" entries or graveyard departures) in `process_event_batch`. |
| Preparation (3) | Grave Researcher, Emeritus of Ideation, Lluwen, Exchange Student | Surveil-before-count + MV life loss; distinct graveyard-card exile costs for attack trigger / sorcery activation + Pest token. |
| Preparation (2) | Variable Chaser, Pyre Rhymer | Separate APNAP optional whole-hand discard/draw; turn-scoped delayed mana trigger on tapping any Mountain for mana. |
| HOB (3) | An Unexpected Party, Bilbo, Luckwearer, Great Ugly-Looking Goblin | Chosen creature type retained on entry; shared-card-type control exchange; Amass (create or grow existing Army). |

### Deferred to MTG-418 (6 preparation rows)

MTG-418 owns six history-dependent rows: Emeritus of Woe and Bloodline Recollector (creature deaths this turn), Leech Collector (first life gain this turn), Scheming Silvertongue (total life gained this turn), Whiplash Wordsmith (noncombat damage to an opponent), and Hexhaven Dueling Arena (specific creature that attacked). On that merged revision, start with `cargo test --manifest-path managym/Cargo.toml -p managym --lib preparation_resolution` as the preparation control before adding the six directed fixtures.

## Interaction evidence still owed

| Position | Required fork or outcome |
|---|---|
| Sanar + resolved Adventure/Omen | Cast Spew Flame (Adventure Sorcery) or Charring Bite (Omen Instant), let it resolve so its stack appearance departs and the card slot reverts to its Creature front face; verify Sanar's Treasure activation remains legal and `SpellsCostLess.first_on_own_turn` recognizes the cast. |
| Tinybones, opposing discarded land and spell (plus an Adventure/Omen) with stash counters, second Tinybones, plus overlapping exile grant | Current controller plays foreign cards exiled by an earlier Tinybones; overlapping grant + stash counter emits no duplicates; stashed Adventure/Omen is castable. Check own-turn timing, foreign land control (enters under Tinybones’s controller, not owner), ownership versus control, mana-of-any-type spending, High Fae Trickster non-bypass of own-turn timing, and `HandOnly` (blocks spell, allows land). Use Vizier's private top card and foretell's hidden exile as opposite-viewer controls. |
| Gandalf’s Flameshape vs. Foretell | Cast Flameshape on turn 3 with no Wizard on board: cards are hidden from owner, counted in owner's `unseen_counts`, and resampled by `determinize(owner)`. When a Wizard enters, they become visible (never labeled "foretold") and playable; persist when Gandalf dies and another Wizard remains; hide and lock when last Wizard leaves; restore when a new Wizard enters; hidden cards stay in `unseen_objects` / `unseen_counts` for viewers who cannot see them (no open-decklist leak) and determinize via `fill_face_down_exile_slots` with `HiddenHandConstraints.face_down_exile` exact. |
| Quilled Greatwurm + Muldrotha | Overlapping `PrintedGraveyard` (removes six +1/+1 counters, preserves Muldrotha quota) vs. `GraveyardType { Creature }` (no counter removal, spends Muldrotha quota). |

## Validation

**Consumer checks per slice.** Compare native observations for both viewers with the Python bindings, semantic capture/LLM text and Academy offers at priority, at mid-announcement (`AnnounceX`/`ChooseTargets`/`Pay`), with a spell on the stack, and at a nested decision. Use v11 rows and learning tokens. Keep hidden information out of the other viewer's data (including `unseen_counts` and determinization). Fork-and-replay checks need exact preconditions and stale rejection. Existing controls:

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards airbend_permission
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards flashback_is_exiled_when_countered_after_iroh_leaves
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards sozins_comet_foretell_spending_choices_preserve_color_and_private_identity
cargo test --manifest-path managym/Cargo.toml -p managym --lib preparation_resolution
uv run pytest tests/semantic/test_text_observation.py tests/semantic/test_restricted_mana_input.py tests/semantic/test_pooled_mana_classes_input.py tests/academy/test_choice_navigation.py tests/model/test_pinned_pilot.py -q
```

After `bf5e38d587` the validation Python controls passed (86) once `scripts/refresh_academy_fixtures.py` refreshed the catalog digests (`6a1adcccef`; labels unchanged). The conformance receipts have their own writers and were not regenerated here; gate should check `conformance/` for the new catalog.

Gate owns the full debug workspace, Clippy and the measured 10% Engine Budget.

## Later phases

**Decomposition.** Follow Muldrotha and Zul from the same graveyard object through payment, Quilled Greatwurm + Muldrotha overlap, Omniscience and Harbinger from hand (and Harbinger via free resolution cast), Tinybones from counter-marked exile, Gandalf from conditional face-down exile, and an Adventure, Modal BackFace, and linked prepare copy through stack and departure. One static-permission owner (`LayerBoard.play_permissions`), one route model (`PlayRoute`), one profile owner (`SpellProfile`), one `CastChoices` builder, one profile-faithful cast history (`RecordedCast`), and one open-ended pooled-mana fact path (`POOLED_MANA`) must explain all of them, including airbend, flashback, foretell, and modal regressions. Remove any exclusive old paths. Review repeated FRA spell programs for card-specific dispatch.

**Performance.** Use the 10% same-runner Engine Budget and the [stress-state protocol](../docs/development.md#stress-states). Add `bench_states` shapes for:
- Muldrotha with a graveyard of duplicate artifact creatures, overlapped by a Zul grant (now reading cached `StaticPlayPermissions.graveyard_type_sources` instead of scanning the battlefield per card).
- A hand of Dragon/Omen cards with restricted mana and several Charring Bite targets.
- Many Phytomedics with their own Seed Sutures, with one transfer and a zero-link control.

This is where to decide whether the linear grant scan needs an index. Measure first.
