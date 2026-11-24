# MTG-419 · Shared casting profiles and play permissions

Task steering, 2026-11-24: “Continue through straightforward cards without returning to review between cards. Realign and review the cohort together. Split only at a concrete shared-rules dependency or genuinely distinct implementation boundary; record that reason and the remaining rows.”

The Task brief requires “one shared casting-profile and play-permission model” and says “Preserve complete legal choices, observations, learning tokens, LLM text, replay/forks, and the 10% Engine Budget.” It also says to “Leave scoped event-history and contextual-query extensions to the history/queries task (defer cards strictly requiring new turn-history counters).” That task is MTG-418, which is open and not merged.

## Shared casting model

A play chooses an exact object, a printed profile and a legal route. The
profile supplies the spell's characteristics, targets and printed costs. The
route supplies permission, timing and any alternative price. Additional costs
apply to the selected combination.

- **One casting authority (`CastChoices`).** Priority, resolution, and `payment_preview` construct `CastChoices` through `Game::cast_choices(player, object, profile, CastAuthority)` (`flow/play.rs`), with `Decision::cast_authority()` binding `CastDuringResolution` offers. Its flags are orthogonal: `resolving: bool` is true for *all* `CastDuringResolution` casts (both free and paid, CR 608.2g), `free: bool` strictly means "without paying its mana cost" (CR 118.9), `flash_surcharge: bool` is required only when `!resolving` and no other instant timing holds, `flashback: bool` marks stack-exit exile (CR 702.34a), and `from_prepared: bool` captures prepared-copy provenance from `PlayGrant.prepared` before the copy leaves exile. `SpellOnStack.{from_prepared, flashback}` carry stack provenance onto the stack, `copy_stack_object` clears both alongside `exile_instead_of_graveyard`, and `StackObjectData` / v11 `stack::{FROM_PREPARED, FLASHBACK}` / `text_observation.py` / `experience_projection.py` project both so policy and human viewers distinguish prepared spells, stack copies, and flashback spells.
- **Printed profiles (`SpellProfile`).** `ContentPack::spell_definition` resolves `Front`, `Adventure`, `Omen`, or `BackFace` for the physical object. Priority and resolution offers evaluate eligibility, land vs. spell type checks (`permits_spell` and `cast_spell_action` using `cast.spell(...)`, never front-face shortcuts), announcement, payment, and targeting against `cast.spell(...)`. A modal back enters the battlefield back face up (`ZoneEntry { transformed: true }`) and restores the front on leaving. Adventures and Omens follow their own stack-exit rules (CR 715.3d / shuffle). `ContinuationFrame` and `DecisionContextView` project the selected profile through announcement; the v11 action and relation rows carry one `SPELL_PROFILE` category (size 4).
- **Play routes & one layered static-permission owner (`PlayRoute` + `LayerBoard.play_permissions`).** `PlayRoute` covers `Hand`, `FreeFromHand` (Omniscience), `LibraryTop` (Vizier), `PrintedFlashback`, `GrantedFlashback` (Iroh), `PrintedGraveyard` (Quilled Greatwurm — binds its `CostComponent::RemoveCounters` strictly to this route, never Muldrotha/Zul), `Grant(usize)` (exact-object grants), `GraveyardType` (Muldrotha per-type quota), and `ExileCounter(CounterKind)` (Tinybones — applies `pay_with_any_mana()` in `casting_mana_cost`). **All** static play permissions live together on `StaticPlayPermissions` in `LayerBoard` (`compute_board`): `look_at_top`, `free_from_hand`, `opponents_hand_only`, `graveyard_flashback`, `library_spells`, `flash_spells`, `graveyard_type_sources` (eliminating the per-card battlefield scan in `play_routes`), `exile_counter_plays` (with `kind`, `own_turn_only`, `any_mana`), and per-seat `extra_land_plays` (combining `AdditionalLandPlay` and `AdditionalLandPlayEachPlayer` and removing the separate `LayerBoard.additional_land_plays_each_player` field). `priority_candidate_objects` visits exiled cards through one deduplicated `exile_playable_objects(player)` iterator spanning exact grants and counter-marked exile (both seats). A land's chosen route authorizes foreign ownership and sets `ZoneEntry { controller: Some(player) }`.
- **Single-route wire compatibility vs. route inspection.** `Action::{CastSpell, PlayLand}` keeps `route: None` when only one route exists so retained single-route `Command`s stay wire-compatible, while `Game::resolved_play_route(action)` resolves the actual `PlayRoute` for observation projection, `action_focus`, and `action_description` so solitary `GraveyardType` (Muldrotha) plays and solitary priced grants always expose `quota_type`, `route_mana_value`, grantor focus, and route description. `Decision::CastDuringResolution` keeps `Action::SelectCard` for front-face offers and `Action::CastSpell` for non-front profiles to preserve retained replay captures.
- **Exact grants, private face-down exile & hidden-card accounting (`PlayGrant`).** Exact grants keep orthogonal fields: caster/liveness (`player`, `source`, `follows_source_controller`, `condition: Option<Condition>`), timing/duration (`this_turn`, `through_next_turn`, `own_turn_only`, `after_turn`, `flash`), cost/payment (`cost`, `any_mana`, `waterbend`), visibility (`face_down`), Foretell semantics (`foretold`), profile restriction (`on_adventure`), stack-exit (`flashback`), and linked copy survival (`prepared`). `is_face_down_in_exile(object)` and `can_view_face_down_exile(object, viewer)` govern **all** private face-down exile visibility (`zones.rs`, `continuation.rs`, `observation.rs`, `policy_history.rs`) **and** hidden-card accounting (`selection_sampling.rs::unseen_objects` includes every exiled card of `owner` where `is_face_down_in_exile && !can_view_face_down_exile(obj, viewer)`). Determinization (`search.rs`, `possible_worlds.rs`) refills `foretold_cards` first via `fill_foretold_slots` (foretell-bearing cards only) and then refills remaining non-foretold face-down exile slots via `fill_face_down_exile_slots` from any remaining unseen cards, preserving pool conservation and preventing open-decklist leaks. In `permission_facts` (`observation_encoder.rs`), keep Hama's legacy token sequence unchanged and encode post-TLA grant terms (`condition`, `prepared`, `face_down`) via one named extension record instead of extending positional tuple branches.
- **Open-ended restricted mana & zone-aware spending (`SpendingContext` + `POOLED_MANA` facts).** `SpendingContext::Cast { spell: SpellSnapshot, from: ZoneType }` supplies the casting zone to `ManaRestriction::allows` (one spending authority for `ExceptSpellsFromHand` and all other restrictions), and `ManaClass::spend_order` treats `ExceptSpellsFromHand` like `ExceptNonartifactSpells` (`SpendOrder::Specific(usize::MAX)`). For observations, the 4 legacy column restrictions (`Lesson`, `Lesson|Shrine`, `Ally`, `ExceptNonartifactSpells`) stay untouched in `PlayerData.restricted_mana` and their existing `player`/`turn_state` columns (`unrepresentable_state` continues refusing unknown restrictions placed in `restricted_mana`, keeping `test_restricted_mana_input.py` and the pinned pilot untouched). **All other `ManaRestriction`s** (starting with `ExceptSpellsFromHand`, `describe(): "anything but spells from hand"`) flow through the existing `PlayerData.pooled_mana_classes` -> `FactKind::EffectProgram(POOLED_MANA)` path (while still counted in `lesson_mana` totals): zero new `turn_state` columns, zero `RestrictedManaData` / PyO3 / schema / UI churn, and automatic support in `text_observation.py`, Academy `Plate.svelte`, and `PaymentPreview`.
- **Sparse exiled-object counters & dynamic pricing.** Non-battlefield counters live in sparse `GameState.object_counters: BTreeMap<ObjectId, Counters>` (projected to `CardData.counters`, v11 `FactKind::Counter`, LLM text, and Academy) so normal objects stay unchanged and CR 400.7 zone moves automatically stale old keys. Omniscience replaces the mana cost while still permitting Kicker/Bargain (CR 118.9d) and requiring mandatory additional costs. Harbinger's `AdditionalCost::FlashSurcharge` buys flash only when required at priority and is skipped during resolution. `StaticAbility::SpellsCostLess` supports dynamic `EffectValue` reductions (equipped creature power for Glamdring). Compound priority offers bind the chosen target with its spell profile and route while keeping compact priority precondition ordering stable.

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

Behavior review at `a805327074` (debug build, constructed fixtures): R1–R3 from the `e0d3c07b74` review are fixed and no defect remains open. Evidence: `harbinger_cast_by_a_resolving_effect_owes_no_flash_surcharge` (Etali, `{2}` floating, no `PayOrNot`, nothing spent), `lone_graveyard_routes_keep_their_permission_in_the_observation` (lone Muldrotha quota/focus/description; lone Zul grant `route_mana_value: 2`), and `bilbo_previews_front_face_and_adventure_graveyard_casts_alike`. A reverted probe confirmed a lone Muldrotha graveyard *land* play keeps `quota_type` Land, both focus ids and its description while the Command keeps `route: None`. Codie passes (original `from_prepared`, copy and ordinary cast not). 38 casting fixtures, `preparation_resolution` (5) and the Validation Python controls (99 passed; the optional U1952 pilot skipped) pass on a rebuilt extension. Still open in Slice 1: Codie's trigger reads `SpellOnStack.from_prepared`, but no viewer can see that flag on the stack yet. The prepared cast offer does name its source. Item 1 closes this gap. Heartwood Crafter and Sanar also remain.

## Execution plan for remaining cohorts

Twenty-one MTG-419 rows remain (9 core casting/permission/payment rows across Slices 1–3, plus 12 non-casting shared-rules rows). Six further preparation rows belong to MTG-418. Concept review (`scratch/concept-review.html`) aligned the remaining design around open-ended `POOLED_MANA` facts, unified `StaticPlayPermissions`, and leak-free face-down exile accounting.

### Slice 1 — Spending provenance, stack provenance projection & Sanar (remaining: Heartwood Crafter, Sanar)

1. **Stack cast provenance in observations:** Expose `from_prepared: bool` and `flashback: bool` on `StackObjectData` (omitted when false for JSON/replay compatibility), append `stack::FROM_PREPARED` and `stack::FLASHBACK` columns in `observation_layout.rs` / `observation_encoder.rs`, and render both in `manabot/semantic/text_observation.py` and `managym/experience_projection.py`. Verify `tests/model/test_pinned_pilot.py` projects the saved prefix with no break.
2. **Heartwood Crafter // Soul Tether (FRA):**
   - Change `SpendingContext::Cast(SpellSnapshot)` to `SpendingContext::Cast { spell: SpellSnapshot, from: ZoneType }` (populated in `effective_spell_cost` from `self.state.zones.zone_of(object)`).
   - Add `ManaRestriction::ExceptSpellsFromHand` (`allows`: false only for `SpendingContext::Cast { from: ZoneType::Hand, .. }`; `describe()`: `"anything but spells from hand"`; `SpendOrder::Specific(usize::MAX)` in `ManaClass::spend_order`), and register it in `record_vocabulary.rs`.
   - In `Observation::player_data` (`observation.rs`), route any `ManaClass` whose `restriction` is not in the 4 legacy column restrictions (`!observation_encoder::representable(restriction)`) into `pooled_mana_classes` (instead of `restricted_mana`, while keeping its total in `lesson_mana`). Leave `restricted_mana`, `turn_state`, `unrepresentable_state`, and `test_restricted_mana_input.py` untouched.
   - Implement Heartwood Crafter // Soul Tether with directed fixture in `fra.rs`: Heartwood's restricted `{C}` pays linked Soul Tether from exile, Codie copies it (`from_prepared: true` on original, `false` on copy, both visible in `Observation` + v11 `stack` + text), an ordinary hand spell rejects the same `{C}`, and `encode_observation` succeeds with `POOLED_MANA` fact rows and zero `unrepresentable_state`.
3. **Sanar, Unfinished Genius // Wild Idea (SOS):** Gate the Treasure activation on an instant/sorcery in `TurnState.spells_cast` (read via `recorded_object_matches_predicate`, needing no MTG-418 counter); Wild Idea searches, reveals, then shuffles.
4. Update `sos.jsonl` and the Heartwood row of `fra.jsonl`.

### Slice 2 — Static permissions consolidation, counter-marked exile, private conditional exile & dynamic pricing (5 cards: Tinybones, Gandalf, Glamdring, Loot, Redcap)

1. **Consolidate `StaticPlayPermissions` & per-profile land check:**
   - Move Muldrotha's `graveyard_type_sources`, Tinybones's `exile_counter_plays`, and per-seat `extra_land_plays` (merging `AdditionalLandPlay` + `AdditionalLandPlayEachPlayer` and removing `LayerBoard.additional_land_plays_each_player`) onto `LayerBoard.play_permissions` in `statics.rs`. Remove the per-card battlefield scan from `play_routes`.
   - Deduplicate exiled priority candidates in one `Game::exile_playable_objects(player)` iterator (covering `exile_granted_objects` + `exile_counter_objects` across both players' exile zones).
   - Move `cast_spell_action`'s `is_land()` rejection (`play.rs:595`) after `cast_choices` so it checks `cast.spell(spell).types.is_land()` (and check profile types in `permits_spell`).
2. **Tinybones, Bauble Burglar (FDN):** Add sparse `GameState.object_counters: BTreeMap<ObjectId, Counters>` (projected to `CardData.counters`, v11 `FactKind::Counter`, LLM text, Academy), `PlayRoute::ExileCounter(CounterKind)` (applying `pay_with_any_mana()` in `casting_mana_cost` when the static permission grants it), and discard-event trigger. Deliver the full required interaction-evidence fixture (including overlapping grant + stash counter deduplication, Tinybones #1 -> #2, foreign land control, `HandOnly`, and High Fae Trickster non-bypass).
3. **Gandalf, Goblins' Bane // Flameshape (HOB):**
   - Decouple `PlayGrant.face_down: bool` and `PlayGrant.condition: Option<Condition>` from `PlayGrant.foretold`.
   - Replace the 5 visibility calls to `is_foretold` (`zones.rs`, `continuation.rs`, `observation.rs`, `policy_history.rs`) with `is_face_down_in_exile` and `can_view_face_down_exile(object, viewer)` (enforcing both `shares_information` and live `condition`).
   - Update `selection_sampling.rs::unseen_objects` to include every exiled card of `owner` where `is_face_down_in_exile(obj) && !can_view_face_down_exile(obj, viewer)`.
   - Pair `fill_foretold_slots` (foretell-bearers only) with `fill_face_down_exile_slots` (any remaining unseen card) in `search.rs` and `possible_worlds.rs` so pool conservation holds and neither viewer leaks face-down exiled cards.
   - Replace the growing positional-tuple branches in `permission_facts` with a named extension record for post-TLA fields (`condition`, `prepared`, `face_down`), keeping Hama's legacy token sequence byte-stable.
   - Test Wizard persistence across Gandalf dying with another Wizard, losing all Wizards (hidden from owner, counted in unseen, unplayable), regaining a Wizard (visible & playable again), opponent `unseen_counts` privacy, and `possible_worlds` / `determinize` resampling.
4. **Glamdring, Foe-hammer // Gleam of Death (HOB):** Extend `SpellsCostLess` (or dynamic reduction in `effective_spell_cost`) with `EffectValue` reading equipped creature's current power; Gleam of Death mills six and returns instants/sorceries before Adventure exile.
5. **Loot, Exuberant Explorer (FDN):** Wire `StaticAbility::AdditionalLandPlay` into `StaticPlayPermissions.extra_land_plays` + private look-six selection bounded by resolution-time land count.
6. **Redcap Gutter-Dweller (FDN):** Commit optional another-creature sacrifice before +1/+1 counter and `exile_top_cards_to_play(ThisTurn)`; create printed nonblocking Rat tokens.
7. Update `fdn.jsonl` and `hob.jsonl` ledgers.

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
| Heartwood Crafter // Soul Tether + Codie | Restricted `{C}` pays exile Soul Tether, rejected by hand spell; encodes via `pooled_mana_classes` / `POOLED_MANA` facts with zero `unrepresentable_state`; `StackObjectData` + v11 `stack` + text distinguish `from_prepared: true` original from `false` Codie copy. |
| Tinybones, opposing discarded land and spell with stash counters, second Tinybones, plus overlapping exile grant | Current controller plays foreign cards exiled by an earlier Tinybones; overlapping grant + stash counter emits no duplicates. Check own-turn timing, foreign land control (enters under Tinybones’s controller, not owner), ownership versus control, mana-of-any-type spending, High Fae Trickster non-bypass of own-turn timing, and `HandOnly` (blocks spell, allows land). Use Vizier's private top card and foretell's hidden exile as opposite-viewer controls. |
| Gandalf’s Flameshape vs. Foretell | Private face-down exile visible only to caster while controlling a Wizard; persists when Gandalf dies and another Wizard remains; hides and locks when last Wizard leaves; restores when a new Wizard enters; hidden cards stay in `unseen_objects` / `unseen_counts` for viewers who cannot see them (no open-decklist leak) and determinize via `fill_face_down_exile_slots` without entering `fill_foretold_slots`. |
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

**Decomposition.** Follow Muldrotha and Zul from the same graveyard object through payment, Quilled Greatwurm + Muldrotha overlap, Omniscience and Harbinger from hand (and Harbinger via free resolution cast), Tinybones from counter-marked exile, Gandalf from conditional face-down exile, and an Adventure, Modal BackFace, and linked prepare copy through stack and departure. One static-permission owner (`LayerBoard.play_permissions`), one route model (`PlayRoute`), one profile owner (`SpellProfile`), one `CastChoices` builder, and one open-ended pooled-mana fact path (`POOLED_MANA`) must explain all of them, including airbend, flashback, foretell, and modal regressions. Remove any exclusive old paths. Review repeated FRA spell programs for card-specific dispatch.

**Performance.** Use the 10% same-runner Engine Budget and the [stress-state protocol](../docs/development.md#stress-states). Add `bench_states` shapes for:
- Muldrotha with a graveyard of duplicate artifact creatures, overlapped by a Zul grant (now reading cached `StaticPlayPermissions.graveyard_type_sources` instead of scanning the battlefield per card).
- A hand of Dragon/Omen cards with restricted mana and several Charring Bite targets.
- Many Phytomedics with their own Seed Sutures, with one transfer and a zero-link control.

This is where to decide whether the linear grant scan needs an index. Measure first.
