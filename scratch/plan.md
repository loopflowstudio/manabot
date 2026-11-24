# MTG-419 · Shared casting profiles and play permissions

Task steering, 2026-11-24: “Continue through straightforward cards without returning to review between cards. Realign and review the cohort together. Split only at a concrete shared-rules dependency or genuinely distinct implementation boundary; record that reason and the remaining rows.”

The Task brief requires “one shared casting-profile and play-permission model” and says “Preserve complete legal choices, observations, learning tokens, LLM text, replay/forks, and the 10% Engine Budget.” It also says to “Leave scoped event-history and contextual-query extensions to the history/queries task (defer cards strictly requiring new turn-history counters).” That task is MTG-418, which is open and not merged.

## Shared casting model

A play chooses an exact object, a printed profile and a legal route: `(object, profile, route)`. The zone opens candidate routes (`play_routes`); the chosen profile (`Front`, `Adventure`, `Omen`, `BackFace`) supplies the spell's or land's characteristics, targets and printed costs; `permits_spell` is the single authority on whether a route permits that profile (for both `CastSpell` and `PlayLand`). Additional costs and reductions settle the total cost (CR 601.2f) before any-type mana spending applies (CR 601.2h).

Built on this branch (`88ce96bd6b..307465b649`):

- **One casting authority (`CastChoices`).** Priority, resolution and `payment_preview` build it through `Game::cast_choices(player, object, profile, CastAuthority)` (`flow/play.rs`); `Decision::cast_authority()` binds `CastDuringResolution` offers. Flags are orthogonal: `resolving` for every resolution cast (CR 608.2g), `free` only for "without paying its mana cost" (CR 118.9), `flash_surcharge` only when `!resolving` and no other instant timing holds, `flashback` for stack-exit exile (CR 702.34a), `from_prepared` from `PlayGrant.prepared`. `SpellOnStack.{from_prepared, flashback}` reach the stack; `copy_stack_object` clears both. `StackObjectData`, v11 `stack::FROM_PREPARED` (38) / `stack::FLASHBACK` (39), LLM text and Academy `StackEntryView` / `Rail.svelte` project them.
- **Printed profiles (`SpellProfile`).** `ContentPack::spell_definition` resolves `Front`, `Adventure`, `Omen` or `BackFace`. Offers evaluate eligibility, announcement, payment and targeting against `cast.spell(...)`. A modal back enters back face up (`ZoneEntry { transformed: true }`). v11 action and relation rows carry one `SPELL_PROFILE` category (size 4).
- **Play routes (`PlayRoute`)**: `Hand`, `FreeFromHand`, `LibraryTop`, `PrintedFlashback`, `GrantedFlashback`, `Grant(usize)`, `GraveyardType`, `ExileCounter(CounterKind)`. One static owner, `LayerBoard.play_permissions` (`StaticPlayPermissions`, `flow/statics.rs`): `look_at_top`, `free_from_hand`, `opponents_hand_only`, `graveyard_flashback`, `library_spells`, `flash_spells`, `graveyard_type_sources` (Muldrotha, collected in the board scan), `exile_counter_plays` (Tinybones) and per-seat `extra_land_plays` (`AdditionalLandPlay` + `AdditionalLandPlayEachPlayer`). Cast-only-from-hand refuses the selected *spell* in `permits_spell`, so a modal card keeps its land routes; `cast_spell_action` rejects a land by the chosen profile.
- **Single-route wire compatibility.** `Action::{CastSpell, PlayLand}` keeps `route: None` for a lone route; `Game::resolved_play_route(action)` resolves it for observation, focus and description. Front-face `CastDuringResolution` offers stay `Action::SelectCard`; other profiles use `Action::CastSpell`.
- **Exact grants (`PlayGrant`, `flow/cast_permission.rs`)**: caster/liveness, timing, cost/payment, `foretold`, `on_adventure`, `flashback`, `prepared`, `face_down`, `condition` (checked in `live_grants`). `grant_base_cost` is the raw selected cost; `grant_cast_cost` is what observation and descriptions show. Airbend, Flameshape, foretell, Redcap, and linked exile share `exile_with_play_grant`.
- **One CR 601.2h owner.** `effective_spell_cost` applies `pay_with_any_mana()` once, after additional costs and reductions, for `ExileCounter`, `PlayGrant.any_mana` and `SpendAnyManaToCast` (`tla::azula_spends_any_mana_on_the_cost_after_reductions_and_kicker`).
- **Face-down exile privacy.** `is_face_down_in_exile` / `can_view_face_down_exile` / `hides_face_down_exile` own visibility (`is_foretold` now only means foretell cost and bearer slots). `hidden_face_down_exile(owner, viewer)` joins `unseen_objects` for both seats; `refill_hidden_exile_slots` fills foretold slots from foretell-bearers, then the rest, in one exchange of exile slots (`ZoneManager::exchange_face_down_exile`) and grants. `shuffle_known_library` resamples an owner's own hidden exile (fast path only when none). `HiddenHandConstraints` / `PossibleWorldSpace` carry `face_down_exile` (omitted at 0; weight × `P(r − F, D)`), `validate_source` compares viewer-hidden counts, and the constrained selection measure enumerates both slot kinds. `Effect::ExileTopCardsToPlay { face_down, condition }` + `PlayWindow::WhileExiled`; `Effect::exile_top_to_play` for the plain form.
- **Grant observability.** `CastPermissionData.{prepared, face_down, condition}` (condition as rules text) through JSON, bindings, experience projection and protocol schemas. `permission_facts` serializes one `PlayGrantTerms` (TLA tuple, later flags only when set, then a named `PlayGrantExtension`), so legacy programs keep their tokens. LLM text separates foretold from other face-down exile; Academy `permissionTerms` renders every term.
- **Zone-aware spending.** `SpendingContext::Cast { spell, from }`, `from` from `Game::casting_zone` (the origin zone while costs are paid, CR 601.2h). `ManaRestriction::ExceptSpellsFromHand`; every exclusion orders after narrower classes (`SpendOrder::Specific(usize::MAX)`). The four column restrictions stay in `PlayerData.restricted_mana`; any other restriction goes to `pooled_mana_classes` → `POOLED_MANA` facts (total still in `lesson_mana`). Exclusion wording reads `"for anything but ..."` across LLM text and Academy.
- **Activation conditions.** `ActivatedAbilityDefinition.condition` ("Activate only if", checked in `check_activation_terms` and the payment probe); `Condition::CastThisTurn(CardPredicate)` and `SpellsCostLess.first_on_own_turn` read profile-faithful history.
- **Profile-faithful cast history.** `TurnState.spells_cast: [Vec<RecordedCast>; 2]`, `RecordedCast { object, profile }`; `Game::recorded_cast_matches_predicate` reads the live stack appearance, else `ContentPack::recorded_spell` (inset or back face of the retained base). A front cast serializes as the bare slot, so existing state hashes hold.
- **Counters off the battlefield.** `GameState.object_counters: BTreeMap<ObjectId, Counters>` (exile only today), cleared by `TrackedState::mint_appearance`, hashed only when nonempty; `ZoneEntry.counters` carries into exile; `CardData.counters` → `Counter` fact rows (card rows), LLM text, Academy `LegacyCardView.counters`/inspect, protocol schemas. `KeywordAction::Discard` (hand → graveyard `CardMoved`), and a discard trigger's `Triggering` is the graveyard card (`Target::GraveyardCard` context).
- **Values and selectors added for the cohort.** `SpellCostReduction.generic: EffectValue` (fixed amounts keep their numeric serialization and tokens), evaluated per reducing permanent by `Game::static_value`; `EffectValue::AttachedCreaturePower`; `ObjectSelector::Matching { of, predicate }` (every matching object, no choice); `LookAndSelect.max_mana_value` (resolution-time bound).
- Omniscience replaces the mana cost but allows Kicker/Bargain (CR 118.9d) and mandatory additional costs; `AdditionalCost::FlashSurcharge` is skipped during resolution.

Remaining model completions (next two slices, detailed below):

- **Unified `(object, profile, route)` across `CastSpell` and `PlayLand`.** Profile-aware `Action::PlayLand`, `permits_spell` as the sole profile-vs-route authority (removing front-face `land` pre-filters in `play_routes`), and full Academy play-tray / `agent.inspected` inspection plus `library_top_departure` on top-library exiles.
- **Printed graveyard self-permission (`PlayRoute::PrintedGraveyard`) for Quilled Greatwurm**, with a route-attached distributed `CostComponent::RemoveCounters` payment and creature combat-damage trigger amount.

## Delivered

The per-set ledgers link each implemented row to its definition and native fixture. The shared authority first shipped on this branch with Zul Ashur, Muldrotha, Vizier, Etali, Smaug, Bilbo, Twinmaw and Scavenger Regent. The first 44 preparation rows include linked copies, source transfers, preparation during resolution and Emeritus of Truce.

- Yue offers Spew Flame from hand and Solstice Revelations offers Charring Bite from exile (`f3d84a7abe`, `be71091782`).
- Monica Rambeau // Photon uses one front/back definition pair. Photon pays its printed cost, enters back face up and returns to Monica on leaving.
- Foundations covers 11 of 17 rows: Omniscience (`FreeFromHand` + manual Kicker on Burst Lightning, deduplicated across copies), High Fae Trickster (timing), Harbinger of the Tides (surcharge + required sacrifices on free cast), Sphinx of Forgotten Lore (exact-object `flashback` grant alongside Think Twice's printed route), Reassembling Skeleton, Suspicious Shambler, Tinybones, Loot, and Redcap.
- Harmonized Trio, Joined Researchers, Codie, Heartwood Crafter // Soul Tether, and Sanar // Wild Idea bring preparation to 49 of 62 rows.
- HOB covers 14 of 17 Adventures (including Glamdring, Foe-hammer // Gleam of Death and Gandalf, Goblins' Bane // Flameshape) plus Bilbo's competing exile replacement. TDM covers 13 of 13 Omens (including Bloomvine Regent // Claim Territory).
- Shared casting authority alignment (`c5c415307f`, `cb30b69432`), Slice 1 stack provenance & open-ended `POOLED_MANA` (`badf9179d3`..`56046282a5`), Slice 2 `RecordedCast` + `StaticPlayPermissions` consolidation + Tinybones/Glamdring/Loot/Redcap/Bloomvine (`ccc91c6c69`..`77173cf219`), and Slice 3 (`a60e4570d7`..`5da0c594f4`: D3 fixed, face-down exile privacy with atomic refill, full grant observability, `exile_with_play_grant`, Gandalf // Flameshape, stale guard repairs).

Validation at Slice 3 (`c01c99e5d9`..`307465b649`): full debug `managym-cards --test cards` (907 pass), `managym --lib` (356), `rules_tests` / `possible_worlds_tests` / `prepared_possible_worlds_tests` / `search_tests` / `engine_tests`, schema tests, Academy board vitest (477), and the Python controls below plus `tests/belief`, `tests/env/test_skip_forced_decisions.py`.

## Remaining work

Post-Slice-3 concept review (`307465b649`, recorded in `scratch/concept-review.html`) keeps **two concrete implementation slices** in MTG-419 — closing the last shared casting/permission model gaps, Academy tray parity, remaining interaction evidence, and **Quilled Greatwurm** (`PlayRoute::PrintedGraveyard`) — and **splits the 12 non-casting rows** into separate Tasks at their concrete shared-rules boundaries (per the 2026-11-24 steering, alongside the 6 MTG-418 history rows).

### Slice 4 (next): Unified `(object, profile, route)` play, Academy tray parity, library-top knowledge & interaction evidence

1. **Profile-aware `Action::PlayLand` and one profile-vs-route authority (`permits_spell`).**
   - `managym/src/agent/action.rs`: add `#[serde(skip_serializing_if = "SpellProfile::is_front")] profile: SpellProfile` to `Action::PlayLand` and return it from `Action::spell_profile()`.
   - `managym/src/flow/play.rs`:
     - In `play_routes`: replace front-face `!land` guards with `let has_nonland = !land || self.state[object].modal_back;` for `FreeFromHand` (line 239) and `LibraryTop` (line 328), so plain lands stay on lone `Hand` (100% wire-compatible) while Land//Spell and Spell//Land MDFCs expose their legal routes; drop `|| (land && kind != Land)` in Muldrotha's `GraveyardType` loop (line 307) so Artifact Lands can use the Artifact slot (CR 305.9 / Muldrotha ruling) and Land//Creature MDFCs expose both faces.
     - In `permits_spell`: restrict `PlayRoute::FreeFromHand` to `!cast.spell(&self.state[object]).types.is_land()` (line 362).
     - In `play_land`: take `profile: SpellProfile`, build `cast` via `cast_choices(player, object, profile, CastAuthority::Priority(route))`, validate both `cast.spell(...).types.is_land()` and `self.permits_spell(player, object, cast)` before spending the route or land play, and enter with `ZoneEntry { controller, transformed: profile == SpellProfile::BackFace, ..Default::default() }`.
     - In `effective_spell_cost`: read `self.casting_zone(object) != ZoneType::Hand` for `SpellsFromOutsideHandCostLess` (line 555), matching `SpendingContext::Cast`.
   - `managym/src/flow/action.rs` & `managym/src/agent/structured_offer.rs`: pass `profile` into `Action::PlayLand` in `priority_actions_for_object`; extend `action_description` (line 109) to label lone-route non-front `Action::PlayLand` (`"Play <BackFace>"`); use `spell_definition(&self.state[*object], *profile).name` for `PublicCommitment::PlayLand` (`structured_offer.rs:659`).
   - Add minimal rules fixture(s) in `managym/src/cards/fixtures.rs` covering both Spell//Land and Land//Creature MDFCs (registered in the default pack, never deck-admitted, matching `Crossroads of Destiny`). Refresh Academy fixture digests (`scripts/refresh_academy_fixtures.py`) and **replace the churning Clone state-hash pin** (`lea::clone_copy_survives_suspended_entry_and_keeps_its_printed_base`) with direct assertions of the copied values (or a hash excluding catalog ordinals) in the same change.
2. **Academy tray card inspection & `agent.inspected` parity.**
   - `academy/ui/src/lib/board/Board.svelte`: include `...(player.inspected ?? [])` in `cards` (line 574) and `textOf` (line 606), matching `view.ts:165`.
   - In `Board.svelte`'s play tray (line 812): resolve `CardState` from `item.placed.anchor.kind === 'tray' && item.placed.anchor.id !== null ? cards.get(item.placed.anchor.id) ?? null : null` and call `previewCard(trayCard, element)` when present (falling back to `previewName` for sideboard/non-card tray items), so hovering any playable exiled, graveyard, library-top, or controlled-opponent `inspected` card displays its `permissionTerms` and `counterTerms` right in `<Inspect>`. Add vitest coverage for tray inspection (Tinybones stash counter, Flameshape condition, and controlled opponent `agent.inspected` card).
3. **Top-of-library exile knowledge & condition vocabulary walker.**
   - `managym/src/flow/cast_permission.rs`: let `exile_with_play_grant` accept `ZoneEntry` (or `library_top_departure: bool`) so `exile_top_cards_to_play` sets `library_top_departure: true` (matching `draw_cards` in `zones.rs:294`). Add a directed test verifying Scry + Redcap/Flameshape preserves shifted `known_library` positions under `assert_library_knowledge()` and `determinize()`.
   - `managym/src/agent/record_vocabulary.rs`: replace the top-level-only `play_grant_conditions` walker (line 304) with a recursive walk over `objects(&json)` (matching `set_names` and `combat_restrictions`) so conditions inside triggers, activations, or `Effect::If` never miss registration.
4. **Remaining Tinybones, Gandalf, and MDFC interaction evidence** (all rows in the evidence table below except Quilled Greatwurm).

### Slice 5: Quilled Greatwurm (FDN) — `PlayRoute::PrintedGraveyard`

Quilled Greatwurm is the last untested play-permission axis in MTG-419: a printed graveyard self-permission with a route-attached additional cost that must coexist cleanly with Muldrotha and Zul on the same card.

1. **`PlayRoute::PrintedGraveyard`** (`flow/cast_permission.rs`, `play.rs`, `action.rs`, `payment.rs`, `observation.rs`):
   - Printed graveyard-cast permission on `PrintedCharacteristics` carrying its route-specific additional cost (`CostComponent::RemoveCounters { count, kind, predicate }`, distributed among controlled permanents matching `predicate`, analogous to `Decision::DistributeCounters`).
   - `play_routes` emits `PlayRoute::PrintedGraveyard` for the card in its owner's graveyard; `cost_components` and `can_pay_cast` attach the counter-removal cost **only** when `cast.route == Some(PlayRoute::PrintedGraveyard)` (never on Muldrotha's `GraveyardType { Creature }` or Zul's `Grant` on the same card).
   - `action_description` labels the route clearly; payment choices, `PaymentPreview`, LLM text, and Academy offers handle distributed counter removal cleanly.
2. **Creature combat-damage trigger with damage amount** (`state/src/ability.rs`, `flow/triggers.rs`, `flow/resolution.rs`):
   - Extend `TriggerCondition::DealsCombatDamage` to match combat damage dealt to creatures as well as players and pass the damage amount ("put that many +1/+1 counters on that creature") to the triggered frame.
3. **Author Quilled Greatwurm** in `managym/cards/src/fdn.rs`, update `fdn.jsonl`, and prove the Quilled Greatwurm + Muldrotha + Zul overlap fixture in `managym/cards/tests/cards/fdn.rs` (normal cast, graveyard cast removing six +1/+1 counters distributed across multiple creatures, Muldrotha casting it without removing counters while spending the Creature quota, and insufficient counters blocking `PrintedGraveyard` while keeping Muldrotha open).

### Split out of MTG-419 (recorded boundaries & deferred rows)

Per the 2026-11-24 steering (*“Split only at a concrete shared-rules dependency or genuinely distinct implementation boundary; record that reason and the remaining rows”*), the **12 non-casting rows** below exercise zero new casting-profile or play-permission concepts and each sits behind a distinct shared-rules boundary outside this Task. Record them as follow-up Tasks rather than adding five unrelated engine subsystems to MTG-419:

| Follow-up boundary | Rows (12 total) | Concrete shared-rules boundary (outside casting/permission core) |
|---|---|---|
| Graveyard-zone triggers | FDN (2): Flamewake Phoenix, Gate Colossus | Graveyard-zone triggered abilities (beginning-of-combat / landfall) retaining exact graveyard appearance through optional return + combat attack/block constraints. |
| Batch-collapsed triggers | Prep (2): Spiritcall Enthusiast, Kirol, History Buff | Batch-collapsed “one or more” entry/graveyard-departure triggers in `process_event_batch` (plus Kirol’s tap-two-creatures triggered-ability copy). |
| Surveil/mill-result queries & graveyard exile costs | Prep (3): Grave Researcher, Emeritus of Ideation, Lluwen, Exchange Student | Surveil-before-count + MV life loss; mill-result library placement; distinct graveyard-card exile costs + Pest token. |
| APNAP wheel & delayed mana triggers | Prep (2): Variable Chaser // Arc of Fortune, Pyre Rhymer // Molten Tide | Separate APNAP optional whole-hand discard/draw; turn-scoped delayed mana trigger on tapping any Mountain for mana (`Pyre Rhymer` also depends on `03_event_history`). |
| Permanent state, control exchange & Amass | HOB (3): An Unexpected Party // At the Door, Bilbo, Luckwearer // Burglar's Plot, Great Ugly-Looking Goblin // Clap! Snap! | Chosen creature type retained on entry (`battlefield-forms`); shared-card-type permanent control exchange (`layers-attachments`); Amass (create or grow existing Army, `scoped-results`). |

**Deferred to MTG-418 (6 preparation rows):** Emeritus of Woe and Bloodline Recollector (creature deaths this turn), Leech Collector (first life gain this turn), Scheming Silvertongue (total life gained this turn), Whiplash Wordsmith (noncombat damage to an opponent), and Hexhaven Dueling Arena (specific creature that attacked). On that merged revision, start with `cargo test --manifest-path managym/Cargo.toml -p managym --lib preparation_resolution` as the preparation control before adding the six directed fixtures.

## Interaction evidence still owed

| Position | Required fork or outcome |
|---|---|
| Modal Spell//Land and Land//Creature MDFC fixtures under Hand / Omniscience / Muldrotha / Vizier / Tinybones / `HandOnly` | `Action::PlayLand { profile: BackFace }` executes cleanly, enters back face up, labels the back face, survives `HandOnly`, and is never duplicated by `FreeFromHand`; Land//Creature exposes `FreeFromHand`, `GraveyardType { Creature }`, and `LibraryTop` for its back face; plain lands keep lone `Hand` with `profile` and `route` omitted on the wire; malformed `PlayLand` routes are rejected by `permits_spell`. |
| Tinybones, opposing discarded land and spell (plus an Adventure/Omen) with stash counters, second Tinybones, plus overlapping exile grant | Current controller plays foreign cards exiled by an earlier Tinybones; overlapping grant + stash counter emits no duplicates (`exile_playable_objects` dedup); stashed Adventure/Omen is castable; Glamdring combines with `ExileCounter`; own-turn timing, foreign land control (enters under Tinybones’s controller, not owner), High Fae Trickster non-bypass of own-turn timing, and `HandOnly` (blocks spell, allows land). Use Vizier’s private top card and foretell’s hidden exile as opposite-viewer controls. |
| Gandalf (remaining) & Scry + top-library exile | Gandalf dies while another Wizard remains (cards stay visible and playable); controlled opponent’s visible Flameshape card stays in `agent.inspected` through Python projection and renders rules text + `permissionTerms` in Academy `Board.svelte` tray hover; Scry + Redcap/Flameshape shifts remaining `known_library` positions cleanly under `assert_library_knowledge()` and `determinize()`. |
| Quilled Greatwurm + Muldrotha + Zul | Overlapping `PrintedGraveyard` (removes six +1/+1 counters distributed among controlled creatures, preserves Muldrotha quota) vs. `GraveyardType { Creature }` (no counter removal, spends Muldrotha quota) vs. Zul `Grant` (no counter removal, preserves Muldrotha quota); insufficient counters closes only `PrintedGraveyard`; combat damage to creatures puts that many +1/+1 counters on the creature. |

## Validation

**Consumer checks per slice.** Compare native observations for both viewers with the Python bindings, semantic capture/LLM text and Academy offers at priority, at mid-announcement (`AnnounceX`/`ChooseTargets`/`Pay`), with a spell on the stack, and at a nested decision. Use v11 rows and learning tokens. Keep hidden information out of the other viewer's data (including `unseen_counts` and determinization). Fork-and-replay checks need exact preconditions and stale rejection. Existing controls:

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards airbend_permission
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards flashback_is_exiled_when_countered_after_iroh_leaves
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards sozins_comet_foretell_spending_choices_preserve_color_and_private_identity
cargo test --manifest-path managym/Cargo.toml -p managym --lib preparation_resolution
uv run pytest tests/semantic/test_text_observation.py tests/semantic/test_restricted_mana_input.py tests/semantic/test_pooled_mana_classes_input.py tests/academy/test_choice_navigation.py tests/model/test_pinned_pilot.py -q
```

Refresh Academy fixture digests with `scripts/refresh_academy_fixtures.py` after adding the MDFC fixture(s) and Quilled Greatwurm. The conformance receipts have their own writers and are checked by gate for the final catalog.

Gate owns the full debug workspace, Clippy and the measured 10% Engine Budget.

Deferred check for gate: `tests/academy/test_play_modes.py::test_curated_combat_tape_is_identical_live_trace_and_inspector_fixture` fails on this branch (the new game stops at the opening Keep/Mulligan decision and `scenario_refresh` refuses it). CI does not run `test_play_modes.py`; compare against main before publishing.

## Later phases

**Decomposition.** Follow Muldrotha and Zul from the same graveyard object through payment, Quilled Greatwurm + Muldrotha + Zul overlap, Omniscience and Harbinger from hand (and Harbinger via free resolution cast), Tinybones from counter-marked exile, Gandalf from conditional face-down exile (including Foretell coexistence and Scry shift), Azula + Glamdring, and an Adventure, Modal BackFace (both Spell//Land and Land//Creature), and linked prepare copy through stack and departure. One static-permission owner (`LayerBoard.play_permissions`), one route model (`PlayRoute`), one profile owner (`SpellProfile`), one profile-vs-route gate (`permits_spell`), one `CastChoices` builder, one CR 601.2h `any_mana` owner, one atomic hidden-exile refill (`refill_hidden_exile_slots`), one profile-faithful cast history (`RecordedCast`), and one open-ended pooled-mana fact path (`POOLED_MANA`) must explain all of them, including airbend, flashback, foretell, and modal regressions. Remove any exclusive old paths. Review repeated FRA spell programs for card-specific dispatch.

**Performance.** Use the 10% same-runner Engine Budget and the [stress-state protocol](../docs/development.md#stress-states). Add `bench_states` shapes for:
- Muldrotha with a graveyard of duplicate artifact creatures, overlapped by a Zul grant (now reading cached `StaticPlayPermissions.graveyard_type_sources` instead of scanning the battlefield per card).
- A hand of Dragon/Omen cards with restricted mana and several Charring Bite targets.
- Many Phytomedics with their own Seed Sutures, with one transfer and a zero-link control.

This is where to decide whether the linear grant scan needs an index. Measure first.
