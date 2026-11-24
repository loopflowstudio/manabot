# MTG-419 · Shared casting profiles and play permissions

Task steering, 2026-11-24: “Continue through straightforward cards without returning to review between cards. Realign and review the cohort together. Split only at a concrete shared-rules dependency or genuinely distinct implementation boundary; record that reason and the remaining rows.”

The Task brief requires “one shared casting-profile and play-permission model” and says “Preserve complete legal choices, observations, learning tokens, LLM text, replay/forks, and the 10% Engine Budget.” It also says to “Leave scoped event-history and contextual-query extensions to the history/queries task (defer cards strictly requiring new turn-history counters).” That task is MTG-418, which is open and not merged.

## Shared casting model

A play chooses an exact object, a printed profile and a legal route. The
profile supplies the spell's or land's characteristics, targets and printed
costs. The route supplies permission, timing and any alternative price.
Additional costs and reductions settle the total cost (CR 601.2f) before
any-type mana spending applies (CR 601.2h).

Built on this branch:

- **One casting authority (`CastChoices`).** Priority, resolution and `payment_preview` build it through `Game::cast_choices(player, object, profile, CastAuthority)` (`flow/play.rs`); `Decision::cast_authority()` binds `CastDuringResolution` offers. Flags are orthogonal: `resolving` for every resolution cast (CR 608.2g), `free` only for "without paying its mana cost" (CR 118.9), `flash_surcharge` only when `!resolving` and no other instant timing holds, `flashback` for stack-exit exile (CR 702.34a), `from_prepared` from `PlayGrant.prepared`. `SpellOnStack.{from_prepared, flashback}` reach the stack; `copy_stack_object` clears both. `StackObjectData`, v11 `stack::FROM_PREPARED` (38) / `stack::FLASHBACK` (39), LLM text and Academy `StackEntryView` / `Rail.svelte` project them.
- **Printed profiles (`SpellProfile`).** `ContentPack::spell_definition` resolves `Front`, `Adventure`, `Omen` or `BackFace`. Offers evaluate eligibility, announcement, payment and targeting against `cast.spell(...)`. A modal back enters back face up (`ZoneEntry { transformed: true }`). v11 action and relation rows carry one `SPELL_PROFILE` category (size 4).
- **Play routes (`PlayRoute`)**: `Hand`, `FreeFromHand`, `LibraryTop`, `PrintedFlashback`, `GrantedFlashback`, `Grant(usize)`, `GraveyardType`, `ExileCounter(CounterKind)`. One static owner, `LayerBoard.play_permissions` (`StaticPlayPermissions`, `flow/statics.rs`): `look_at_top`, `free_from_hand`, `opponents_hand_only`, `graveyard_flashback`, `library_spells`, `flash_spells`, `graveyard_type_sources` (Muldrotha, collected in the board scan), `exile_counter_plays` (Tinybones) and per-seat `extra_land_plays` (`AdditionalLandPlay` + `AdditionalLandPlayEachPlayer`; `LayerBoard.additional_land_plays_each_player` is gone). Cast-only-from-hand refuses the selected *spell* in `permits_spell`, so a modal card keeps its land routes; `cast_spell_action` rejects a land by the chosen profile.
- **Single-route wire compatibility.** `Action::{CastSpell, PlayLand}` keeps `route: None` for a lone route; `Game::resolved_play_route(action)` resolves it for observation, focus and description. Front-face `CastDuringResolution` offers stay `Action::SelectCard`; other profiles use `Action::CastSpell`.
- **Exact grants (`PlayGrant`, `flow/cast_permission.rs`)**: caster/liveness, timing, cost/payment, `foretold`, `on_adventure`, `flashback`, `prepared`, `face_down`, `condition` (checked in `live_grants`). `grant_base_cost` is the raw selected cost; `grant_cast_cost` is what observation and descriptions show.
- **One CR 601.2h owner.** `effective_spell_cost` applies `pay_with_any_mana()` once, after additional costs and reductions, for `ExileCounter`, `PlayGrant.any_mana` and `SpendAnyManaToCast` (`tla::azula_spends_any_mana_on_the_cost_after_reductions_and_kicker`).
- **Face-down exile privacy.** `is_face_down_in_exile` / `can_view_face_down_exile` / `hides_face_down_exile` own visibility (`is_foretold` now only means foretell cost and bearer slots). `hidden_face_down_exile(owner, viewer)` joins `unseen_objects` for both seats; `refill_hidden_exile_slots` fills foretold slots from foretell-bearers, then the rest, in one exchange of exile slots (`ZoneManager::exchange_face_down_exile`) and grants. `shuffle_known_library` resamples an owner's own hidden exile (fast path only when none). `HiddenHandConstraints` / `PossibleWorldSpace` carry `face_down_exile` (omitted at 0; weight × `P(r − F, D)`), `validate_source` compares viewer-hidden counts, and the constrained selection measure enumerates both slot kinds. `Effect::ExileTopCardsToPlay { face_down, condition }` + `PlayWindow::WhileExiled`; `Effect::exile_top_to_play` for the plain form.
- **Grant observability.** `CastPermissionData.{prepared, face_down, condition}` (condition as rules text) through JSON, bindings, experience projection and protocol schemas. `permission_facts` serializes one `PlayGrantTerms` (TLA tuple, later flags only when set, then a named `PlayGrantExtension`), so legacy programs keep their tokens. LLM text separates foretold from other face-down exile; Academy `permissionTerms` renders every term.
- **Zone-aware spending.** `SpendingContext::Cast { spell, from }`, `from` from `Game::casting_zone` (the origin zone while costs are paid, CR 601.2h). `ManaRestriction::ExceptSpellsFromHand`; every exclusion orders after narrower classes (`SpendOrder::Specific(usize::MAX)`). The four column restrictions stay in `PlayerData.restricted_mana`; any other restriction goes to `pooled_mana_classes` → `POOLED_MANA` facts (total still in `lesson_mana`). Exclusion wording reads `"for anything but ..."` across LLM text and Academy.
- **Activation conditions.** `ActivatedAbilityDefinition.condition` ("Activate only if", checked in `check_activation_terms` and the payment probe); `Condition::CastThisTurn(CardPredicate)` and `SpellsCostLess.first_on_own_turn` read profile-faithful history.
- **Profile-faithful cast history.** `TurnState.spells_cast: [Vec<RecordedCast>; 2]`, `RecordedCast { object, profile }`; `Game::recorded_cast_matches_predicate` reads the live stack appearance, else `ContentPack::recorded_spell` (inset or back face of the retained base). A front cast serializes as the bare slot, so existing state hashes hold.
- **Counters off the battlefield.** `GameState.object_counters: BTreeMap<ObjectId, Counters>` (exile only today), cleared by `TrackedState::mint_appearance`, hashed only when nonempty; `ZoneEntry.counters` carries into exile; `CardData.counters` → `Counter` fact rows (card rows), LLM text, Academy `LegacyCardView.counters`/inspect, protocol schemas. `KeywordAction::Discard` (hand → graveyard `CardMoved`), and a discard trigger's `Triggering` is the graveyard card (`Target::GraveyardCard` context).
- **Values and selectors added for the cohort.** `SpellCostReduction.generic: EffectValue` (fixed amounts keep their numeric serialization and tokens), evaluated per reducing permanent by `Game::static_value`; `EffectValue::AttachedCreaturePower`; `ObjectSelector::Matching { of, predicate }` (every matching object, no choice); `LookAndSelect.max_mana_value` (resolution-time bound).
- Omniscience replaces the mana cost but allows Kicker/Bargain (CR 118.9d) and mandatory additional costs; `AdditionalCost::FlashSurcharge` is skipped during resolution.

Planned (not built yet):

- **Profile-aware `Action::PlayLand`** (next slice item 1). Unreachable until a spell//land card exists.
- **Quilled Greatwurm's shared rules** (see "Then", item 2).

## Delivered

The per-set ledgers link each implemented row to its definition and native
fixture. The shared authority first shipped on this branch with Zul Ashur,
Muldrotha, Vizier, Etali, Smaug, Bilbo, Twinmaw and Scavenger Regent. The
first 44 preparation rows include linked copies, source transfers, preparation
during resolution and Emeritus of Truce.

- Yue offers Spew Flame from hand and Solstice Revelations offers Charring Bite from exile (`f3d84a7abe`, `be71091782`).
- Monica Rambeau // Photon uses one front/back definition pair. Photon pays its printed cost, enters back face up and returns to Monica on leaving.
- Foundations covers 11 of 17 rows: Omniscience (`FreeFromHand` + manual Kicker on Burst Lightning, deduplicated across copies), High Fae Trickster (timing), Harbinger of the Tides (surcharge + required sacrifices on free cast), Sphinx of Forgotten Lore (exact-object `flashback` grant alongside Think Twice's printed route), Reassembling Skeleton, Suspicious Shambler, Tinybones, Loot, and Redcap.
- Harmonized Trio, Joined Researchers, Codie, Heartwood Crafter // Soul Tether, and Sanar // Wild Idea bring preparation to 49 of 62 rows.
- HOB covers 14 of 17 Adventures (including Glamdring, Foe-hammer // Gleam of Death and Gandalf, Goblins' Bane // Flameshape) plus Bilbo's competing exile replacement. TDM covers 13 of 13 Omens (including Bloomvine Regent // Claim Territory).
- Shared casting authority alignment (`c5c415307f`, `cb30b69432`), Slice 1 stack provenance & open-ended `POOLED_MANA` (`badf9179d3`..`56046282a5`), and Slice 2 `RecordedCast` + `StaticPlayPermissions` consolidation + Tinybones/Glamdring/Loot/Redcap/Bloomvine (`ccc91c6c69`..`77173cf219`).
- Slice 3 (`a60e4570d7`..`940b60edf7`, plus `5da0c594f4`: airbend, Flameshape, foretell and Tinybones exile all go through `exile_with_play_grant(object, face_down, grant)`): D3 fixed (one CR 601.2h owner), face-down exile privacy with the atomic refill, full grant observability, Gandalf // Flameshape, and two stale guards repaired (`cr_106_mana::restrictions_on_any_subtype_pay_exactly_their_purposes` now reads the pooled class; two preparation fixtures no longer turn `skip_trivial` off).

Behavior reviews at `a805327074` and `592331aa12` verified the shared casting authority and Slice 1; D1 (`RecordedCast`), D2 (Clone state-hash pin), and consumer nits (`Rail.svelte` badges, exclusion wording) are fixed in `ccc91c6c69` and `24637208bd`.

Behavior review of Slice 2 at `89e6e53d4a` (`ccc91c6c69`..`77173cf219`; debug build, constructed fixtures, extension rebuilt from HEAD):

- **Met.** D1 is closed. `sanar_counts_a_resolved_adventure_and_omen_as_the_sorceries_cast` and `momo_still_discounts_the_first_flier_after_a_resolved_omen` read `RecordedCast` after the spell departs. Front casts keep their bare-slot hash. Tinybones stashes opposing discards with a public `Counter` fact row for both viewers, plus LLM text, `experience_projection` and Academy `counterTerms`. It offers Front and Adventure from exile only on its controller's turn. Drannith blocks those spells but not the land. Mana of any type pays the Adventure, and the foreign land enters under Tinybones's controller. A reverted probe found that stash counters survive a clone hash, `PossibleWorldSpace::materialize_index` (PreserveViewerRoot) and `determinize` for both viewers, with identical observations and action spaces. In the same probe, the Spew Flame cast through Tinybones went to the caster's adventure exile. Its creature was then castable by the caster for `{5}{R}{R}` but not with seven blue, so the adventure grant does not inherit any-type spending. Glamdring scales with equipped power and is 0 when unattached. Gleam returns every milled instant and sorcery. Loot's extra land play is per seat, and its MV bound is read at resolution. Redcap and Bloomvine pass. The Bloomvine ruling about simultaneous Dragons matches its per-object trigger. Consumer nits are fixed: `Rail.svelte` shows provenance tags, and `spend_phrase` / `spendPhrase` plus the native label read "for anything but spells from hand".
- **D3 (open, next slice item 1). Azula's `any_mana` grant converts colored pips before reductions.** Reverted probe in `tla.rs`: Azula exiles an opposing Lightning Strike, then Glamdring is equipped to our own Gray Ogre (no lands, empty pool). `can_cast` is true, and the Strike resolves for nothing (opponent at 17). It should cost one mana of any type: `{1}{R}` less 2 is `{R}` (CR 601.2f), and any-type spending applies after that (CR 601.2h). The cause is `grant_cost_for_spell` (`cast_permission.rs:496`), which applies `pay_with_any_mana()` before `effective_spell_cost` reduces. Tinybones's `ExileCounter` route already applies it at the end and is correct. Fix as planned. Use this probe as the regression.
- **Not reachable today: the modal-land `PlayLand` profile gap.** Monica // Photon is the only `modal_back` card, and its back face is a creature. The planned `PlayLand.profile` fix and regression need a spell//land MDFC definition first. It can be ordered after Gandalf.
- **Unexercised (carried in "Interaction evidence still owed").** A grant and a stash counter on the same object; a second Tinybones; a stashed Omen; Glamdring together with `ExileCounter`; Claim Territory with a single Forest (it enters tapped; no Oracle ruling found).
- **Controls.** Full debug `managym-cards --test cards`: 903 pass. Python validation controls: 101 pass, 1 skipped, with no break in `test_pinned_pilot`. Academy `inspect`/`view`/`mana` vitest: 298 pass. Slice 2 is satisfied except D3. The next slice owns D3, Gandalf and the remaining rows. Gate owns Clippy, the full workspace, the Engine Budget and `conformance/`.

## Remaining work

Thirteen MTG-419 rows remain: Quilled Greatwurm and 12 non-casting shared-rules rows. Gandalf and D3 landed in Slice 3; validation for it: full debug `managym-cards --test cards` (907 pass), `managym --lib` (356), `rules_tests` / `possible_worlds_tests` / `prepared_possible_worlds_tests` / `search_tests` / `engine_tests`, schema tests, Academy board vitest (477), and the Python controls below plus `tests/belief`, `tests/env/test_skip_forced_decisions.py` (extension rebuilt from `c01c99e5d9`).

### Next slice

1. **Profile-aware `Action::PlayLand`.** Add `#[serde(skip_serializing_if = "SpellProfile::is_front")] profile` to `Action::PlayLand`, pass it from `priority_actions_for_object`, validate `spell_definition(card, profile)` is a land in `play_land`, enter `transformed` for `BackFace`, and restrict `PlayRoute::FreeFromHand` to non-lands in `permits_spell`. It needs a spell//land definition to run: author a real Standard spell//land MDFC (a coverage row) or a rules fixture in `src/cards/fixtures.rs`; either shifts catalog ids (Academy fixtures, Clone hash pin, belief receipts). Regression as in the evidence table.
   The Clone state-hash pin (`lea::clone_copy_survives_suspended_entry_and_keeps_its_printed_base`) moves with every definition sorting before "Wolf Cove Villager"; the new definition will move it again. Replace the pin with direct assertions of the copied values (or a hash excluding catalog ordinals) in the same change rather than refreshing it a fifth time.
2. Tinybones evidence, then Quilled Greatwurm, then the 12 rows (below).

### Then

1. **Tinybones evidence not yet exercised** (built code, untested paths): an exact grant and a stash counter on the same exiled object (`exile_playable_objects` dedup), a second Tinybones re-enabling plays after the first leaves, a stashed Omen or modal back face, and opposite-viewer controls with Vizier/foretell. Full list in the evidence table.
2. **Quilled Greatwurm (FDN).** Needs two shared rules found absent: `TriggerCondition::DealsCombatDamage` matches only `CombatDamageToPlayers` (no creature recipients, no "that many" amount), and no `AdditionalCost` removes counters distributed among creatures. Add `PlayRoute::PrintedGraveyard` so the distributed `CostComponent::RemoveCounters` payment attaches only to its own graveyard permission, never to Muldrotha's or Zul's on the same card. Build the two rules with the card or split them with the history/queries work; record which.
3. **The 12 non-casting shared-rules rows.** None exercises a casting-profile or play-permission concept. Batch the small clean ones within the 10% Engine Budget; split any remaining family at its named rule, recording reason and rows (2026-11-24 steering):

| Cohort | Cards | Shared rule needed (outside casting/permission core) |
|---|---|---|
| Foundations (2) | Flamewake Phoenix, Gate Colossus | Graveyard-zone triggers (beginning-of-combat / landfall) retaining exact graveyard appearance through optional return. |
| Preparation (2) | Spiritcall Enthusiast, Kirol, History Buff | Batch-collapsed triggers ("one or more" entries or graveyard departures) in `process_event_batch`. |
| Preparation (3) | Grave Researcher, Emeritus of Ideation, Lluwen, Exchange Student | Surveil-before-count + MV life loss; distinct graveyard-card exile costs for attack trigger / sorcery activation + Pest token. |
| Preparation (2) | Variable Chaser, Pyre Rhymer | Separate APNAP optional whole-hand discard/draw; turn-scoped delayed mana trigger on tapping any Mountain for mana. |
| HOB (3) | An Unexpected Party, Bilbo, Luckwearer, Great Ugly-Looking Goblin | Chosen creature type retained on entry; shared-card-type control exchange; Amass (create or grow existing Army). |

4. Ledgers (`fdn.jsonl`, `hob.jsonl`, preparation) for each row as it lands.

### Deferred to MTG-418 (6 preparation rows)

MTG-418 owns six history-dependent rows: Emeritus of Woe and Bloodline Recollector (creature deaths this turn), Leech Collector (first life gain this turn), Scheming Silvertongue (total life gained this turn), Whiplash Wordsmith (noncombat damage to an opponent), and Hexhaven Dueling Arena (specific creature that attacked). On that merged revision, start with `cargo test --manifest-path managym/Cargo.toml -p managym --lib preparation_resolution` as the preparation control before adding the six directed fixtures.

## Interaction evidence still owed

| Position | Required fork or outcome |
|---|---|
| Modal spell//land DFC under Muldrotha / Tinybones / `HandOnly` / Omniscience | `Action::PlayLand { profile: BackFace }` executes cleanly, enters back face up, survives `HandOnly`, and is not duplicated by `FreeFromHand`. |
| Tinybones, opposing discarded land and spell (plus an Adventure/Omen) with stash counters, second Tinybones, plus overlapping exile grant | Current controller plays foreign cards exiled by an earlier Tinybones; overlapping grant + stash counter emits no duplicates; stashed Adventure/Omen is castable. Check own-turn timing, foreign land control (enters under Tinybones’s controller, not owner), ownership versus control, mana-of-any-type spending, High Fae Trickster non-bypass of own-turn timing, and `HandOnly` (blocks spell, allows land). Use Vizier's private top card and foretell's hidden exile as opposite-viewer controls. |
| Gandalf (remaining) | Gandalf dies while another Wizard remains (cards stay visible and playable); a controlled opponent's visible Flameshape card stays in Academy `agent.inspected` through the Python projection. The rest of the Flameshape row is covered by `hob::flameshape_*` and `test_flameshape_cards_show_face_down_with_their_condition`. |
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

Deferred check for gate: `tests/academy/test_play_modes.py::test_curated_combat_tape_is_identical_live_trace_and_inspector_fixture` fails on this branch (the new game stops at the opening Keep/Mulligan decision and `scenario_refresh` refuses it). CI does not run `test_play_modes.py`; whether main fails it too is unknown. Compare against main before publishing.

## Later phases

**Decomposition.** Follow Muldrotha and Zul from the same graveyard object through payment, Quilled Greatwurm + Muldrotha overlap, Omniscience and Harbinger from hand (and Harbinger via free resolution cast), Tinybones from counter-marked exile, Gandalf from conditional face-down exile (including Foretell coexistence), Azula + Glamdring, and an Adventure, Modal BackFace (spell and land), and linked prepare copy through stack and departure. One static-permission owner (`LayerBoard.play_permissions`), one route model (`PlayRoute`), one profile owner (`SpellProfile`), one `CastChoices` builder, one CR 601.2h `any_mana` owner, one atomic hidden-exile refill (`refill_hidden_exile_slots`), one profile-faithful cast history (`RecordedCast`), and one open-ended pooled-mana fact path (`POOLED_MANA`) must explain all of them, including airbend, flashback, foretell, and modal regressions. Remove any exclusive old paths. Review repeated FRA spell programs for card-specific dispatch.

**Performance.** Use the 10% same-runner Engine Budget and the [stress-state protocol](../docs/development.md#stress-states). Add `bench_states` shapes for:
- Muldrotha with a graveyard of duplicate artifact creatures, overlapped by a Zul grant (now reading cached `StaticPlayPermissions.graveyard_type_sources` instead of scanning the battlefield per card).
- A hand of Dragon/Omen cards with restricted mana and several Charring Bite targets.
- Many Phytomedics with their own Seed Sutures, with one transfer and a zero-link control.

This is where to decide whether the linear grant scan needs an index. Measure first.
