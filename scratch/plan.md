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
- **Play routes (`PlayRoute`)**: `Hand`, `FreeFromHand`, `LibraryTop`, `PrintedFlashback`, `GrantedFlashback`, `Grant(usize)`, `GraveyardType`, `ExileCounter(CounterKind)`. One static owner, `LayerBoard.play_permissions` (`StaticPlayPermissions`, `flow/statics.rs`): `look_at_top`, `free_from_hand`, `opponents_hand_only`, `graveyard_flashback`, `library_spells`, `flash_spells`, `graveyard_type_sources` (Muldrotha, collected in the board scan), `exile_counter_plays` (Tinybones) and per-seat `extra_land_plays` (`AdditionalLandPlay` + `AdditionalLandPlayEachPlayer`; `LayerBoard.additional_land_plays_each_player` is gone). Cast-only-from-hand refuses the selected *spell* in `permits_spell`, so a modal card keeps its land routes; `cast_spell_action` rejects a land by the chosen profile.
- **Single-route wire compatibility.** `Action::{CastSpell, PlayLand}` keeps `route: None` for a lone route; `Game::resolved_play_route(action)` resolves it for observation, focus and description. Front-face `CastDuringResolution` offers stay `Action::SelectCard`; other profiles use `Action::CastSpell`.
- **Exact grants (`PlayGrant`, `flow/cast_permission.rs`)**: caster/liveness, timing, cost/payment, `foretold`, `on_adventure`, `flashback`, `prepared`. Face-down exile visibility still goes through `is_foretold`.
- **Zone-aware spending.** `SpendingContext::Cast { spell, from }`, `from` from `Game::casting_zone` (the origin zone while costs are paid, CR 601.2h). `ManaRestriction::ExceptSpellsFromHand`; every exclusion orders after narrower classes (`SpendOrder::Specific(usize::MAX)`). The four column restrictions stay in `PlayerData.restricted_mana`; any other restriction goes to `pooled_mana_classes` → `POOLED_MANA` facts (total still in `lesson_mana`).
- **Activation conditions.** `ActivatedAbilityDefinition.condition` ("Activate only if", checked in `check_activation_terms` and the payment probe); `Condition::CastThisTurn(CardPredicate)` and `SpellsCostLess.first_on_own_turn` read profile-faithful history.
- **Profile-faithful cast history.** `TurnState.spells_cast: [Vec<RecordedCast>; 2]`, `RecordedCast { object, profile }`; `Game::recorded_cast_matches_predicate` reads the live stack appearance, else `ContentPack::recorded_spell` (inset or back face of the retained base). A front cast serializes as the bare slot, so existing state hashes hold.
- **Counters off the battlefield.** `GameState.object_counters: BTreeMap<ObjectId, Counters>` (exile only today), cleared by `TrackedState::mint_appearance`, hashed only when nonempty; `ZoneEntry.counters` carries into exile; `CardData.counters` → `Counter` fact rows (card rows), LLM text, Academy `LegacyCardView.counters`/inspect, protocol schemas. `KeywordAction::Discard` (hand → graveyard `CardMoved`), and a discard trigger's `Triggering` is the graveyard card (`Target::GraveyardCard` context).
- **Values and selectors added for the cohort.** `SpellCostReduction.generic: EffectValue` (fixed amounts keep their numeric serialization and tokens), evaluated per reducing permanent by `Game::static_value`; `EffectValue::AttachedCreaturePower`; `ObjectSelector::Matching { of, predicate }` (every matching object, no choice); `LookAndSelect.max_mana_value` (resolution-time bound).
- Omniscience replaces the mana cost but allows Kicker/Bargain (CR 118.9d) and mandatory additional costs; `AdditionalCost::FlashSurcharge` is skipped during resolution.

Planned (not built yet):

- **Complete two-seat face-down exile privacy (Gandalf):** `PlayGrant.face_down` and `condition: Option<Condition>`; `is_face_down_in_exile` / `can_view_face_down_exile(object, viewer)` replace visibility uses of `is_foretold`; hidden-exile resampling for **both** seats; `HiddenHandConstraints.face_down_exile`; `CastPermissionData` projects `prepared` and `condition`. Design under "Next slice" below.
- **Quilled Greatwurm's shared rules** (see "Then", item 2).

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
- **D1 (fixed in `ccc91c6c69`). Cast history lost the profile.** A reverted probe in `sos.rs` used Sanar and Smaug (`cast_profile(.., SpellProfile::Adventure)`), targeting an opposing Ogre. `offers_ability(sanar, 0)` is true while Spew Flame is on the stack and false once it resolves. After the stack appearance departs, `recorded_object_matches_predicate` reads `source_base`, which is the Creature front. `SpellsCostLess.first_on_own_turn` reads the same history. Fix: the planned `RecordedCast { object, profile }`. Use this probe as its regression, plus an Omen (Charring Bite) control.
- **D2 (fixed here).** `lea::clone_copy_survives_suspended_entry_and_keeps_its_printed_base` pinned a state hash from `f3d84a7abe`. Catalog additions since then (Codie included, before Slice 1) moved its definition ordinals. The pin is refreshed to `v19:5ab0103d…`, and the rest of the test passes. Full debug `managym-cards --test cards`: 897 tests pass. Plus the refreshed test, the earlier failure was this pin alone.
- **Consumer nits (fixed in `ccc91c6c69`).** Exclusion wording dropped the extra "only"; Academy's `Rail.svelte` shows `prepared`/`flashback` badges.
- **Controls.** Pass: the plan's five Rust controls, `heartwood`/`sanar`/`codie`/`harbinger`/`lone_graveyard`/`bilbo`, `managym --lib` (355) and the Validation Python controls (100 passed, U1952 pilot skipped, `test_pinned_pilot` shows no break). Slice 1 is satisfied. Gate owns Clippy, the full workspace, Engine Budget and `conformance/`.


## Remaining work

Fourteen MTG-419 rows remain: Gandalf, Quilled Greatwurm and 12 non-casting shared-rules rows. Built since Slice 1: `RecordedCast` with the Sanar Adventure/Omen regression and Momo first-flier control, exclusion wording without "only" and Rail `prepared`/`flashback` badges (`ccc91c6c69`); `StaticPlayPermissions` consolidation with profile-based hand-only refusal (`1bcaed4b62`); Tinybones (`d1e5558d19`); Glamdring (`083420edf3`); Loot and Redcap (`08ab9d7d0f`); Bloomvine Regent, TDM 13/13 (`40a4db9bc1`); their ledgers. Both D1 tests fail with the profile lookup reverted (debug build). The Clone state-hash pin moves with each new definition only through name-ordered catalog ordinals (checked once by removing Tinybones from `cards()`).

### Next slice — Gandalf, Goblins' Bane // Flameshape (HOB): two-seat face-down exile privacy

This teaches the most: it is the one remaining change to a shared invariant (viewer privacy and determinization), not a card on existing machinery.

- Decouple `PlayGrant.face_down: bool` and `PlayGrant.condition: Option<Condition>` from `PlayGrant.foretold`.
- Replace the visibility uses of `is_foretold` (`zones.rs`, `continuation.rs`, `observation.rs`, `policy_history.rs`) with `is_face_down_in_exile` and `can_view_face_down_exile(object, viewer)` (both `shares_information` and the live `condition`). `is_foretold` stays for the Foretell cost and determinization.
- `selection_sampling.rs::unseen_objects` includes every exiled card of `owner` with `is_face_down_in_exile(obj) && !can_view_face_down_exile(obj, viewer)`, **for `owner == viewer` and `owner != viewer` alike**.
- One hidden-exile resampling path for both seats in `search.rs`, `possible_worlds.rs` and `knowledge.rs::shuffle_known_library`: pool `hidden_face_down_exile(owner, viewer)` with the unknown library, run `fill_foretold_slots` (foretell-bearers only) then `fill_face_down_exile_slots`.
- `HiddenHandConstraints` / `PossibleWorldSpace` gain `face_down_exile: u32` (non-foretold hidden exile, skipped when zero); `validate_source` checks it and library slots subtract `foretold + face_down_exile`.
- `CastPermissionData` projects `prepared` and `condition` (omitted when false/absent); `permission_facts` (`observation_encoder.rs`) emits conditional/prepared singleton grants through a named extension record, keeping Hama's token sequence byte-stable; text distinguishes generic face-down exile from foretold (`"face down in exile (hidden from players who cannot look at it)"`) and renders `condition`/`prepared` in `text_observation.py` and Academy `inspect.ts`.
- Evidence: see the Gandalf row of "Interaction evidence still owed".

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
