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
- **Exact grants (`PlayGrant`, `flow/cast_permission.rs`)**: caster/liveness, timing, cost/payment, `foretold`, `on_adventure`, `flashback`, `prepared`. Face-down exile visibility still goes through `is_foretold`.
- **Zone-aware spending.** `SpendingContext::Cast { spell, from }`, `from` from `Game::casting_zone` (the origin zone while costs are paid, CR 601.2h). `ManaRestriction::ExceptSpellsFromHand`; every exclusion orders after narrower classes (`SpendOrder::Specific(usize::MAX)`). The four column restrictions stay in `PlayerData.restricted_mana`; any other restriction goes to `pooled_mana_classes` → `POOLED_MANA` facts (total still in `lesson_mana`). Exclusion wording reads `"for anything but ..."` across LLM text and Academy.
- **Activation conditions.** `ActivatedAbilityDefinition.condition` ("Activate only if", checked in `check_activation_terms` and the payment probe); `Condition::CastThisTurn(CardPredicate)` and `SpellsCostLess.first_on_own_turn` read profile-faithful history.
- **Profile-faithful cast history.** `TurnState.spells_cast: [Vec<RecordedCast>; 2]`, `RecordedCast { object, profile }`; `Game::recorded_cast_matches_predicate` reads the live stack appearance, else `ContentPack::recorded_spell` (inset or back face of the retained base). A front cast serializes as the bare slot, so existing state hashes hold.
- **Counters off the battlefield.** `GameState.object_counters: BTreeMap<ObjectId, Counters>` (exile only today), cleared by `TrackedState::mint_appearance`, hashed only when nonempty; `ZoneEntry.counters` carries into exile; `CardData.counters` → `Counter` fact rows (card rows), LLM text, Academy `LegacyCardView.counters`/inspect, protocol schemas. `KeywordAction::Discard` (hand → graveyard `CardMoved`), and a discard trigger's `Triggering` is the graveyard card (`Target::GraveyardCard` context).
- **Values and selectors added for the cohort.** `SpellCostReduction.generic: EffectValue` (fixed amounts keep their numeric serialization and tokens), evaluated per reducing permanent by `Game::static_value`; `EffectValue::AttachedCreaturePower`; `ObjectSelector::Matching { of, predicate }` (every matching object, no choice); `LookAndSelect.max_mana_value` (resolution-time bound).
- Omniscience replaces the mana cost but allows Kicker/Bargain (CR 118.9d) and mandatory additional costs; `AdditionalCost::FlashSurcharge` is skipped during resolution.

Planned (not built yet — updated by post-Slice-2 concept review `scratch/concept-review.html`):

- **Single CR 601.2h owner for `any_mana` and profile-aware `Action::PlayLand`:** fix the two live casting bugs exposed on HEAD before expanding cards (`grant_base_cost` vs. `effective_spell_cost` for `PlayGrant.any_mana`; `Action::PlayLand.profile` + `play_land` + `FreeFromHand` land guard).
- **Atomic two-seat face-down exile privacy & full grant observability (Gandalf):** `PlayGrant.{face_down, condition}`, unified visibility helpers, one **atomic** `refill_hidden_exile_slots(owner, viewer, pool)` (replacing sequential two-pass refill), `knowledge.rs:180` fast-path guard, `experience_projection.py:481` `card.face_down` fix, `HiddenHandConstraints.face_down_exile` + `validate_source` + `selection_sampling` parity, named extension record in `permission_facts`, and complete `inspect.ts::permissionTerms` parity.
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
- HOB covers 13 of 17 Adventures (including Glamdring, Foe-hammer // Gleam of Death) plus Bilbo's competing exile replacement. TDM covers 13 of 13 Omens (including Bloomvine Regent // Claim Territory).
- Shared casting authority alignment (`c5c415307f`, `cb30b69432`), Slice 1 stack provenance & open-ended `POOLED_MANA` (`badf9179d3`..`56046282a5`), and Slice 2 `RecordedCast` + `StaticPlayPermissions` consolidation + Tinybones/Glamdring/Loot/Redcap/Bloomvine (`ccc91c6c69`..`77173cf219`).

Behavior reviews at `a805327074` and `592331aa12` verified the shared casting authority and Slice 1; D1 (`RecordedCast`), D2 (Clone state-hash pin), and consumer nits (`Rail.svelte` badges, exclusion wording) are fixed in `ccc91c6c69` and `24637208bd`.

## Remaining work

Fourteen MTG-419 rows remain: Gandalf, Quilled Greatwurm and 12 non-casting shared-rules rows, plus two small live casting bugs found by the post-Slice-2 concept review (`scratch/concept-review.html`).

### Next slice — Live casting fixes + Gandalf, Goblins' Bane // Flameshape (HOB): atomic two-seat face-down exile privacy

This slice closes the two live rules/contract bugs on HEAD first, then delivers the one remaining shared-invariant change (viewer privacy, determinization, possible worlds, and Academy grant parity).

1. **Unify `any_mana` at CR 601.2h (fixes live Azula + Glamdring / Kicker bug):**
   - Split raw base/alternative grant cost (`grant_base_cost`: `foretell` / `grant.cost` / `definition.mana_cost`, without `pay_with_any_mana()`) from observation-facing `grant_cast_cost` (preserving `CastPermissionData.cost`'s existing wire shape for retained TLA captures).
   - Call `grant_base_cost` in `casting_mana_cost` (`play.rs:447`), so CR 601.2f additional costs (`flash_surcharge`, `required_waterbend`, `replicate`, `kicker`) and generic cost reductions (`Glamdring`, affinity, etc.) operate on true colored pips.
   - Apply `cost.pay_with_any_mana()` **once** at the end of `effective_spell_cost` (`play.rs:544-584`) for all three sources together: `PlayRoute::ExileCounter(_)`, `cast.grant(self, object).is_some_and(|grant| grant.any_mana)`, and `StaticAbility::SpendAnyManaToCast`.
   - Regression: Azula-exiled `{1}{U}{U}` instant with Glamdring (`-{2}` generic) still costs 2 mana of any type (not `{1}`), and a spell with colored Kicker/Replicate under `PlayGrant.any_mana` pays the total cost with any mana.
2. **Make `Action::PlayLand` profile-aware (fixes live modal-land offer-vs-execute bug):**
   - Add `#[serde(skip_serializing_if = "SpellProfile::is_front")] profile: SpellProfile` to `Action::PlayLand` (`agent/action.rs:44`) — wire-identical for every front-face land — and return `*profile` from `Action::spell_profile()`.
   - Update `priority_actions_for_object` (`action.rs:701`) to pass `profile`, and `play_land` (`play.rs:587`) to accept `profile`, validate `spell_definition(card_ref, profile)` is a land and `permits_spell` accepts `(route, profile)`, and enter with `ZoneEntry { controller, transformed: profile == SpellProfile::BackFace, ..Default::default() }`.
   - In `permits_spell` (`play.rs:362`), restrict `PlayRoute::FreeFromHand => !cast.spell(&self.state[object]).types.is_land()` so Omniscience never duplicates land plays for a spell//land MDFC.
   - Regression: a `modal_back` spell//land card in hand (with Omniscience), graveyard (with Muldrotha), or exile (with Tinybones + Drannith Magistrate) offers one `PlayLand { profile: BackFace }` per legal route, executes cleanly, and enters back face up.
3. **Decouple face-down exile visibility from Foretell and refill slots atomically (Gandalf):**
   - Add `PlayGrant.face_down: bool` and `PlayGrant.condition: Option<Condition>` (checked in `live_grants` via `check_static_condition(cond, caster)`).
   - Replace the visibility uses of `is_foretold` (`zones.rs:690`, `continuation.rs:274`, `observation.rs:1538,1734`, `policy_history.rs:78`) with `is_face_down_in_exile(object)` and `can_view_face_down_exile(object, viewer)` (checks `shares_information(viewer, caster)` AND the grant's live `condition`, if any). Keep `is_foretold` strictly for Foretell cost and foretell-bearer slot eligibility.
   - `selection_sampling.rs::unseen_objects` includes every exiled card of `owner` with `is_face_down_in_exile(obj) && !can_view_face_down_exile(obj, viewer)`, **for `owner == viewer` and `owner != viewer` alike**.
   - Replace `fill_foretold_slots` with **one atomic `refill_hidden_exile_slots(owner, viewer, pool)`** (`cast_permission.rs`), called by `search.rs`, `possible_worlds.rs`, and `knowledge.rs::shuffle_known_library`:
     1. Snapshot all hidden face-down exile slots `(exile_vec_index, old_object, requires_foretell)` for `(owner, viewer)` before mutating anything.
     2. Choose pairwise-distinct replacements from `pool`: foretold slots take foretell-bearers first, then generic face-down slots take from remaining `pool` (returning `false` untouched if foretell-bearers are insufficient).
     3. Clear `zone` on vacated old objects, place every chosen replacement at its `exile_vec_index` with `zone = Some(Exile)`, and remap every `grant.object` in **one simultaneous lookup** against the `old_object -> new_object` map (preventing duplicate exile entries or merged grants when Foretell and Flameshape coexist).
   - Gate `knowledge.rs:180`'s `shuffle_known_library` fast path on `self.hidden_face_down_exile(owner, viewer).is_empty()` so normal turn-3 Flameshape boards (empty `library_facts`, no suspended decision) do not skip owner hidden-exile resampling.
   - `HiddenHandConstraints` / `PossibleWorldSpace` gain `#[serde(skip_serializing_if = "crate::state::is_zero")] face_down_exile: u32`. Both `HiddenHandConstraints::for_viewer` and `PossibleWorldSpace::validate_source` (`possible_worlds.rs:1192`) read `hidden_foretold_count(opponent, viewer)` and `hidden_face_down_exile_count(opponent, viewer)` (fixing the latent `shares_information` `validate_source` mismatch too). `selection_sampling.rs` reserves `foretold + face_down_exile` non-library slots when `commitments.constrains()`.
   - Fix `experience_projection.py:481` to inspect `card.face_down` (not `permission.foretold`) so a controlled opponent's visible Flameshape card is carried into `agent.inspected` instead of vanishing under `_redact_face_down_exile`.
4. **Full `CastPermissionData` observability across policy, text, and Academy:**
   - Project `prepared: bool` and `condition` (omitted when false/absent) on `CastPermissionData`, `observation_json.rs`, `python/bindings.rs`, `experience.rs`, `experience_projection.py`, and protocol schemas.
   - Replace the 4-branch positional-tuple ladder in `permission_facts` (`observation_encoder.rs:1989-2053`) with a named extension record for post-TLA fields (`prepared`, `condition`), keeping Hama's legacy token sequence byte-stable and ensuring singleton `prepared` or `condition.is_some()` grants are never skipped.
   - In `text_observation.py:952`, distinguish foretold (`"foretold, face down (hidden from other players)"`) from generic face-down exile (`"face down in exile (hidden from players who cannot look at it)"`) and render `prepared` + `condition`.
   - Complete `academy/ui/src/lib/board/inspect.ts::permissionTerms` (and `Inspect.svelte` face-down badge) to render every active `CastPermissionData` term (caster, cost, duration, own-turn, after-turn, flash, any-mana, waterbend, foretold, on-adventure, prepared, condition), matching `text_observation.py::_cast_permission`.
5. **Author Gandalf, Goblins' Bane // Flameshape (`hob.rs`)** and update `hob.jsonl` with all required interaction evidence (see table below, including the Foretell + Flameshape coexistence determinization regression).

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
| Azula (`PlayGrant.any_mana`) + Glamdring / Kicker | Exiled `{1}{U}{U}` instant with Glamdring (`-{2}` generic) still costs 2 mana of any type (CR 601.2f before CR 601.2h); colored additional costs under `PlayGrant.any_mana` accept any mana. |
| Modal spell//land DFC under Muldrotha / Tinybones / `HandOnly` / Omniscience | `Action::PlayLand { profile: BackFace }` executes cleanly, enters back face up, survives `HandOnly`, and is not duplicated by `FreeFromHand`. |
| Tinybones, opposing discarded land and spell (plus an Adventure/Omen) with stash counters, second Tinybones, plus overlapping exile grant | Current controller plays foreign cards exiled by an earlier Tinybones; overlapping grant + stash counter emits no duplicates; stashed Adventure/Omen is castable. Check own-turn timing, foreign land control (enters under Tinybones’s controller, not owner), ownership versus control, mana-of-any-type spending, High Fae Trickster non-bypass of own-turn timing, and `HandOnly` (blocks spell, allows land). Use Vizier's private top card and foretell's hidden exile as opposite-viewer controls. |
| Gandalf’s Flameshape vs. Foretell (including coexistence) | Cast Flameshape on turn 3 with no Wizard on board: cards are hidden from owner, counted in owner's `unseen_counts`, and resampled by `determinize(owner)` (verifying `knowledge.rs:180` does not short-circuit). With a Foretold card AND a Foretell-bearing Flameshape card in the same exile, `determinize` / `PossibleWorldSpace::materialize` keeps exile slots disjoint and grants intact (`refill_hidden_exile_slots`). When a Wizard enters, cards become visible (never labeled "foretold") and playable; persist when Gandalf dies and another Wizard remains; hide and lock when last Wizard leaves; restore when a new Wizard enters; controlled opponent's visible Flameshape card stays in Academy `agent.inspected`; hidden cards stay in `unseen_objects` / `unseen_counts` with `HiddenHandConstraints.face_down_exile` exact; Academy `inspect.ts` renders full permission terms. |
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

**Decomposition.** Follow Muldrotha and Zul from the same graveyard object through payment, Quilled Greatwurm + Muldrotha overlap, Omniscience and Harbinger from hand (and Harbinger via free resolution cast), Tinybones from counter-marked exile, Gandalf from conditional face-down exile (including Foretell coexistence), Azula + Glamdring, and an Adventure, Modal BackFace (spell and land), and linked prepare copy through stack and departure. One static-permission owner (`LayerBoard.play_permissions`), one route model (`PlayRoute`), one profile owner (`SpellProfile`), one `CastChoices` builder, one CR 601.2h `any_mana` owner, one atomic hidden-exile refill (`refill_hidden_exile_slots`), one profile-faithful cast history (`RecordedCast`), and one open-ended pooled-mana fact path (`POOLED_MANA`) must explain all of them, including airbend, flashback, foretell, and modal regressions. Remove any exclusive old paths. Review repeated FRA spell programs for card-specific dispatch.

**Performance.** Use the 10% same-runner Engine Budget and the [stress-state protocol](../docs/development.md#stress-states). Add `bench_states` shapes for:
- Muldrotha with a graveyard of duplicate artifact creatures, overlapped by a Zul grant (now reading cached `StaticPlayPermissions.graveyard_type_sources` instead of scanning the battlefield per card).
- A hand of Dragon/Omen cards with restricted mana and several Charring Bite targets.
- Many Phytomedics with their own Seed Sutures, with one transfer and a zero-link control.

This is where to decide whether the linear grant scan needs an index. Measure first.
