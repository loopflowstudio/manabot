# MTG-419 · Shared casting profiles and play permissions

Task steering, 2026-11-24: “Continue through straightforward cards without returning to review between cards. Realign and review the cohort together. Split only at a concrete shared-rules dependency or genuinely distinct implementation boundary; record that reason and the remaining rows.”

The Task brief requires “one shared casting-profile and play-permission model” and says “Preserve complete legal choices, observations, learning tokens, LLM text, replay/forks, and the 10% Engine Budget.” It also says to “Leave scoped event-history and contextual-query extensions to the history/queries task (defer cards strictly requiring new turn-history counters).” That task is MTG-418, which is open and not merged.

## Shared casting model

A play chooses an exact object, a printed profile and a legal route. The
profile supplies the spell's characteristics, targets and printed costs. The
route supplies permission, timing and any alternative price. Additional costs
apply to the selected combination.

- **One casting authority (`CastChoices`).** Priority and resolution plays construct `CastChoices` through `Game::cast_choices` with a `CastAuthority` (`Resolution` carries its offer and payment, built by `Decision::cast_authority`, so the builder fails only for a missing profile), using orthogonal flags: `resolving: bool` is true for *all* `CastDuringResolution` casts (both free and paid, CR 608.2g), `free: bool` strictly means "without paying its mana cost" (CR 118.9), `flash_surcharge: bool` is required only when `!resolving` and no other instant timing holds, `flashback: bool` marks stack-exit exile (CR 702.34a), and `from_prepared: bool` captures prepared-copy provenance from `PlayGrant.prepared` before the copy leaves exile. `SpellOnStack.from_prepared` carries that provenance onto the stack and `copy_stack_object` clears it (`false`) on stack copies alongside `flashback` and `exile_instead_of_graveyard`.
- **Printed profiles (`SpellProfile`).** `ContentPack::spell_definition` resolves `Front`, `Adventure`, `Omen`, or `BackFace` for the physical object. Priority and resolution offers evaluate eligibility, land vs. spell type checks (`permits_spell` and `cast_spell_action`), announcement, payment, and targeting against `cast.spell(...)`, never front-face shortcuts. A modal back enters the battlefield back face up (`ZoneEntry { transformed: true }`) and restores the front on leaving. Adventures and Omens follow their own stack-exit rules (CR 715.3d / shuffle). `ContinuationFrame` and `DecisionContextView` project the selected profile through announcement; the v11 action and relation rows carry one `SPELL_PROFILE` category (size 4).
- **Play routes & layered permissions (`PlayRoute` + `LayerBoard.play_permissions`).** `PlayRoute` covers `Hand`, `FreeFromHand` (Omniscience), `LibraryTop` (Vizier), `PrintedFlashback`, `GrantedFlashback` (Iroh), `Grant(usize)` (exact-object grants), `GraveyardType` (Muldrotha per-type quota), and `ExileCounter(CounterKind)` (Tinybones). Static play permissions — `free_from_hand`, `library_spells`, `graveyard_flashback`, `opponents_hand_only`, `flash_spells`, `exile_counter_plays`, and controller-specific `extra_land_plays` — live together on `StaticPlayPermissions` in `LayerBoard`. A land's chosen route authorizes foreign ownership and sets `ZoneEntry { controller: Some(player) }`.
- **Single-route wire compatibility vs. route inspection.** `Action::{CastSpell, PlayLand}` keeps `route: None` when only one route exists so retained single-route `Command`s stay wire-compatible, while `Game::resolved_play_route(action)` resolves the actual `PlayRoute` for observation projection, `action_focus`, and `action_description` so solitary `GraveyardType` (Muldrotha) plays always expose `quota_type`, grantor focus, and route description (not only on overlap). `Decision::CastDuringResolution` keeps `Action::SelectCard` for front-face offers and `Action::CastSpell` for non-front profiles to preserve retained replay captures.
- **Exact grants & private face-down exile (`PlayGrant`).** Exact grants keep orthogonal fields: caster/liveness (`player`, `source`, `follows_source_controller`, `condition: Option<Condition>`), timing/duration (`this_turn`, `through_next_turn`, `own_turn_only`, `after_turn`, `flash`), cost/payment (`cost`, `any_mana`, `waterbend`), visibility (`face_down`), Foretell semantics (`foretold`), profile restriction (`on_adventure`), stack-exit (`flashback`), and linked copy survival (`prepared`). `face_down` and `can_view_face_down_exile` govern all private face-down exile visibility (`zones.rs`, `continuation.rs`, `observation.rs`, `policy_history.rs`); `is_foretold` is reserved strictly for Foretell cost lookup and `fill_foretold_slots` determinization.
- **Sparse exiled-object counters & spending origin** (planned, Slices 1–2). Non-battlefield counters will live in sparse `GameState.object_counters: BTreeMap<ObjectId, Counters>` (projected to `CardData.counters`, v11 `FactKind::Counter`, LLM text, and Academy) so normal objects stay unchanged and CR 400.7 zone moves automatically stale old keys. `SpendingContext::Cast { spell: SpellSnapshot, from: ZoneType }` will supply the casting zone to `ManaRestriction::allows` (one spending authority for `ExceptSpellsFromHand` and all other restrictions).
- **Additional costs & dynamic pricing.** Omniscience replaces the mana cost while still permitting Kicker/Bargain (CR 118.9d) and requiring mandatory additional costs. Harbinger's `AdditionalCost::FlashSurcharge` buys flash only when required at priority and is skipped during resolution. `StaticAbility::SpellsCostLess` supports dynamic `EffectValue` reductions (equipped creature power for Glamdring). Compound priority offers bind the chosen target with its spell profile and route while keeping compact priority precondition ordering stable.

## Delivered

The per-set ledgers link each implemented row to its definition and native
fixture. The shared authority first shipped on this branch with Zul Ashur,
Muldrotha, Vizier, Etali, Smaug, Bilbo, Twinmaw and Scavenger Regent. The
first 44 preparation rows include linked copies, source transfers, preparation
during resolution and Emeritus of Truce.

- Yue offers Spew Flame from hand and Solstice Revelations offers Charring Bite from exile (`f3d84a7abe`, `be71091782`). The two fixtures in `tla.rs` verify the legal offer, the selected spell in `ContinuationFrame` and `DecisionContextView`, and resolution/departure.
- Monica Rambeau // Photon uses one front/back definition pair. Photon pays its printed cost, enters back face up and returns to Monica on leaving. Compound casts bind face, permission and payable target; compact priority preconditions retain recorded candidate addresses.
- Foundations covers 11 of 17 rows: Omniscience (`FreeFromHand` + manual Kicker on Burst Lightning, deduplicated across copies), High Fae Trickster (timing), Harbinger of the Tides (surcharge + required sacrifices on free cast), Sphinx of Forgotten Lore (exact-object `flashback` grant alongside Think Twice's printed route), Reassembling Skeleton, and Suspicious Shambler.
- Harmonized Trio and Joined Researchers bring preparation to 46 of 62 rows.
- HOB covers 12 of 17 Adventures plus Bilbo's competing exile replacement. TDM covers 12 of 13 Omens (Feral Deathgorger same-graveyard targeting, Till and Tend stacked extra land plays).

## Execution plan for remaining cohorts

Twenty-one MTG-419 rows remain (9 core casting/permission/payment rows across Slices 1–3, plus 12 non-casting shared-rules rows). Six further preparation rows belong to MTG-418. Post-build concept review (`scratch/concept-review.html`) sequences the core casting-authority fixes first so every remaining permission card plugs into one clean model.

### Slice 1 — Casting-authority alignment & provenance (remaining: Heartwood Crafter, Sanar)

Done in `c5c415307f` and `bf5e38d587`:

- **One builder.** `Game::cast_choices(player, object, profile, CastAuthority)` (`flow/play.rs`) replaces `cast_via` and `Decision::resolution_cast_choices`. `CastAuthority::Resolution` sets `resolving: true` for every `CastDuringResolution` cast (R1 fixed: `fdn.rs::harbinger_cast_by_a_resolving_effect_owes_no_flash_surcharge`). `payment_preview` builds front-face `SelectCard` resolution offers through it (R3 fixed: `hob.rs::bilbo_previews_front_face_and_adventure_graveyard_casts_alike`).
- **`Game::resolved_play_route(action)`** feeds observation route price/quota, `action_focus` and `action_description`; Commands keep `route: None` (R2 fixed: `fdn.rs::lone_graveyard_routes_keep_their_permission_in_the_observation`). Lone routes now describe themselves in text, e.g. "Cast Sozin's Comet using foretell for {2}{R}" and "… from library"; the two text-observation expectations were updated. Pinned pilot: no break (86 passed with the R1–R3 extension).
- **Provenance.** `CastChoices.from_prepared` comes from the preparation grant in the builder; `SpellOnStack.from_prepared` carries it; `copy_stack_object` clears it. `TriggerCondition::Prepared { condition }` wraps a cast trigger. Codie, Ravenous Codex is implemented (`fra.rs::codie_copies_prepared_spells_and_prepares_each_creature`; `Book` added as an artifact subtype, see questions).

Next in this slice:

1. **Heartwood Crafter // Soul Tether.** `SpendingContext::Cast { spell, from: ZoneType }` and `ManaRestriction::ExceptSpellsFromHand`. Boundary found: restricted pool mana must be policy-representable or `unrepresentable_state` refuses the whole observation (`observation_encoder.rs`). Append `turn_state` columns for "can't be spent on hand spells" mana (by colour, like `NONARTIFACT_MANA_*`), with `RestrictedManaData` / Python binding / text in the same change, then check that the pinned pilot projects the saved prefix. Fixture: the restricted {C} pays the linked Soul Tether from exile, Codie copies it, and an ordinary hand spell rejects the same {C}.
2. **Sanar // Wild Idea** (SOS): gate the Treasure activation on an instant/sorcery in `TurnState.spells_cast` (already per-turn cast object ids; read their types through LKI, so this needs no MTG-418 counter); Wild Idea searches, reveals, then shuffles.
3. Update `sos.jsonl` and the Heartwood row of `fra.jsonl`.

Item 4 (per-profile land check) stays latent until the first spell // land modal card.

### Slice 2 — Counter-marked exile, private conditional exile & dynamic pricing (5 cards: Tinybones, Gandalf, Glamdring, Loot, Redcap)

1. **Tinybones, Bauble Burglar (FDN):** Add sparse `GameState.object_counters: BTreeMap<ObjectId, Counters>` (projected to `CardData.counters`, v11 `FactKind::Counter`, LLM text, Academy), `StaticPlayPermissions.exile_counter_plays`, `PlayRoute::ExileCounter(CounterKind)` (visited in `priority_candidate_objects`), and discard-event trigger. Deliver the full required interaction-evidence fixture (see table below).
2. **Gandalf, Goblins' Bane // Flameshape (HOB):** Decouple `PlayGrant.face_down: bool` and `PlayGrant.condition: Option<Condition>` from `PlayGrant.foretold`. Replace the 5 visibility calls to `is_foretold` (`zones.rs`, `continuation.rs`, `observation.rs`, `policy_history.rs`) with `is_face_down_in_exile` (for public-source, history, and `CardData.face_down`) and `can_view_face_down_exile(object, viewer)` (which enforces both viewer information-sharing and the grant's live `condition`, so losing the last Wizard hides the cards even from their owner). Keep `is_foretold` strictly for Foretell cost lookup and `foretold_cards` / `fill_foretold_slots` determinization (`search.rs`, `possible_worlds.rs`, `selection_sampling.rs`). Test Wizard persistence across Gandalf dying with another Wizard, losing all Wizards (hidden & unplayable), and regaining a Wizard (visible & playable again).
3. **Glamdring, Foe-hammer // Gleam of Death (HOB):** Extend `SpellsCostLess` (or dynamic reduction in `effective_spell_cost`) with `EffectValue` reading equipped creature's current power; Gleam of Death mills six and returns instants/sorceries before Adventure exile.
4. **Loot, Exuberant Explorer (FDN):** Add controller-specific `StaticAbility::AdditionalLandPlay` on `StaticPlayPermissions.extra_land_plays` + private look-six selection bounded by resolution-time land count.
5. **Redcap Gutter-Dweller (FDN):** Commit optional another-creature sacrifice before +1/+1 counter and `exile_top_cards_to_play(ThisTurn)`; create printed nonblocking Rat tokens.
6. Update `fdn.jsonl` and `hob.jsonl` ledgers.

### Slice 3 — Graveyard counter-payment, TDM completion & non-casting split boundary

1. **Quilled Greatwurm (FDN):** Graveyard casting permission with distributed `CostComponent::RemoveCounters` payment in `payment.rs` + combat-damage counter trigger.
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
| Tinybones, opposing discarded land and spell with stash counters, second Tinybones | Current controller plays foreign cards exiled by an earlier Tinybones. Check own-turn timing, foreign land control (enters under Tinybones’s controller, not owner), ownership versus control, mana spending, High Fae Trickster non-bypass of own-turn timing, and `HandOnly` (blocks spell, allows land). Use Vizier's private top card and foretell's hidden exile as opposite-viewer controls. |
| Gandalf’s Flameshape vs. Foretell | Private face-down exile visible only to caster while controlling a Wizard; persists when Gandalf dies and another Wizard remains; hides and locks when last Wizard leaves; restores when a new Wizard enters; never enters `fill_foretold_slots` determinization. |

## Validation

**Consumer checks per slice.** Compare native observations for both viewers with the Python bindings, semantic capture/LLM text and Academy offers at priority, at mid-announcement (`AnnounceX`/`ChooseTargets`/`Pay`), with a spell on the stack, and at a nested decision. Use v11 rows and learning tokens. Keep hidden information out of the other viewer's data. Fork-and-replay checks need exact preconditions and stale rejection. Existing controls:

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards airbend_permission
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards flashback_is_exiled_when_countered_after_iroh_leaves
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards sozins_comet_foretell_spending_choices_preserve_color_and_private_identity
cargo test --manifest-path managym/Cargo.toml -p managym --lib preparation_resolution
uv run pytest tests/semantic/test_text_observation.py tests/academy/test_choice_navigation.py tests/model/test_pinned_pilot.py -q
```

The 31 ledger rows in `be46a3fc8c` passed their directed checks in a 148-test native card/observation run, plus 10 structured-offer tests, 5 preparation resolution tests, the execution vocabulary test, and the Scryfall check. Clone Legion copied a prepared creature without duplicating its cast; Honest Work removed abilities while its linked spell remained castable; Seed Suture replayed through ordinary naming Commands. The five Academy navigation positions retained all 252 Commands/offers, and the authored match retained 193 Commands across 194 checkpoints and 1,164 viewer projections with stale rejection. After merging `edee24dad5`, 8 focused Python checks and 11 native `offers_` tests passed.

After `bf5e38d587` the validation Python controls passed (86) once `scripts/refresh_academy_fixtures.py` refreshed the catalog digests (`6a1adcccef`; labels unchanged). The conformance receipts have their own writers and were not regenerated here; gate should check `conformance/` for the new catalog.

Gate owns the full debug workspace, Clippy and the measured 10% Engine Budget.

## Later phases

**Decomposition.** Follow Muldrotha and Zul from the same graveyard object through payment, Omniscience and Harbinger from hand (and Harbinger via free resolution cast), Tinybones from counter-marked exile, Gandalf from conditional face-down exile, and an Adventure, Modal BackFace, and linked prepare copy through stack and departure. One permission owner (`PlayRoute` + `LayerBoard.play_permissions`), one profile owner (`SpellProfile`), and one `CastChoices` builder must explain all of them, including airbend, flashback, foretell, and modal regressions. Remove any exclusive old paths. Review repeated FRA spell programs for card-specific dispatch.

**Performance.** Use the 10% same-runner Engine Budget and the [stress-state protocol](../docs/development.md#stress-states). Add `bench_states` shapes for:
- Muldrotha with a graveyard of duplicate artifact creatures, overlapped by a Zul grant.
- A hand of Dragon/Omen cards with restricted mana and several Charring Bite targets.
- Many Phytomedics with their own Seed Sutures, with one transfer and a zero-link control.

This is where to decide whether the linear grant scan needs an index. Measure first.
