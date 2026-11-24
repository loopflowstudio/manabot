# MTG-419 · Shared casting profiles and play permissions

Task steering, 2026-11-24: “Continue through straightforward cards without returning to review between cards. Realign and review the cohort together. Split only at a concrete shared-rules dependency or genuinely distinct implementation boundary; record that reason and the remaining rows.”

The Task brief requires “one shared casting-profile and play-permission model” and says “Preserve complete legal choices, observations, learning tokens, LLM text, replay/forks, and the 10% Engine Budget.” It also says to “Leave scoped event-history and contextual-query extensions to the history/queries task (defer cards strictly requiring new turn-history counters).” That task is MTG-418, which is open and not merged.

## Target model (aligned across layers)

Every legal play is one coherent offer over four orthogonal authorities — **object**, **printed profile**, **play route / permission**, and **optional additional costs**:

- **Profiles (`SpellProfile`: `Front`, `Adventure`, `Omen`, `BackFace`).**
  - A physical `CardDefinition` (`managym/state/src/card.rs`) carries its front face, optional `InsetSpell` (`Adventure`, `Omen`, `Prepared`), and optional `back_face` with `modal_back: bool` (reusing `ContentPack::back_face` so modal and transforming DFCs share one back-face representation).
  - One resolver (`Game::spell_definition` / `Object::spell_definition`) serves priority enumeration, `ResolutionCastOffer` (including `Hand` and `Exiled`, not only `Graveyard`/`AmongExiled`/`Cascade`), targeting, payment, restricted mana, alternative costs, cascade, copies, and resolution.
  - Casting `SpellProfile::BackFace` shows the back face on the stack via `turn_back_face_up`; resolving a back-face permanent spell sets `ZoneEntry { transformed: true }` (CR 712.11a) so it enters back-face up and returns to its front face on any zone departure via `mint_appearance`.
  - `CastChoices.profile` is projected through the **entire** announcement pipeline: `ContinuationFrame` carries `profile`, `DecisionContextView` reads `cast.spell(...)` for `source.label`, `cast_option()`, and `replicate()`, `structured_offer.rs` carries `profile` on `CastSingleTarget` / `AtomicCommand::CastSpell`, and `action` tensors encode one `action::SPELL_PROFILE` category (`ValueKind::Category { size: 4 }`, replacing the branch-added `ADVENTURE` and `OMEN` flags; zero pinned-pilot break because cols 25–29 are new on this branch).
- **Play routes and permissions (`managym/src/flow/cast_permission.rs`, `play.rs`).**
  - `PlayRoute` owns every zone/permission/static-alternative-cost path: `Hand`, `FreeFromHand` (Omniscience, deduplicated across multiple Omnisciences), `LibraryTop` (Vizier), `PrintedFlashback`, `GrantedFlashback` (Iroh), `Grant(usize)` (exact-object grants), `GraveyardType { source, permanent_id, kind }` (Muldrotha), and `ExileCounter(CounterKind)` (Tinybones stash counters).
  - Keeping `FreeFromHand` on `PlayRoute` leaves `CastChoices.kicked` dedicated to optional additional costs so a free Omniscience cast can still be kicked or bargained (CR 118.9d) while blocking a second alternative cost (CR 118.9a) and fixing printed X to 0 (CR 107.3b).
  - `PlayGrant` keeps its terms orthogonal: `follows_source_controller: bool` (caster follows current controller of `source`) is distinct from `prepared: bool` (linked exile copy survives CR 704.5e and marks cast spells as prepared); `flashback: bool` gives exact-object grants (Sphinx of Forgotten Lore) CR 702.34a stack-exit to exile.
  - `AdditionalCost::FlashSurcharge(ManaCost)` + `CastChoices.flash_surcharge: bool` handles Harbinger of the Tides so instant-speed timing requires the surcharge (never offering an illegal decline) while composing cleanly with `FreeFromHand`.
  - `play_land` authorizes ownership via `selected_play_route` (removing the hardcoded `card_ref.owner != player` check) and passes `ZoneEntry { controller: Some(player), .. }` so foreign-owned lands (Tinybones) enter under the player’s control (CR 110.2, 305.4).
  - High Fae Trickster plugs into `has_instant_timing` after route eligibility, so it never bypasses `own_turn_only`, `after_turn`, or `HandOnly`.
- **Preparation and stack provenance.**
  - Entry (`flow/zones.rs`) creates the linked exile `SpellCopy` and a `PlayGrant { source: Some(source), follows_source_controller: true, prepared: true, .. }`. Cleanup (`flow/sba.rs`) consults `preparation_source`. Repeat-prepare and unprepare resolve against the live link without a reverse index.
  - Casting from a `prepared` grant sets `from_prepared: true` on `CastChoices` and `SpellOnStack` so triggers like Codie, Ravenous Codex distinguish cast prepared spells from Pyromancer’s Goggles, Replicate, or Codie copies (copies set `from_prepared = false`).
- **Exiled-card counters and cast-origin mana restrictions.**
  - Sparse `GameState.object_counters: BTreeMap<ObjectId, Counters>` stores counters on non-battlefield objects (CR 122.2; automatic CR 400.7 staleness on zone move, zero bloat on normal `Object`s). Projected into `CardData.counters`, `LegacyCardView.counters`, v11 `FactKind::Counter`, LLM text, and Academy inspect.
  - `SpendingContext::Cast { spell: SpellSnapshot, from: ZoneType }` + `ManaRestriction::ExceptHandSpells` lets Heartwood Crafter enforce its cast-origin restriction at the single mana-spending authority.
- **Resolution-time casts.**
  - Etali, Bilbo, Cascade, and Yue resume their parent via `GameState::cast_caller` when the child spell commits (`flow/decision.rs`), with every `ResolutionCastOffer` evaluating `spell_profiles()`.

## Delivered (detail in git and the set ledgers)

- FDN 4/17 rows: Zul Ashur `97593abf`, Muldrotha `8d380c05`, Vizier, Etali `192bde78` (fixtures in `managym/cards/tests/cards/fdn.rs`, `rix.rs`, `cmr.rs`).
- HOB: Smaug // Spew Flame and Bilbo `f7aa87fe`, with overlap pricing fixed in `897faeaf` (`hob.rs`).
- TDM 2/13: Twinmaw // Charring Bite `c30088d4`, Scavenger Regent // Exude Toxin `04fbb56a` (`tdm.rs`).
- SOS/FRA 44/62 preparation rows: prepared entry for 31 definitions `1852a68d`, the two Seed Suture sources and transfer `b5ff828c`/`25bf25d7`/`fae97841`, 12 resolution-time sources `a3555c39`, Emeritus of Truce `3beb79c5`.
- Inset spells cast through their own definition (alternative costs, cost predicates, stack matching) `8be13fed`.

After `8be13fed`: 47 focused native tests and 95 Python checks passed (cold replays, semantic text, both bundled pilots, Academy navigation). All 252 navigation prefix Commands replayed with identical offers and revisions.

## Lessons that steer the remaining work

- **Every catalog insertion shifts definition rows.** Recompare the five Academy navigation positions with registry references resolved by name before regenerating digests. Then refresh with the fixture writer (`scripts/refresh_card_fixture.py`) and replay the prefixes.
- **Opening decisions must finish before constructed boards.** Fixtures that build a board need both opening-hand decisions completed first, or state-based actions never run (the Seed Suture repair).
- **Mana in fixtures.** With interchangeable lands the engine offers a payment decision after targeting. Submit the offered automatic payment before passing priority.
- **New hashed field names need tokens.** The runtime vocabulary test caught `exile_instead_of_graveyard`. Run the learning-token guards whenever a rules type gains a field.

## Remaining slices

1. **Shared model alignment & decisive probes (proves the target model holds before batch authoring).**
   Land the five concept-review alignments together with their directed probes and cross-consumer checks:
   - **Unified `SpellProfile` (`BackFace` + announcement observability):** Add `SpellProfile::BackFace` and `CardDefinition.modal_back`; resolve permanent back faces with `ZoneEntry { transformed: true }`; replace `action::{ADVENTURE, OMEN}` + `ActionOption.{adventure, omen}` with one `action::SPELL_PROFILE` category (`size: 4`); project `profile` on `ContinuationFrame` and fix `decision_context.rs` (`source.label`, `cast_option()`, `replicate()`); carry `profile` in `structured_offer.rs`; allow non-`Front` profiles in `ResolutionCastOffer::{Hand, Exiled}`. **Probe:** Monica Rambeau // Photon, Living Light (offer both faces, enter back-face up, return to front face off battlefield) + announcement-stage `DecisionContextView` / `ContinuationFrame` check on Exude Toxin / Spew Flame.
   - **Orthogonal `PlayRoute` / `PlayGrant` / `kicked`:** Separate `PlayGrant.follows_source_controller` from `prepared`; add `PlayGrant.flashback`; add `PlayRoute::FreeFromHand` and `AdditionalCost::FlashSurcharge`; add `SpellOnStack.from_prepared`; extend `SpendingContext::Cast` with `from: ZoneType` and `ManaRestriction::ExceptHandSpells`. **Probes:** Omniscience (paid vs free hand cast, Kicker/Bargain still offered on free cast, X forced to 0 on free cast, mandatory additional costs still enforced) + Harbinger of the Tides (instant-speed surcharge mandatory, composes with Omniscience) + Sphinx of Forgotten Lore (`PlayGrant.flashback` overlap with printed flashback).
   - **Exiled-card counters & foreign land play:** Add sparse `GameState.object_counters` + `CounterKind::Stash`, `PlayRoute::ExileCounter(CounterKind)`, `StaticAbility::PlayExiledWithCounter`, `CardData.counters` / v11 `FactKind::Counter` / LLM text / Academy inspect, discard trigger event (`CardDiscarded`), and fix `play_land` to allow route-authorized foreign lands entering under `controller: Some(player)`. **Probe:** Tinybones, Bauble Burglar (Tinybones #1 exiles opposing land & spell with stash counters, #1 dies, #2 enters, current controller plays foreign land under their own control and casts foreign spell with any mana, `HandOnly` blocks spell but not land).

2. **Preparation rows without new history (SOS/FRA, 12 rows).** Author straightforwardly on the aligned model using the existing prepared-source, pending-resolution and source-loss forks:

   | Card | Behavior to distinguish |
   |---|---|
   | Joined Researchers | Opponent hand size checked at end-step trigger and again at resolution |
   | Spiritcall Enthusiast | One or more tokens entering in a batch prepare once |
   | Grave Researcher | Surveil finishes before counting graveyard creatures; Reanimate reads the moved creature's mana value |
   | Kirol, History Buff | One or more cards leaving your graveyard in a batch prepare once |
   | Emeritus of Ideation | Decline, or exile eight distinct graveyard cards, as the attack cost |
   | Harmonized Trio | Tap two other untapped creatures plus the source as one cost |
   | Lluwen, Exchange Student | Sorcery-speed graveyard-creature exile as a cost; the Pest token's attack trigger |
   | Variable Chaser | Each player's optional whole-hand discard, collected in order, then draw seven |
   | Pyre Rhymer | Turn-scoped extra red when tapping Mountains, including ones gained later |
   | Heartwood Crafter | Uses `SpendingContext::Cast.from` + `ManaRestriction::ExceptHandSpells`: its mana cannot pay for hand spells |
   | Sanar, Unfinished Genius | Treasure activation requires an instant or sorcery cast this turn (`spells_cast` already records the objects) |
   | Codie, Ravenous Codex | Uses `SpellOnStack.from_prepared`: copies an actually cast prepared spell (copy has `from_prepared = false`); mass-prepare acts once per eligible source |

3. **Printed programs (HOB/TDM, 27 rows).** Run the other 16 HOB Adventures and 11 TDM Omens through the hand-profile, target/payment and stack-departure fixtures. Check each front and inset program. Group any card that needs a new effect by the effect's owner. (Monica Rambeau // Photon is already proved in Slice 1; battlefield transform overhaul stays with Task 05.)

4. **Remaining zone permissions and alternate costs (FDN, 9 remaining rows after Slice 1 probes).** High Fae Trickster (controller-wide flash in `has_instant_timing`, respecting route restrictions), Loot, Think Twice, Reassembling Skeleton, Flamewake Phoenix, Suspicious Shambler, Quilled Greatwurm, Redcap Gutter-Dweller, Gate Colossus. Keep each card's other printed abilities.

5. **Blocked on MTG-418 event history (6 rows).** Emeritus of Woe and Bloodline Recollector need creature deaths this turn. Leech Collector needs the first life gain of the turn, and Scheming Silvertongue needs total life gained. Whiplash Wordsmith needs noncombat damage dealt to an opponent. Hexhaven Dueling Arena needs to know which creature attacked. `PlayerTurnHistory` (`managym/src/flow/turn.rs`) records only attacked, sacrifices, land entry, draws, bending and life lost. Once MTG-418 merges, run `lf sync main` and read its history API. Then build these rows on the Slice 2 fixtures.

Advance between slices when the ledger gives each completed row a directed witness. Track each row in its set ledger with definition, test names and result.

## Interaction evidence still owed

| Position | Required fork or outcome |
|---|---|
| Muldrotha, artifact creature and a Zul-eligible Zombie | Fork the type choice. Check a new Muldrotha that turn, the next-turn reset, land allowance and a prohibition. With Omniscience, compare paid and free hand casts, required additional costs, Kicker/Bargain on a free cast, and manual spending. |
| Tinybones, opposing discarded land and spell with stash counters, second Tinybones | The current controller plays foreign cards that an earlier Tinybones exiled. Check own-turn timing, foreign land control (enters under Tinybones’s controller, not owner), ownership versus control, mana spending, High Fae Trickster non-bypass of own-turn timing, and a hand-only prohibition (blocks spell, allows land). Use Vizier's private top card and foretell's hidden exile as opposite-viewer controls. |
| Angel and Phytomedic with Seed Suture | Not yet run: remove abilities (stays prepared), copy the permanent (no copied preparation), naming Seed Suture, and Codie's copied prepared spell retargeting (non-recursive). |
| Monica Rambeau // Photon & Exude Toxin announcement | Offer both faces in hand; Photon enters back-face up and returns to Monica off battlefield; mid-announcement `ContinuationFrame` and `DecisionContextView` show the selected profile rather than the front face. |

## Behavior review findings (2026-11-23, at `1018a342`)

Delivered rows hold their directed outcomes: 28 card fixtures (Zul, Muldrotha, Vizier, Etali, Smaug/Bilbo, Charring Bite, Exude Toxin, Volcanic Torrent, airbend, Iroh flashback, foretell spending), 5 `preparation_resolution` lib tests and 90 Python checks (text observation, casting profiles, Academy navigation, pinned pilot) pass in debug. Spending context is built from the selected profile (`effective_spell_cost`, `play.rs:337`), and Vizier's any-type rule reaches Twinmaw's front but not Charring Bite. Muldrotha and printed or granted flashback admit only `Front`; an Adventure exile grant admits only the permanent spell (CR 715.4). The complete Command surface (`compound_offers`) keeps non-`Front` casts as ordinary offers.

Two defects reproduced with temporary probes (now removed). Both are missing legal choices from resolution-time hand and exile casts, and slice 1's `ResolutionCastOffer::{Hand, Exiled}` item must fix them:

1. **Yue cannot cast an Adventure or Omen.** Board: Yue with Smaug, the Great Calamity in hand, an opposing Gray Ogre and {5} paid. Yue's waterbend ability resolves with no decision. Spew Flame is a noncreature sorcery and is legal (CR 715.3a). Cause: the `Hand` filter in `resolution_cast_candidates` (`decision.rs:3344`) checks the predicate against the front object, and the profile loop then breaks on non-`Front` profiles. Fix: check the predicate per profile against `cast.spell(definition)`, the way Cascade checks mana value.
2. **Solstice Revelations cannot reach Charring Bite.** Board: Twinmaw Stormbrood on top of the library and three Mountains. Twinmaw goes to hand, although Charring Bite (MV 2 < 3) is castable for free. Cause: `ExileUntilNonlandCastFree` compares the front's `object_mana_value` before building candidates (`resolution.rs:1349`), and the `Exiled` offer drops non-`Front` profiles. Fix: move the "spell's mana value" bound into the per-profile candidate filter, as `Cascade { below }` does, and send the card to hand only when no profile qualifies.

Add both as directed fixtures (`tla.rs`) beside the existing Yue and Solstice controls. The mid-announcement front-face labels in `decision_context.rs` and on `ContinuationFrame`, which the concept review recorded, are still unprobed. Slice 1's Exude Toxin/Spew Flame announcement check covers them.

**Phase status.** The delivered batch's rows are satisfied. Slice 1 (alignment and probes) has not started: none of `BackFace`, `FreeFromHand`, `object_counters`, `SPELL_PROFILE`, `from_prepared` or `FlashSurcharge` exists in code. Slices 2–5, decomposition and the Engine Budget remain. Gate owns the full debug workspace, fmt and Clippy.

**Consumer checks per slice.** Compare native observations for both viewers with the Python bindings, semantic capture/LLM text and Academy offers at priority, at mid-announcement (`AnnounceX`/`ChooseTargets`/`Pay`), with a spell on the stack, and at a nested decision. Use v11 rows and learning tokens. Keep hidden information out of the other viewer's data. Fork-and-replay checks need exact preconditions and stale rejection. Existing controls:

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards airbend_permission
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards flashback_is_exiled_when_countered_after_iroh_leaves
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards sozins_comet_foretell_spending_choices_preserve_color_and_private_identity
cargo test --manifest-path managym/Cargo.toml -p managym --lib preparation_resolution
uv run pytest tests/semantic/test_text_observation.py tests/academy/test_choice_navigation.py tests/model/test_pinned_pilot.py -q
```

Gate owns the full debug workspace, fmt, Clippy and consumer suites.

## Later phases

**Decomposition.** Follow Muldrotha and Zul from the same graveyard object through payment, Omniscience and Harbinger from hand, Tinybones from counter-marked exile, and an Adventure, Modal BackFace, and linked prepare copy through stack and departure. One permission owner (`PlayRoute`) and one profile owner (`SpellProfile`) must explain all of them, including airbend, flashback and modal regressions. Remove any exclusive old paths. Review repeated FRA spell programs for card-specific dispatch.

**Performance.** Use the 10% same-runner Engine Budget and the [stress-state protocol](../docs/development.md#stress-states). Add `bench_states` shapes for:
- Muldrotha with a graveyard of duplicate artifact creatures, overlapped by a Zul grant.
- A hand of Dragon/Omen cards with restricted mana and several Charring Bite targets.
- Many Phytomedics with their own Seed Sutures, with one transfer and a zero-link control.

This is where to decide whether the linear grant scan needs an index. Measure first.
