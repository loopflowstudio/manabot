# MTG-431 — Foundations stack copies, choices, costs and procedure tail

All 56 of the 56 scoped Foundations rows have directed implementation evidence.
Slices 1–10 are complete. Demonic Pact and Maze’s End already had working
definitions.

**Next step: Slice 11 (Concept-Alignment Pass)** — close the four
state/observation/UI mismatches found in `scratch/concept-review.html`, then
run **Gate / Integrated validation** (full debug suite, extension rebuild,
Academy fixtures, current-world admission, pinned-pilot checks, Engine Budget,
stress states, and cross-card synergies).

Remaining implementation passes: **1 (Slice 11 below)**. Every one of the 56
scoped ledger rows is `implemented`; Abyssal Harvester's `blocked_by` is
cleared.

## Slice 11 — Concept-Alignment Pass (from `scratch/concept-review.html`)

Behavior review (2026-11-25, head `bc39f13ca3`): none of Slice 11 is built yet —
`Retarget` projects no `"new target"`, `Player.combat_mana_pool` and
`clear_combat_mana_pools` remain (`flow/mana.rs:2969,3476`), `KeepPermanents`
still emits `keep_creatures` (`decision_context.rs:527-528`), and `prompt.ts`
has no cases for the new decision kinds. These four items are the open defects.

Build in this order; each piece is independently testable and preserves
patch-clean 0.2.x compatibility and pinned-pilot tokens:

1. **Project `Decision::Retarget`'s already-chosen targets (`chosen: Vec<Target>`):**
   - *Model:* Whole-assignment retargeting keeps the stack object's original
     targets until the last slot commits (CR 115.7e), so `Decision::Retarget`
     is the sole owner of earlier slot choices. Projecting only `chosen.len()`
     erases those choices from policies, text play, and Academy reconnect.
   - *Changes:* In `managym/src/agent/effect_execution.rs`, add `"new target"`
     to `EXECUTION_TEXT` and emit `out.target("new target", *target)` for every
     target in `chosen` inside `project_decision` for `Decision::Retarget`
     (keep the existing `"retargeted stack object"` and `"retarget"` fields).
     Verify `record_vocabulary.rs` registers the new label cleanly and confirm
     text play / `decision_context` shows the pending new targets.
   - *Verification:* Add a directed test copying a multi-target spell
     (Biogenic Upgrade with `[2, 1]` shares or Drakuseth on a 4-creature board),
     choosing two different targets for slot 0 on two forks, and asserting that
     the encoded observations and text descriptions at slot 1 differ and name
     the slot-0 new target while the stack object still holds its original
     targets.

2. **One mana-expiry authority (`ManaExpiry`) from pool storage through Academy & text play:**
   - *Model:* Mana expiry (`Step`, `EndOfCombat`, `EndOfTurn`) is one concept on
     `ManaClass`, not two pools (`mana_pool` + `combat_mana_pool`) or a boolean
     (`combat: bool`). Today Savage Ventmaw's `EndOfTurn` mana renders as
     *"expires at step end"* in text play, Academy, and payment preview, and
     `ManaSpend` gives `EndOfCombat` and `EndOfTurn` identical blank suffixes.
   - *Changes:*
     - Unify `Player.combat_mana_pool` into `Player.mana_pool` using
       `ManaExpiry::EndOfCombat`. Delete `Player.combat_mana_pool`,
       `combat_mana_pool_mut`, `clear_combat_mana_pools()`, and the
       strip-and-fallback branch in `spend_mana_unit`. Route
       `Effect::AddMana { until_end_of_combat: true, .. }` into `mana_pool`
       with `ManaExpiry::EndOfCombat` (keeping `record_bending(Fire)` and
       keeping the `until_end_of_combat` field on serialized card definitions
       unchanged so TLA pinned-pilot tokens stay intact). Update
       `state/hash.rs`, `mana/preview.rs`, and `card_capture.rs`.
     - Derive `PlayerData.combat_mana` by summing `ManaExpiry::EndOfCombat`
       units from `mana_pool`.
     - Add an additive `expiry` field (defaulting to `Step` when omitted for
       backward compatibility) to `PooledManaClassData`, `PyPooledManaClass`,
       `PooledManaView`, `managym/experience.py`, `protocol/*.schema.json`, and
       `academy/ui/src/lib/types.ts`, while keeping `combat: bool` derived as
       `expiry == EndOfCombat` for patch-clean 0.2.x compatibility.
     - Update `managym/src/flow/action.rs` (`ManaSpend` descriptions),
       `manabot/semantic/text_observation.py`, and
       `academy/ui/src/lib/mana.ts` (`pooledManaTerms`) to distinguish
       `Step`, `EndOfCombat`, and `EndOfTurn` accurately.
   - *Verification:* Directed Rust + Python/Academy tests verifying Savage
     Ventmaw's floating mana renders *"expires at end of turn"* in text play,
     Academy `pooledManaTerms`, payment preview, and `ManaSpend` labels, and
     that firebending (`EndOfCombat`), Ozai retention, and Ventmaw (`EndOfTurn`)
     all behave identically through the single `mana_pool`.

3. **Decision-memory hygiene, offer-kind alignment & Academy prompt lines:**
   - *Changes:*
     - `managym/src/agent/decision_memory.rs`: skip non-leaf
       `NameOption::Group(_)` clicks on `Decision::ChooseEntryValue` (matching
       `Decision::NameCard`), so tree-navigation steps do not pollute public
       decision memory.
     - `managym/src/flow/decision.rs`: emit `ActionSpaceKind::Modal` (matching
       `Action::ChooseMode { 0 | 1 }`) for `Decision::PartitionPiles` and
       `Decision::ChoosePile` instead of `ActionSpaceKind::LookAndSelect`.
       Update any Curator tests checking the action space kind.
     - `managym/src/decision_context.rs`: emit stage `"keep_permanents"` for
       `Decision::KeepPermanents`, keeping `"keep_creatures"` accepted
       alongside `"keep_permanents"` in `managym/academy/recording.py` and
       `manabot/semantic/text_observation.py` for older recordings.
     - `academy/ui/src/lib/board/prompt.ts`: add explicit `promptLine` cases
       for `target_shares`, `entry_choice`, `order_entry_choices`,
       `partition_piles`, and `choose_pile` (with unit tests in
       `academy/ui/src/lib/board/`).

### Choices made for review & explicitly deferred work

- **Unifying `combat_mana_pool` while keeping `Effect::AddMana.until_end_of_combat`:**
  removes the duplicate pool lifecycle inside MTG-431 without churning TLA
  pinned-pilot definition tokens.
- **Keeping `PooledManaClassData.combat: bool` alongside additive `expiry`:**
  preserves patch-clean 0.2.x compatibility.
- **Outside MTG-431 (separate follow-up Tasks):**
  - Semantic candidate-token encoding for `Decision::ChooseEntryValue` (Color
    and 367 CreatureTypes) — generalizing `NameCandidateEncoder` to
    `RecordNamespace::Token` vocabularies across `managym`,
    `manabot/policy/prepare.py`, Torch, and MLX.
  - Retiring legacy `Effect::LookAndSelect` onto `SelectCards { from: LibraryTop }`
    and replacing `Effect::AddMana.until_end_of_combat` with `ManaExpiry` on
    definitions (pinned-pilot rollover).

## Finished slices (detail in git and the ledger)

1. Entry choices on one battlefield incarnation (Banner, Haven, Mare, Automaton, Courtyard, Spyglass) — `47e0e34eda`, `9e1bd7c4a8`, `1635633a40`.
2. Curator's opponent-chosen piles — `49417f76ad`, `cb84b1c0fe`.
3. Biogenic Upgrade's cast-time counter shares — `49417f76ad`, `cb84b1c0fe`.
4. `TargetRelation` and whole-assignment `Decision::Retarget` (Drakuseth, Fiery Annihilation, Run Away Together, Divine Resilience, Charming Prince, Affectionate Indrik) — `4e481dc0a1`.
5. Stack copies and retargeting (Thousand-Year Storm, Bolt Bend, Teach by Example) — `d73b18db4a`; stress `bench_states storm-copies --sizes 8`.
6. `LibraryTop`/`Unselected` selectors, `SelectCards.max_mana_value`, `Effect::GrantPlay` (Gutless Plunderer, Strongbox Raider, Genesis Wave) — `460fe18685`.
7. Last-known values and shared sacrifice (Tribute to Hunger, Deadly Brew) — `30fd60f499`.
8. Kicked entries, composite ward, `ManaExpiry`, Changeling, `CardPredicate.keyword`, `CounterKind::Wish` (Gnarlid Colony, Gatekeeper, Ovika, Ventmaw, Mascot, Teachings, Wishclaw, Ballyrush) — `460fe18685`.
9. 23 compose-only cards (`token.raccoon_3_3_g`, `token.cat_beast_2_2_w`, `Effect::double_counters`, graveyard trigger `intervening_if` and `source_present` fixes for Flamewake Phoenix and Gate Colossus) — `1eb342e57b`.
10. Abyssal Harvester (`TurnState::graveyard_arrivals`, `CardPredicate.put_into_graveyard_this_turn`, `Except::CreatedThisWay`, `CardData.put_into_graveyard_this_turn`, v11 `card::PUT_INTO_GRAVEYARD_THIS_TURN` column 56) — `1eb342e57b`; `blocked_by` cleared on its ledger row.

## Integrated validation

- Rust: `CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2 --manifest-path managym/Cargo.toml`
  in debug before landing; `cargo fmt --all`.
- Rebuild the extension and refresh Academy fixtures
  (`scripts/refresh_academy_fixtures.py`).
- Current-world admission:
  `uv run --extra dev --extra artifacts --extra reports pytest tests/semantic/test_elemental_world.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py tests/model/test_choice_separability.py -q`,
  plus `tests/model/test_pinned_pilot.py`.
- Stress states through `bench_states`; Engine Budget stays within 10%:
  8–16 Courtyards with distinct mana restrictions and eight differently
  chosen Banners or Automatons (payment and layers); `storm-copies --sizes 8`;
  Genesis Wave X = 16; eight Ventmaw-style pools; Changelings under eight
  chosen-type anthems.
- Synergies to exercise together:
  - Thousand-Year Storm with Teach by Example (a Teach copy is not cast, so
    it never raises Storm's count).
  - Bolt Bend redirecting a Drakuseth trigger's single-target copy is not
    legal; it is legal on Fiery Annihilation only if that spell has one
    target.
  - Three Tree Mascot with Adaptive Automaton, Secluded Courtyard and
    Ballyrush Banneret.
  - Sorcerous Spyglass naming Three Tree Mascot leaves its mana ability
    working; naming Wishclaw Talisman stops it.
  - Genesis Wave putting Heraldic Banner and Uncharted Haven onto the
    battlefield simultaneously: each asks its entry choice.
  - Deadly Brew against Desecration Demon.
  - Gate Colossus with Maze's End.
- Pinned pilot: v11 card column 56 (`put_into_graveyard_this_turn`) is an
  appended input column; `tests/model/test_pinned_pilot.py` must pass or the
  change must be declared with `scripts/pinned_pilot.py record --reason`.
