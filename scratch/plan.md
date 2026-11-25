# MTG-431 — Foundations stack copies, choices, costs and procedure tail

All 56 of the 56 scoped Foundations rows have directed implementation evidence.
Slices 1–11 are complete. Demonic Pact and Maze’s End already had working
definitions.

**Next step: Gate / Integrated validation** — full debug suite, extension
rebuild, Academy fixtures, current-world admission, pinned-pilot checks,
Engine Budget, stress states, and cross-card synergies.

Remaining implementation passes: **0 (all 11 slices complete)**. Every one of
the 56 scoped ledger rows is `implemented`; Abyssal Harvester's `blocked_by` is
cleared.

## Slice 11 — Concept-Alignment Pass (from `scratch/concept-review.html`) — DONE

Implemented and verified with directed Rust, Python, and Academy UI tests:

1. **Project `Decision::Retarget`'s already-chosen targets (`chosen: Vec<Target>`):**
   - Added `"new target"` to `EXECUTION_TEXT` in `managym/src/agent/effect_execution.rs`,
     emitted `out.target("new target", *target)` for every chosen target in
     `Decision::Retarget` plus a `"New targets chosen so far: ..."` summary,
     and included `"retarget"` in `manabot/semantic/text_observation.py`'s
     pending-summary stages.
   - Verified in `rna::retarget_projects_earlier_chosen_targets_before_final_commit`
     and `tests/semantic/test_text_observation.py::test_retarget_and_keep_permanents_surface_in_text_and_academy`.
2. **One mana-expiry authority (`ManaExpiry`) from pool storage through Academy & text play:**
   - Unified `Player.combat_mana_pool` into `Player.mana_pool` with
     `ManaExpiry::EndOfCombat`, deleted `Player.combat_mana_pool`,
     `mana_pools_mut`, `combat_mana_pool_mut`, `clear_combat_mana_pools()`, and
     the strip-and-fallback branch in `spend_mana_unit`, bumped
     `MATCH_STATE_HASH_VERSION` to 33, and derived `PlayerData.combat_mana`
     from `Mana::expiry_total(ManaExpiry::EndOfCombat)`.
   - Added additive `expiry` (`"step"` / `"end_of_combat"` / `"end_of_turn"`,
     omitted/defaulting to `"step"` for backward compatibility) to
     `PooledManaClassData`, `PyPooledManaClass`, `managym/__init__.pyi`,
     `PooledManaView` (Rust + Python + all 6 `protocol/*.schema.json`),
     `experience_projection.py`, and `academy/ui/src/lib/types.ts`, keeping
     `combat: bool` derived as `expiry == EndOfCombat`. Recorded additive
     `tests/public_api.json` snapshot.
   - Updated `manabot/semantic/text_observation.py` and
     `academy/ui/src/lib/mana.ts` (`pooledManaTerms`) to distinguish `step`,
     `end_of_combat`, and `end_of_turn`.
   - Verified in `dtk::savage_ventmaw_mana_survives_until_cleanup`,
     `tle::animal_attendant_mana_in_the_combat_pool_keeps_its_spend_effect`,
     `tla::ozai_keeps_unspent_mana_as_red_and_flies_on_six`,
     `tla::firebending_mana_lasts_until_end_of_combat_and_casts_spells`,
     `tests/semantic/test_text_observation.py::test_savage_ventmaw_end_of_turn_mana_renders_in_text_and_academy`,
     and `academy/ui/src/lib/mana.test.ts`.
3. **Decision-memory hygiene, offer-kind alignment & Academy prompt lines:**
   - `managym/src/agent/decision_memory.rs`: skipped non-leaf `NameOption::Group(_)`
     clicks on `Decision::ChooseEntryValue` and recorded the canonical leaf row
     (`choose_entry_value_skips_group_navigation_and_records_canonical_leaf_row`).
   - `managym/src/flow/decision.rs`: emitted `ActionSpaceKind::Modal` for
     `Decision::PartitionPiles` and `Decision::ChoosePile` (updated
     `fdn::curator_of_destinies_keeps_the_split_private_and_resolves_either_pile`).
   - `managym/src/decision_context.rs`: emitted `"keep_permanents"` for
     `Decision::KeepPermanents`, keeping `"keep_creatures"` accepted alongside
     it in `managym/academy/recording.py` and `manabot/semantic/text_observation.py`.
   - `academy/ui/src/lib/board/prompt.ts`: added `promptLine` cases for
     `target_shares`, `entry_choice`, `order_entry_choices`, `partition_piles`,
     and `choose_pile` (`academy/ui/src/lib/board/prompt.test.ts`, `svelte-check`).

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
