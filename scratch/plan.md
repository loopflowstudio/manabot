# MTG-431 — Foundations stack copies, choices, costs and procedure tail

All 56 of the 56 scoped Foundations rows have directed implementation evidence.
Slices 1–10 are complete. Demonic Pact and Maze’s End already had working
definitions.

**Next step: Gate / Integrated validation** — run the full debug suite, rebuild
the extension and Academy fixtures, run current-world admission and pinned-pilot
checks, Engine Budget, stress states, and the cross-card synergies below.

Remaining implementation passes: none. Every one of the 56 scoped ledger rows
is `implemented`; Abyssal Harvester's `blocked_by` is cleared.

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
