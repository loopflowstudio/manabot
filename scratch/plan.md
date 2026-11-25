# MTG-431 — Foundations stack copies, choices, costs and procedure tail

All 56 of the 56 scoped Foundations rows have directed implementation evidence.
Slices 1–11 are complete. Demonic Pact and Maze’s End already had working
definitions.

Every one of the 56 scoped ledger rows is `implemented`; Abyssal Harvester's
`blocked_by` is cleared. Slice 11's four concept defects are built (no
`combat_mana_pool`, `keep_permanents` stage, Retarget `"new target"`, Academy
prompt lines; `prompt.test.ts`/`mana.test.ts` pass).

## Slice 12 — Defects from behavior review (head `1278923dc6`)

Full debug suite (`cargo nextest run --profile ci --workspace`): 2157 passed,
16 failed. `tests/semantic/test_text_observation.py`: 100 passed, 2 failed.
Fix all three, then go to gate.

1. **Azula fixture clobbers firebending mana.**
   `tla::fire_lord_azula_copies_spells_cast_while_she_attacks`
   (`managym/cards/tests/cards/tla.rs:15249`) panics "expected a pending mana
   payment". With one unified pool, `set_player_mana_pool(0, "R")` replaces the
   pool and deletes the end-of-combat {R}{R} that Azula's firebending added, so
   only one spend remains and payment never asks. The engine is right; the
   fixture is wrong. Fix: add the regular {R} to the existing pool (empty the
   step-expiring funds first, keep the `EndOfCombat` class) so both expiries are
   offered, and keep the spend-order assertion. Search the other combat-mana
   tests for the same replace-after-firebend pattern.
2. **LLM text misses `put_into_graveyard_this_turn`.**
   `test_every_observed_field_has_llm_text[Card]` fails: `managym.Card`
   exposes `put_into_graveyard_this_turn` (v11 column 56, Abyssal Harvester),
   but `manabot/semantic/text_observation.py:_cards_lines` never renders it.
   New state must reach policy input, LLM text and tokens together. Fix: render
   it on the card line, beside the other per-card turn facts, and add it to
   `_TEXT_FIELDS[managym.Card]` in `tests/semantic/test_text_observation.py`.
   `[PooledManaClass]` fails only because the table lacks `expiry`, which
   `text_observation.py:903-906` already renders. Add the
   `"expiry": "expires at end of"` entry.
3. **Frozen semantic-kernel evidence is stale.** All 15
   `managym::semantic_conformance_tests frozen_*` cases fail on `trace_sha256`.
   The inventory test also fails on `content_digest`
   (`evidence-battlefield-history-v1.json` content digest `76d0fa…` vs live
   `21ab02…`). The expected causes are the new FDN definitions and
   `MATCH_STATE_HASH_VERSION` 33. Before re-recording, show that the Commands
   and event boundaries match the base `3be5a0cfb2` and that only the
   state-hash and content-digest fields differ (replay a drifted case from
   `target/semantic-conformance/semantic-kernel-w17-v6/` with
   `semantic_conformance -- replay`, then compare it with the base's
   regenerated case with hashes stripped). Then re-record with
   `semantic_conformance -- record --root <staged corpus>`
   (`conformance/README.md`). Regenerate the authored-match authority and
   parity receipts that bind the same catalog
   (`scripts/generate_authored_match_receipt.py`,
   `uv run --extra dev python -m conformance.authored_match_parity write`). If any
   Command or event differs, that is a behavior change: find which slice caused it.

Reviewed and holding: synergy tests exist for Storm/Teach copies not counting
as casts, Bolt Bend at its actual cost, Drakuseth's distinct secondary targets,
Mascot changeling mana, Spyglass and Banner simultaneous entry, Genesis Wave
simultaneous entry, Deadly Brew's choose-before-sacrifice, Gate Colossus, and
Maze's End. Nit: Retarget appears in Academy as "Choose a target for …"
(`CHOOSE_TARGET`), and `KeepPermanents` falls under `LOOK_AND_SELECT`'s "Choose
from the revealed cards". Neither prompt is wrong enough to block.

**After Slice 12: Gate / Integrated validation** (below).

## Follow-ups outside MTG-431 (separate Tasks)

- Semantic candidate-token encoding for `Decision::ChooseEntryValue` (Color and
  367 creature types): generalize `NameCandidateEncoder` to
  `RecordNamespace::Token` vocabularies across `managym`,
  `manabot/policy/prepare.py`, Torch and MLX.
- Retire `Effect::LookAndSelect` onto `SelectCards { from: LibraryTop }`, and
  replace `Effect::AddMana.until_end_of_combat` with `ManaExpiry` on
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

11. Concept alignment: Retarget projects chosen `"new target"`s; one `Player.mana_pool` with `ManaExpiry` (combat pool deleted, additive `expiry` beside derived `combat`, `MATCH_STATE_HASH_VERSION` 33); group clicks skipped in decision memory; `PartitionPiles`/`ChoosePile` are `Modal`; `keep_permanents` stage (`keep_creatures` still read); Academy prompt lines — `e1f42473b0`. The commit records no test output; its directed tests run first at gate.

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
- Hash version 33 moves match-state digests: expect Engine Budget digest and
  retained-fixture changes; adopt with `scripts/engine_budget.py record --reason`.
- Pinned pilot: v11 card column 56 (`put_into_graveyard_this_turn`) is an
  appended input column; `tests/model/test_pinned_pilot.py` must pass or the
  change must be declared with `scripts/pinned_pilot.py record --reason`.
