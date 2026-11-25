# MTG-431 — Foundations stack copies, choices, costs and procedure tail

All 56 scoped Foundations rows are `implemented` in the FDN ledger with directed
evidence (Demonic Pact and Maze's End already had working definitions;
Abyssal Harvester's `blocked_by` is cleared). Slices 1–17 and `lf compress`
(`558cc971b8`) are built.

**Next: gate — fix `CI/Academy Integration (ubuntu-latest, play)` failure on PR
#605 (`558cc971b8`), re-verify green CI, then demo.** No unbuilt slices and no
further engine-performance iteration needed (`CI/Engine Budget` and `CI/Rust
Tests` already passed on `558cc971b8`; all `baseline.json` digests matched
without needing `engine_budget.py record`).

## Measured performance (`558cc971b8`, post-Slice-17 + compress)

**Same-machine uninstrumented release (Apple M4 Max, contended host; vs merge-base `63bdfc0356`):**

- `bench_steps`: `45.543B, 45.494B` (mean **45.518B** instructions, `13.975B` cycles)
  vs base `47.182B, 47.109B` (mean **47.145B**, `14.917B` cycles) → **−3.45%
  instructions** (−6.32% cycles); `512` games, `0` capped, `118,559` steps,
  digest `10601622128491155944` identical.
- `bench_search`: **967.342B** instructions (`267.498B` cycles) vs base
  **1,033.723B** (`272.810B` cycles) → **−6.42% instructions** (and −5.3B vs
  pre-Slice-17 `972.648B`); seeds 1–3 digests (`13638875039089371439`,
  `5466143068363283977`, `9998925852575961750`), `score_digest_blake3`,
  `simulations` (`7,680 / 8,064 / 7,424`), `cap_hits` (`0`), and every
  `action_counts` array match base and `managym/perf/baseline.json` identically.
- **CI cross-check (`Engine Budget` on `558cc971b8`, Ubuntu 24.04 x86_64 AMD EPYC 9V45 vs reference `7d3ac1eb1`):**
  `steps` **32,207.8/s** vs `30,980.5/s` (**+3.96%**), `search total`
  **81.854 s** vs `87.583 s` (**−6.54%**), `search worst` **1.992 s** vs
  `2.141 s` (**−6.96%**), all digests identical (`PASS`).
- **CPU profiles (`/usr/bin/sample`, 1 ms):**
  - `bench_steps` (4,100 samples): `Observation::new` `39.66%` (`populate_permanents`
    `14.22%`, `player_data` `8.78%`, `populate_cards` `7.27%`), `layer_board`
    `19.10%` (`layers::Context::evaluate` `15.44%`), `indexed_sources` `16.85%` /
    `drain_pending_events` `16.49%`, `priority_action_space` `16.10%`,
    `payment_model` `5.39%`, `finish_mana_payment` `4.78%`. All MTG-431 / Slice 17
    mana paths are `< 0.8%`.
  - `bench_search` (23,163 samples): `layer_board` `35.17%` (`layers::Context::evaluate`
    `28.73%`, `Context::apply` `9.55%`), `indexed_sources` `32.37%` /
    `drain_pending_events` `31.87%`, `priority_action_space` `24.59%`,
    `advance_event_batch` `20.93%`, `payment_model` `8.83%`,
    `finish_mana_payment` `8.30%`, `advance_mana_spend` `5.98%`. Zero MTG-431
    regressions or bottlenecks.
- **`bench_states` (release `--features benchmark`, touched MTG-431 shapes):**
  - `mixed-pools`: `--sizes 2,3,4` **1.265B** inst (vs pre-Slice-17 `1.328B`,
    **−4.74%**); `--sizes 4,8,16` **4.172B** inst (vs pre-Slice-17 `4.497B`,
    **−7.23%**); linear single-step `pooled/actions` (`62 → 94 → 158` calls,
    `42 → 66 → 114` states) and `priority/search` (`5,392 → 11,520 → 24,026`
    calls); `pooled/search` `plans 638 → 668 → 825` (down from `680 → 710 → 879`
    via unified `SourceClassCache`), `calls 9,043 → 15,328 → 33,291`,
    `states 6,029 → 10,651 → 23,801`. Multi-run size-16 profile (5,063 samples):
    `SearchEquivalence::insert_projected` `0.00% → 8.33%`,
    `Mana::spending_choices` `7.59% → 0.06%` (`pop_next_pip` `0.10%`,
    `spend_one` `0.02%`), `[(ManaClass, [u8; 6])]::hash_slice` `8.33% → 3.99%`,
    `CardPredicate::hash` `4.89% → 2.51%`. Digest moved (`3ad67dc7` → `fe24b40f`)
    solely because `bench_states` folds `payment.witness_digest` and fewer
    redundant `plan()` calls ran; debug oracles confirm unchanged legality.
  - `entry-choices --sizes 4,8,16`: **10.342B** inst (29.93B pre-Slice-15),
    linear `priority/step:activate` (`69 → 110 → 188`) and `priority/search`
    (`6,543 → 9,513 → 14,311`).
  - `changeling-anthems --sizes 4,8,16`: **9.393B** inst (80.6B pre-Slice-15),
    0 payment calls on all single-step layers.
  - `storm-copies --sizes 4,8` (**1.576B**), `genesis-wave --sizes 4,8,16`
    (**0.490B**), `ventmaw-pools --sizes 4,8,16` (**0.223B**): all flat with 0
    payment search calls; `float-origins --sizes 4,8` (**2.165B**).
- **Debug invariants & semantics:** `61/61` debug `flow::mana::` tests, `14/14`
  debug `managym-state` `mana` tests, full `CI/Rust Tests`, and debug
  `bench_states` with search/dedup oracles (`mixed-pools --sizes 2,3`,
  `entry-choices --sizes 4,8`, `changeling-anthems --sizes 4,8`,
  `ventmaw-pools --sizes 4,8`, `float-origins --sizes 4`) all passed with zero
  assertions. Audited `fd3596f409`'s unified `SourceClassCache`: both callers
  (`mana_ability_actions`, `useful_payment_taps`) cache only pure feasibility
  results while emitting every source's own `Action`.
- **Negative results / open bottlenecks ruled out for MTG-431 (0% curated-deck impact):**
  - `mixed-pools` remaining `CardPredicate::hash` (`2.51%` of `mixed-pools`
    size 16) occurs only inside small pre-`GROWTH` plans before feasibility
    projection triggers; total `mixed-pools --sizes 4,8,16` is `0.40 s` user.
  - `changeling-anthems` `cast-choice/search` (`1,384 → 6,061 → 19,158` calls:
    inherent layer board + `automatic_upfront_path`) and `entry-choices`
    `cast-choice/search` (`8,999 → 24,891 → 55,810` calls: requires cross-step
    payment-model reuse).

Review note for gate: Slices 16–17 and `lf compress` (`fd3596f409`, which
already deleted ~125 lines of duplicate exact-rider projection) left
`fixed_sufficient`, unified `SourceClassCache`, sibling pruning, and
`PurposeProjection` in `managym/src/flow/mana.rs`; all four are supported by
the measured `bench_steps`/`bench_search`/`bench_states` gains above and
verified by debug oracles.

Behavior review: no open defects. Nits, not blocking: Academy shows Retarget
as "Choose a target for …" and `KeepPermanents` as "Choose from the revealed
cards".

## Coverage decomposition at `47e21050c2`: no further slices

Review criterion: the representative cards compose from small orthogonal
primitives, with no card-specific engine logic, multi-axis special variant or
duplicate state. The batch meets it, so no refactoring slice is planned.

- Stack copies: Thousand-Year Storm is `Repeat { count:
  SpellsCastBeforeTriggering, effect: CopyTriggeringObject }`; Teach by Example
  is a delayed trigger over the same copy verb; Bolt Bend is the existing
  `ChangeTarget`. One copy owner (`CopiableSpell`), one retarget owner
  (`flow/retarget.rs`).
- Targets: `TargetRelation` on the clause, `ObjectSelector::ClauseTarget(n)`,
  `Simultaneously` and `Fight { first, second }` carry Drakuseth, Fiery
  Annihilation, Run Away Together, Indrik and Primal Might.
- Library and selection: `LibraryTop`, `Unselected`, `SelectCards.max_mana_value`
  and `GrantPlay { what, until }` carry Genesis Wave, Plunderer and Raider and
  replaced `ExileTopCardsToPlay` and `GrantGraveyardCastThisTurn`.
- Entry choices: one `Permanent.chosen: EntryChoice` read by predicates, mana
  output and the type layer. A second chosen-type land (cast-only) is
  `creature_purposes: [Cast]` with no engine change.
- Costs and values: `PaymentCost::All`, `ManaExpiry`, `CardPredicate.keyword`,
  `EachPlayerSacrifices { predicate }`.
- The engine diff names no card (checked by grep over added lines).

Single-consumer variants left as they are, because splitting them now would
guess at a second card. Split each when that card arrives:

- `Effect::SeparatePiles { look }` (Curator). The shape then is `SelectCards`
  over `LibraryTop` with `Selected`/`Unselected` as the piles and a `ForPlayer`
  pile choice.
- `TargetRequirement.up_to_if_kicked` (Divine Resilience).
- `ReplacementEffect::AsEntersChoose.look_at_opponent_hand` (Spyglass).
- `PrintedCharacteristics.division` (Biogenic Upgrade); it moves onto the
  divided clause when a spell divides among one clause of several.

## Follow-ups outside MTG-431 (separate Tasks)

- Retire `Effect::TargetCreaturesDealPowerDamageToLastTarget` (Felling Blow,
  Bite Down, Rocky Rebuke, Allies at Last) onto
  `DealDamage { by }` over `ClauseTarget`. Not done here: Allies at Last needs
  a per-source amount read, two TLA definitions change their learning tokens
  (pinned-pilot rollover), and `agent/heuristic/knowledge.rs` matches the
  variant by name.

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

11. Concept alignment: Retarget projects chosen `"new target"`s; one `Player.mana_pool` with `ManaExpiry` (combat pool deleted, additive `expiry` beside derived `combat`, `MATCH_STATE_HASH_VERSION` 33); group clicks skipped in decision memory; `PartitionPiles`/`ChoosePile` are `Modal`; `keep_permanents` stage (`keep_creatures` still read); Academy prompt lines — `e1f42473b0`.
12. Behavior-review defects: Azula fixture preserves firebending `EndOfCombat` mana alongside step `{R}`; `text_observation._cards_lines` renders `put_into_graveyard_this_turn` and `_TEXT_FIELDS` covers `Card.put_into_graveyard_this_turn` and `PooledManaClass.expiry`; verified all 14 frozen semantic-kernel traces match base `3be5a0cfb2` identically on Commands and event boundaries (only `state_hash` `v31`→`v33`, `pending_choice_digest`, and `content_digest` moved), re-recorded `evidence-battlefield-history-v1.json`, rebuilt the release extension, and regenerated the authored-match authority and parity receipts (all 174 commands, presentation events, and terminal board identical; event diffs are pure `card_def_id` renumbering).
13. Layer performance: per-object dirty bits (`FINISH_TYPES` / `FINISH_ACTIVATIONS` / `FINISH_STATICS`) skip unchanged `finish_object` work (uncached debug oracle still forces `FINISH_ALL`); `PredicateSource` guarded by `reads_source()` in `layers::Context` and `Game::object_matches_with_source`; O(1) `This`/`Attached` targets and single-pass `objects.iter()` in `Context::targets` — `1811194e41`.
14. Payment-search performance: allocation-free `PaymentResources::cmp_source` + bit-packed `PaymentResources::hash`, `Arc<SpendPredicates>` precomputed hash digest on `ManaRestriction::Only` (byte-identical serde) and `ManaChoices` digest on `Obligation::Output`, single-pass `optimistic_supply` and per-`SpendingContext` `PurposeSupply` memo in `PaymentSearch` — `a17915a437`. Engine Budget beats CI reference `7d3ac1eb1c` (`bench_steps` -1.74% instructions, `bench_search` -4.85% instructions) with identical digests; `entry-choices` (`size=16`) `step:activate` 8.0× faster (`7.80`→`0.98 ms`) and `search` 3.43× faster (`789.64`→`230.19 ms`); `changeling-anthems` (`size=16`) `step:choice` 2.20× faster (`6.00`→`2.73 ms`) and `search` 1.89× faster (`2,080.77`→`1,098.88 ms`).
15. Payment-search equivalence: purpose-projected dedup in `Game::mana_ability_actions` and `PaymentSearch::groups_for`/`CanonicalState`, `PaymentResources` bitmasks, `FastHasher` — `e8ed428d2c`.
16. Entry-choice layer mana resolution, `useful_payment_taps` dedup/cache, `fixed_sufficient` feasibility fast-path, DFS sibling pruning & lean `PurposeProjection` / boxed `SearchEquivalence` — `entry-choices` `-40.1%` (`16.92B→10.14B`, linear `priority step:activate`), `changeling-anthems` `-53.0%` (`19.96B→9.39B`, 0-search `damage actions` / `step:choice`), `bench_steps`/`bench_search` digests identical (`45.97B` / `980.95B`).
17. Mana refinements and PR #605 CI fixes: `ProjectedClass` deleted (feasibility projection over `PurposeMask`), `EmptyDisposition { Drop, Keep, Recolor }` step-end emptying, `ManaTypeSet(u8)` pip mask, `CostPermissionContext`; v11 layout/learn-contract tests and Academy fixtures refreshed — `a8818a8ab8`, `bdd839c313`.

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
