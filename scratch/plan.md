# MTG-431 — Foundations stack copies, choices, costs and procedure tail

All 56 of the 56 scoped Foundations rows have directed implementation evidence.
Slices 1–16 are complete. Demonic Pact and Maze’s End already had working
definitions.

Every one of the 56 scoped ledger rows is `implemented`; Abyssal Harvester's
`blocked_by` is cleared. Slices 1–17 (including Slice 11's four concept
defects, Slice 12's three behavior-review defects, Performance Slices 13–16,
Academy's holistic Curator of Destinies pile-partition UI (`cfa110e6d8`), and
Slice 17's mana refinements + PR #605 CI fixes) are built and verified.

**Next: `pr-publish` → CI (`Engine Budget`, `Academy UI`, `Integration`, `Academy Integration`) → `demo`.**

### Completed Slice 17 — Mana refinements & PR #605 CI fixes

1. **Feasibility-only `ProjectedClass` stripping:**
   - Deleted `ProjectedClass { expiry, restriction, effect }` in `managym/src/flow/mana.rs`; `ProjectedClasses` now carries `PurposeMask` directly and merges unrestricted non-ordinary classes (and restrictions admitting all reachable purposes) straight into `ProjectedMana.mana`.
   - Replaced `obligation_has_restriction` / `state_has_restriction` / `model_has_restriction` in `PaymentSearch` with non-ordinary class checks (`!mana.classes().is_empty()`) and routed all states to `projected_visited` while projection is active so ordinary and unrestricted classed pools collapse into one table, while keeping `SourceClassCache` exact on `ManaExpiry`/`SpendEffect` for player-visible offers.
   - Deduplicated feasibility-only `spending_choices` inside `PaymentModel::witness` per `(Option<&ManaRestriction>, color)` using `ManaCost::pop_next_pip` + `Mana::spend_one` (no heap unit-`Mana` allocations).
2. **Class-level `Drop | Keep | Recolor(Color)` step-end emptying:**
   - Added `EmptyDisposition { Drop, Keep, Recolor(Color) }`, `EmptySummary`, and single-pass `ManaPool::empty_expiring` in `managym/state/src/mana.rs`, deleting `ManaPool::split_expiring`, `become_red`, `red_only`, `Mana::become_red`, `Mana::red_only`, and `Game::unspent_mana_retained`.
3. **Unified 6-bit `ManaTypeSet(u8)` pip mask & `CostPermissionContext` bundle:**
   - Added `ManaTypeSet(u8)` and `ManaCost::pop_next_pip` in `managym/state/src/mana.rs`; unified `Mana::usable_colors` across `can_pay`/`spendable_units` (with a hybrid-free fast path), `spending_choices`, `paid_with`, `PaymentModel::colors_coverable`, and convoker color masks in `managym/src/flow/mana/help.rs`.
   - Added copyable `CostPermissionContext { help, waterbend, reserved_tap }` in `managym/src/flow/payment.rs`, deleting parallel `(help, waterbend, also_tapped)` plumbing and `#[allow(clippy::too_many_arguments)]` across `help.rs`, `mana.rs`, `payment.rs`, `play.rs`, and `decision.rs`.
4. **PR #605 CI fixes:**
   - Updated `tests/env/test_named_observation_layout.py` for v11 `FactKind`s 17–19 (`ChosenColor`, `ChosenSubtype`, `ChosenName`).
   - Updated `tests/semantic/test_learn_contract.py` to read named `outside`/`valid` columns instead of hardcoded `-1`/`-2` indices after appended column 56 (`put_into_graveyard_this_turn`).
   - Rebuilt release `_managym` extension and refreshed `managym/academy/fixtures/*.json` via `scripts/refresh_academy_fixtures.py` (fixing the stale runtime identity / projection digest failures in `Academy Integration` and `Academy UI`).
5. **Checks run (minimal local checks under high host load; heavy release benchmarks & Playwright E2E deferred to CI):**
   - Focused debug `nextest` (`managym-state`, `flow::mana::`, `tla::fire_lord_azula|ozai|ventmaw`): all passed.
   - Debug `bench_states` with debug search oracles (`mixed-pools --sizes 2,3`, `entry-choices --sizes 4,8`, `changeling-anthems --sizes 4,8`): all passed with zero assertions.
   - `cargo fmt --manifest-path managym/Cargo.toml --all --check`: clean.
   - Rebuilt release `_managym` + refreshed Academy fixtures + targeted pytest (`test_named_observation_layout.py`, `test_learn_contract.py`, `test_pinned_pilot.py`, `test_choice_navigation.py`, `test_learn_setup.py`, `test_learn_evidence.py`, `test_decision_surface.py`): 91 passed (only 2 pre-existing base-vs-`origin/main` diffs already fixed in `80ef548bf5` on `main` and passing in CI).
   - Deferred to CI under high host load (~100+): release `bench_steps`/`bench_search` (`Engine Budget`), release `bench_states`, full workspace `nextest`, and Playwright `Academy UI`.

## Engine measure pass 4 (pre-Slice-17 baseline at `ff0f872c3e`, Apple M4 Max, contended host)

No Rust files changed between `04a8f55e6d` (Pass 3) and `ff0f872c3e`. Fresh
uninstrumented release measurements (`/usr/bin/time -l`, host load `67–110`)
and `/usr/bin/sample` CPU profiles confirm Pass 3's base comparison against
`fcc96be003` and establish the pre-Slice-17 profile:

- **Curated benchmarks (uninstrumented release, all digests & counts match `fcc96be003` and `managym/perf/baseline.json`):**
  - `bench_steps`: `512` games, `0` capped, `118,559` steps, digest
    `10601622128491155944`; **45.961B** instructions retired (vs base
    `fcc96be003` `46.731B`, **-1.65%**; Pass 3 C/R/R/C mean `45.707B`,
    **-2.19%**), `14.708B` cycles, wall `35.00s` (sample run `22.11s`, 15,700
    involuntary ctx switches).
  - `bench_search`: seeds 1–3 digests `13638875039089371439`
    (`score_digest_blake3` `2f29dd10...`), `5466143068363283977`
    (`09de41e9...`), `9998925852575961750` (`9692421a...`), `0` cap hits,
    `7,680 / 8,064 / 7,424` simulations; **975.339B** instructions retired
    (vs base `fcc96be003` `1,027.552B`, **-5.08%**; Pass 3 C/R/R/C mean
    `972.648B`, **-5.34%**), `275.306B` cycles, wall `130.03s` (`worst_overall_s`
    `4.975s`).
  - **CPU profiles (`/usr/bin/sample`, 1ms):**
    - `bench_steps` (6,493 samples): `Observation::new` `39.32%`
      (`populate_permanents` `13.86%`, `populate_cards` `8.13%`, `player_data`
      `7.67%`), `layer_board` `18.44%` (`layers::Context::evaluate` `15.22%`),
      `drain_pending_events` `18.37%` (`indexed_sources` `17.83%`),
      `priority_action_space` `13.09%`, `finish_mana_payment` `5.71%`,
      `payment_model` `5.16%`. `PaymentModel::witness`, `Mana::can_pay`,
      `Mana::spending_choices`, `clear_mana_pools`, and hybrid branching are
      all `< 0.1%`.
    - `bench_search` (12,158 samples): `layer_board` `35.24%`
      (`layers::Context::evaluate` `28.97%`, `layers::Context::apply` `9.89%`),
      `indexed_sources`/`drain_pending_events` `31.31%`
      (`collect_waiting_triggers` `17.97%`), `priority_action_space` `25.70%`,
      `advance_event_batch` `21.39%`, `payment_model` `8.75%`
      (`Vec<ManaSource>` `5.80%`), `finish_mana_payment` `8.41%`
      (`advance_mana_spend` `6.05%`), `PaymentModel::payable` `1.78%`
      (`direct_payable` `1.20%`), `Mana::can_pay` `0.32%`.
      `PaymentModel::witness`, `ProjectedClass`, `clear_mana_pools`, and hybrid
      branching are all `< 0.15%`.
- **Stress states (`bench_states --features benchmark`, release):**
  - Existing MTG-431 / mana shapes reproduce Pass 3: `ventmaw-pools --sizes 4,8,16`
    (**0.244B** inst, **0** payment search calls — fixed sources + single
    `EndOfTurn` class), `entry-choices --sizes 4,8,16` (**10.099B** inst,
    `cast-choice search` `38,123` states), `changeling-anthems --sizes 4,8,16`
    (**9.363B** inst, `cast-choice search` `14,233` states), `storm-copies --sizes 2,4,8`
    (**1.665B**), `genesis-wave --sizes 4,8,16` (**0.492B**), `diverse-mana --sizes 4,8,16`
    (**3.724B**), `restricted-filter` + `distinct-restrictions` (**3.994B**),
    `float-origins --sizes 4,8,16` (**8.439B** inst, `cast-choice search`
    `81,566` states; 529-sample profile: `witness` `65.22%`,
    `PaymentSearch::insert` `21.36%`, `insert_projected` `8.32%`).
  - **Added `mixed-pools` stress state (`managym/examples/bench_states.rs`):**
    Because `ventmaw-pools` makes 0 payment search calls and no existing state
    combined multiple `ManaExpiry` (`Step`, `EndOfCombat`, `EndOfTurn`) and
    `SpendEffect` (`CopySpell`, `SpellGainsKeyword`,
    `CreatureEntersWithPlusOneCounter`) classes inside `PaymentSearch`, added
    `mixed-pools` (`Savage Ventmaw`, `Firebending Student`, `Mountain`,
    `Pyromancer's Goggles`, `Carnelian Orb of Dragonkind`, `Animal Attendant`,
    `N` `Prophetic Prism`s, casting `Arcane Epiphany` / `Claws Out` /
    `Abandon Attachments` / `Pillar Launch` at `priority` and `pooled`).
    - Default `--sizes 2,3,4`: **1.328B** inst, `pooled search`
      `calls 4,645 → 7,302 → 9,553`, `states 2,977 → 4,818 → 6,362`
      (`maxc 24 → 28 → 32`), digest
      `3ad67dc77c5de7823e0ed48a3c73e523fc6c5c677a388ff11996558dea4ecb79`.
    - `--sizes 4,8,16`: **4.497B** inst, `pooled actions`
      `states 42 → 66 → 114` (`maxc 32 → 48 → 80`), `pooled search`
      `plans 680 → 710 → 879`, `calls 9,553 → 16,110 → 35,239`,
      `states 6,362 → 11,192 → 25,196`.
    - CPU profile (`mixed-pools --sizes 16`, 1,740 samples): `PaymentModel::witness`
      `53.62%`, `PaymentSearch::insert` `17.76%` (`visited` `HashMap::insert`
      `13.91%`, `Arc<SearchState>::hash` `11.03%`,
      `[(ManaClass, [u8; 6])]::hash_slice` `8.33%`, `CardPredicate::hash`
      inside `SpendEffect` `4.89%`), `SearchEquivalence::insert_projected`
      `0.00%`, `Mana::spending_choices` + iteration `7.59%`.
- **No additional slices planned beyond Slice 17:** curated-deck profiles show
  no new MTG-431 bottleneck; Slice 17 item 1 directly targets the measured
  `mixed-pools` feasibility-projection bottleneck, while items 2–3 are tracked
  as non-bottleneck architectural cleanups guarded against regression.
- **Deferred unavailable hardware (no x86_64 speed claim):** Ubuntu `x86_64`
  Engine Budget runner — verify on CI via PR #605 or:
  `gh workflow run ci.yml --ref jack/mtg-431-close-foundations-stack-copies-choices-costs -f measure=<commit>`.

## Performance outcome (Slices 13–16 + compress, Apple M4 Max, contended host)

Engine performance review pass 3 at `04a8f55e6d` (post-`7c75706ef1` compress),
uninstrumented release C/R/R/C via `/usr/bin/time -l`, vs checkout base
`fcc96be003` (and CI reference `7d3ac1eb1c`). Earlier notes at `5600d6b8a6`
predated `7c75706ef1` and lacked post-Slice-16 CPU profiles and `fcc96be003`
comparison. All counts, digests, `score_digest_blake3` and `action_counts`
match base and `managym/perf/baseline.json` identically (`bench_steps`: 512
games, 118,559 steps, digest `10601622128491155944`; `bench_search`: seeds 1–3
`13638875039089371439`, `5466143068363283977`, `9998925852575961750`, 0 cap
hits; no `benchmark`-feature instrumentation keys):

- **Supported gains (`bench_steps` / `bench_search`):**
  - `bench_steps`: `45.711B, 45.703B` (mean **45.707B** instructions) vs base
    `46.735B, 46.727B` (mean **46.731B**, **-2.19%**); cycles `10.752B, 10.772B`
    (mean **10.762B**) vs base `11.079B, 11.080B` (mean **11.080B**, **-2.87%**);
    wall `3.09s / 2.93s` vs base `3.33s / 3.09s`. Beats Slice 16 target
    (`≤ 45.9B`) and pre-Slice-15 (`45.868B`).
  - `bench_search`: `972.394B, 972.902B` (mean **972.648B** instructions) vs
    base `1,027.349B, 1,027.755B` (mean **1,027.552B**, **-5.34%**); cycles
    `228.787B, 240.468B` (mean **234.628B**) vs base `234.971B, 245.433B`
    (mean **240.202B**, **-2.32%**). Beats Slice 16 target (`≤ 977B`) and
    pre-Slice-15 (`977.34B`); `7c75706ef1` compress + fresh build recovered an
    extra -8.3B instructions over pre-compress `082e16fb37` (`980.95B`).
  - Paired `/usr/bin/sample` (`cand-search` 10,071 vs `base-search` 9,228
    samples) confirms layer gains (`layer_board` `39.03% → 35.13%`,
    `layers::Context::evaluate` `32.95% → 27.86%`) with no new hot path
    regressing curated decks.
- **Supported gains (`bench_states`, MTG-431 shapes):**
  - `entry-choices --sizes 4,8,16`: `29.93B` (pre-15) → `16.92B` (Slice 15) →
    **10.091B** instructions (**3.0× faster**). Pass 2's three hotspots in
    `entry16` (`/usr/bin/sample`, 3,206 samples) dropped sharply:
    `useful_payment_taps` `19.10% → 8.42%`, `finalize_characteristics`
    `10.83% → 1.56%`, `SourceClassCache`/`sources_equivalent` `9.59% → 1.47%`.
    All single-step layers (`actions`, `step:activate`, `vector`) are linear
    (`0.04–0.22 ms`, `priority step:activate` states `34→68→141`); `priority`
    and `block` search are linear (`2.0–2.1×` for `2×` size).
  - `changeling-anthems --sizes 4,8,16`: `80.55B` (pre-15) → `19.96B` (Slice
    15) → **9.348B** instructions (**8.6× faster**). All single-step layers
    have zero payment search and `< 0.5 ms`; `changeling16` (`/usr/bin/sample`,
    3,436 samples) shows `PaymentModel::witness` down `58.61% → 15.54%`,
    `mana_ability_actions` `23.64% → 7.89%`, `mana_choice_completes_payment`
    `21.93% → 1.83%`.
  - `storm-copies --sizes 2,4,8` (**1.662B**), `genesis-wave --sizes 4,8,16`
    (**0.489B**), `ventmaw-pools --sizes 4,8,16` (**0.225B**): 0 payment search
    calls, linear/flat scaling. Pre-existing shapes (`diverse-mana`,
    `restricted-filter`, `distinct-restrictions`, `lathril-elves`,
    `convoke-diverse`, `mixed`) match their documented residuals.
- **Open bottlenecks & ruled-out further iteration:**
  - `changeling-anthems` `cast-choice search` (`8.92→31.76→67.33 ms`,
    `plans 152→302→441`, `states 998→4,455→14,233`, `3.19×` for `2×` size,
    down 10× from Slice 15's `142,189` states): `66.5%` of `changeling16` is
    now inherent 32-changeling × 8-anthem `layer_board` + 16×16
    `blocker_can_block_attacker` (`29.8%`); `automatic_upfront_path`
    (`SourcePolicy::AutomaticPrefix`) accounts for `18.8%` (`~1.7B`
    instructions) because iterative-deepening prefix search runs before
    `GROWTH = 16` triggers equivalence pruning. Ruled out for MTG-431: 0%
    curated-deck impact and not worth adding more complexity to `mana.rs`.
  - `entry-choices` `cast-choice search` (`7.54→28.09→78.13 ms`,
    `plans 1,119→2,298→3,534`, `states 4,571→13,644→38,123`, `2.79×` for `2×`
    size, `maxc 27→41→49`): bounded polynomial cost from `N` pairwise-distinct
    chosen creature-type restrictions (`plans 1.54×` × `maxc 1.20×`); residual
    `entry16` CPU (`mana_ability_actions` `39.4%`, `payment_model` `19.5%`)
    requires cross-step payment-model/answer reuse (separate architectural
    design, matching `mixed`/`counted-tap`).
  - No negative curated-deck results remain; deferred to gate/CI: Ubuntu
    `x86_64` Engine Budget (`engine-budget.yml`), full workspace debug
    `nextest`, extension rebuild / Academy fixture refresh, and Python
    admission + pinned-pilot checks.
- **Debug invariants at `04a8f55e6d`:** `286/286` focused debug `nextest` tests
  (`flow::mana::`, `flow::layers`, `flow::statics`, `fdn::`, `entry_choice`,
  `changeling`, `retarget`, `storm`), `2/2` debug `bench_states` self-tests,
  and debug `bench_states` runs on `entry-choices --sizes 4,8` and
  `changeling-anthems --sizes 4,8` (exercising `fixed_sufficient`,
  `useful_payment_taps`, `mana_ability_actions`, and layer dirty-bit debug
  oracles) all pass.
- **Pre-publish validation:** full workspace debug `nextest` (`2,179/2,179`,
  0 failures), `cargo fmt --all --check`, rebuilt release `_managym` extension,
  fixed `tests/verify/test_native_observation_contract.py` (`card::VALID` vs
  appended column 56), `197 passed` across native observation contract,
  `test_pinned_pilot.py`, Elemental/evaluation/mirror/monitor/separability
  admission and `test_text_observation.py`, `52 passed` Academy protocol &
  launcher tests, `56 passed` package boundaries & public API contract,
  `scripts/public_api.py check`, `scripts/check_loopflow_workflows.py`, and
  `academy/ui` (`660/660` Vitest + `svelte-check` 0 errors/warnings).

Review note for gate: Slice 16 + compress (`7c75706ef1`) grew
`managym/src/flow/mana.rs` by ~720 net lines (`fixed_sufficient` constructive
checks with a debug search oracle, `SourceClassCache` dedup oracle, sibling
pruning, `ProjectedClasses`). Gate's readability pass should check each fast
path still earns its place against the measurements above, deleting any that
the numbers do not support.

Behavior review at `6af8463ff9`: no open defects. The 16 debug failures found
at `1278923dc6` (15 frozen semantic-kernel traces, Azula fixture) and the two
LLM-text field failures are fixed; after the compress commits,
`tla::fire_lord_azula*` (2 passed) and `tests/semantic/test_text_observation.py`
(103 passed) still pass. Remaining nits, not blocking: Academy shows Retarget
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
