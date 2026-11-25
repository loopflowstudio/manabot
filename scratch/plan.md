# MTG-431 — Foundations stack copies, choices, costs and procedure tail

All 56 of the 56 scoped Foundations rows have directed implementation evidence.
Slices 1–16 are complete. Demonic Pact and Maze’s End already had working
definitions.

Every one of the 56 scoped ledger rows is `implemented`; Abyssal Harvester's
`blocked_by` is cleared. Slice 11's four concept defects, Slice 12's three
behavior-review defects, and Performance Slices 13–16 are built and verified.

**Next: Gate / Integrated validation** (below).

## Performance Slice 16 verification (Apple M4 Max, contended host)

Implemented across `managym/state/src/card.rs`, `managym/src/flow/layers.rs`, and `managym/src/flow/mana.rs`:
- **Fixed entry-choice mana resolution in layers:** shared `materialize_one_color_mana` + `ActivatedAbilityDefinition::resolve_fixed_entry_mana` run once during `layers::Context::finish_object` (with `SourceSeed` tracking `chosen`), so `Secluded Courtyard`, `Heraldic Banner`, and `Uncharted Haven` keep `has_dynamic_mana_output() == false` and share immutable `LayerBoard` `Arc<Characteristics>` views across reads and pool/tap revisions.
- **Deduplicated `useful_payment_taps` + cached `after_tap`:** `PaymentModel::after_tap` retains `self.cache` (already keyed by reserved `sources`), and `Game::useful_payment_taps` deduplicates equivalent untapped fixed sources with a debug oracle (`debug_assert_eq!`) and shared `convoker_key`.
- **`FixedOnly` / fixed-sufficient feasibility fast-path:** `PaymentModel::payable` short-circuits `SourcePolicy::All` to `true` via O(N) `fixed_sufficient` (single-pass `direct_payable` without wasted `usable_total` scans when no tap split is present, plus `[Output, rest]` and `[Pay(fee), Output, rest]` constructive checks verified by a debug search oracle); `can_activate_mana_ability` and `mana_ability_actions` route through `payable`.
- **Symmetric DFS sibling pruning & lean `PurposeProjection`:** `PaymentModel::witness` skips later siblings in the same equivalence group with identical `(present, untapped, used_mask)` after an earlier sibling failed (preserving first concrete witness in source order); `SearchEquivalence` is boxed behind `Option` until `GROWTH`; restriction-free states stay in exact `visited`; `ProjectedClasses::One` + `uniform_class` + lazy `sources_equivalent_with` eliminate per-mode projection allocations and `Vec<bool>` masks in `mana_ability_actions`.

Measurements vs Pass 2 (`578fa0dd3e`) and checkout base (`6203fff202`):
- **Digests preserved identically:** `bench_steps` digest `10601622128491155944` (512 games, 118,559 steps); `bench_search` seeds 1–3 digests `13638875039089371439`, `5466143068363283977`, `9998925852575961750` (0 cap hits, identical `score_digest_blake3` and `action_counts`).
- **`bench_steps`** (uninstrumented release, `/usr/bin/time -l`): `45.932B, 46.001B` (mean **45.967B** vs `45.936B` at `578fa0dd3e`, **-1.68%** vs base `46.754B`).
- **`bench_search`** (uninstrumented release, `/usr/bin/time -l`): `979.989B, 981.906B` (mean **980.947B** vs `981.202B` at `578fa0dd3e`, **-4.54%** vs base `1,027.594B`; ~0.2% run-to-run instruction noise under heavy host contention).
- **`entry-choices --sizes 4,8,16`**: **10.14B** instructions retired (**-40.1%** vs `16.92B` at `578fa0dd3e`, **-66.1%** vs pre-Slice-15 `29.93B`); `priority step:activate` is now linear (`plans 10→12→13`, `calls 69→110→188`, `states 34→68→141` [2.00× / 2.07×], `0.045→0.092→0.216 ms`); `block search` (`n=8→16`, 48 sims) dropped to `plans 912→1,056`, `calls 8,071→15,434` (1.91×), `states 4,952→11,555` (`-58.3%` at `size=16`); `cast-choice search` (`size=16`) dropped `plans 9,800→3,534`, `states 68,722→38,123`; `characteristics` (`size=16`) dropped `0.059→0.023 ms`.
- **`changeling-anthems --sizes 4,8,16`**: **9.39B** instructions retired (**-53.0%** vs `19.96B` at `578fa0dd3e`, **-88.3%** vs pre-Slice-15 `80.55B`); `damage actions` dropped to **0 plans, 0 calls, 0 states** (was `4N+4` calls / `3N+3` states); `cast-choice step:choice` (`size=16`) dropped to **0 plans, 0 calls, 0 states** (`0.030 ms` uncontended vs `0.42 ms`); `cast-choice search` (`size=16`) dropped `plans 5,074→441` (`-91.3%`), `calls 192,077→19,158` (`-90.0%`), `states 142,189→14,233` (`-90.0%`).
- **Debug invariants verified:** 322/322 focused debug `nextest` tests (including 3 new Slice 16 tests covering layer-resolved entry-choice mana, `fixed_sufficient` zero-search feasibility + `useful_payment_taps` dedup, and `witness` sibling pruning parity against identity-keyed DFS) and 2/2 debug `bench_states` self-tests passed.
- **Deferred to gate/CI:** Ubuntu `x86_64` CI Engine Budget (`engine-budget.yml`), full workspace debug `nextest`, extension rebuild / Academy fixture refresh, and Python admission / pinned-pilot checks.

## Engine performance review pass 2 at `578fa0dd3e` (Apple M4 Max, 16 cores, contended host load ~8.7–28.3)

Earlier post-compress notes (`63d66a8417`) were missing post-compress `bench_steps`/`bench_search` retired instructions, a paired `cand-search` vs `base-search` CPU profile, and post-Slice-15 `bench_states` CPU profiles. Fresh measurements at `578fa0dd3e` vs checkout base `6203fff202` (identical Rust to `3be5a0cfb2`) and CI reference `7d3ac1eb1c` (`managym/perf/baseline.json`):

### Supported gains (uninstrumented release, C/R/R/C via `/usr/bin/time -l`)

- Identical counts and digests to both base `6203fff202` and CI reference `7d3ac1eb1c`:
  - `bench_steps` (512 games, 118,559 steps, digest `10601622128491155944`): instructions retired `45.944B, 45.927B` (mean **45.936B**) vs base `46.738B, 46.771B` (mean **46.754B**, **-1.75%**); cycles `11.165B, 11.226B` (mean **11.196B**) vs base `11.085B, 11.605B` (mean **11.345B**, **-1.31%**); wall `5.69s / 3.36s` (`22,291 / 35,294` steps/s) vs base `3.98s / 3.11s`. `63d66a8417` recovered most of `e8ed428d2c`'s `46.07–46.24B` regression (now within +0.15% of pre-Slice-15 `45.868B`).
  - `bench_search` (seeds 1–3 digests `13638875039089371439`, `5466143068363283977`, `9998925852575961750`, 0 cap hits, identical `score_digest_blake3` and `action_counts`): instructions retired `980.717B, 981.687B` (mean **981.202B**) vs base `1,027.621B, 1,027.567B` (mean **1,027.594B**, **-4.51%**); cycles `234.07B / 250.49B` (contended spike) vs base `236.35B / 239.56B`. Remains +3.86B (+0.39%) above pre-Slice-15 `977.34B`.
- Paired `/usr/bin/sample` (`cand-search` 5,675 vs `base-search` 5,872 samples) confirms Slice 13's layer gains (`layers::Context::evaluate` `32.99% → 26.01%`, `Game::layer_board` `38.49% → 32.05%`, `layers::Context::finish_object` `6.20% → 0.70%`, `subtype_category` `3.05% → 0.56%`, `is_creature_type` `3.00% → 0.56%`, `_platform_memcmp` `2.55% → 0.70%`). `mana_ability_actions` is flat (`2.01% → 2.04%`); `payment_model` shifted `7.58% → 8.90%` (+1.32pp, mostly inside `payment_sources` `4.99% → 5.97%`).
- `bench_states --sizes 4,8,16` gains from Slice 15 + `63d66a8417`:
  - `changeling-anthems`: `80.55B → 20.65B →` **19.96B** instructions retired (`5.26B` cycles, **-75.2%** vs pre-Slice-15); `cast-choice step:choice` is now linear (`plans 4→4→4`, `calls 40→72→136`, `states 29→53→101`, `0.11→0.24→0.42 ms`; digest `a24fca0e…161f5`).
  - `entry-choices`: `29.93B → 17.91B →` **16.92B** instructions retired (`3.97B` cycles, **-43.5%** vs pre-Slice-15); `cast-choice search` plans `21,280 → 9,800`, states `120,859 → 68,722` (digest `f8fb94da…970ab`).
- Debug invariants verified at `578fa0dd3e`: `bench_states` debug self-tests **2/2 passed**; debug `cargo nextest` across `flow::mana`, `flow::layers`, `fdn::`, `payment`, `changeling`, `courtyard`, `automaton`, `banner`, `mascot`, `spyglass`, `retarget`, `storm`, `genesis_wave`, `ventmaw` **312/312 passed** (including uncached layer oracle, `mana_ability_actions` dedup `debug_assert_eq!`, `PaymentCache`, `PaymentModel::direct` and `PaymentScan` checks).
- Ruled out / flat touched shapes: `storm-copies` (`n=4,8`, `step:choice ≤ 0.049 ms`, `search ≤ 5.38 ms`), `genesis-wave` (`n=4,8,16`, `step:* ≤ 0.033 ms`, `search ≤ 3.81 ms`), `ventmaw-pools` (`n=4,8`, `step:* ≤ 0.018 ms`, `search ≤ 0.82 ms`), `characteristics` on `entry-choices` (`≤ 0.059 ms`) and `changeling-anthems` (`≤ 0.158 ms`), plus existing mana/payment shapes (`restricted-filter`, `distinct-restrictions`, `diverse-mana`, `convoke-diverse`, `improvise-artifacts`, `mixed`).

### Negative results and open bottlenecks (profiled at `578fa0dd3e`)

- `entry-choices --sizes 4,8,16` (**16.92B** instructions) still has superlinear state growth on `priority step:activate` (`plans 17→23→33`, `calls 119→232→552`, `states 60→146→420` [2.43× / 2.88×; **7.0×** over 4× size], `0.16→0.31→1.13 ms`) and `block search` (`sims=48` constant at `n=8→16`, `plans 1,676→2,260`, `calls 14,952→36,651`, `states 9,232→27,682` [**3.00×** for 2× size]). Direct-PID `/usr/bin/sample` (`entry16.sample`, 49,738 samples) isolates three concrete hotspots (~40% of total CPU):
  - **`Game::useful_payment_taps` = 19.10%** (`managym/src/flow/mana.rs:3269-3346`): when `!fixed_only`, calls `model.payable_after_tap` for every untapped fixed source (`Heraldic Banner`, etc.) with `after_tap` dropping `cache: None` and zero source-equivalence deduplication.
  - **`Game::finalize_characteristics` = 10.83%** inside `payment_model` (`17.54%`; `current_mana_output` `4.31%`, `Arc<Characteristics>::clone_from_ref_in` `4.23%`): fixed-amount `Effect::AddOneColorManaValue` (`Secluded Courtyard`, `Heraldic Banner`, `Uncharted Haven`) is flagged dynamic (`has_dynamic_mana_output`), cloning `Characteristics` and rebuilding `ManaRestriction` on every read instead of resolving once from `permanent.chosen` in layer evaluation.
  - **`PurposeProjection::sources_equivalent` = 9.59%** (`7.26%` in `PaymentSearch::insert` / `classify_sources`, `2.23%` in `mana_ability_actions`; `project_choices` `8.72%`, `project_mana` `4.66%`): re-projects and heap-allocates `ProjectedMana`/`ProjectedChoices` per color mode from scratch on every search hitting `GROWTH` and every `mana_ability_actions` call.
- `changeling-anthems --sizes 4,8,16` (**19.96B** instructions — still ~43% of all 512 games of `bench_steps`) still scales linearly **per single plan** (`damage actions`: `1 plan`, `calls 20→36→68` [`4N+4`], `states 15→27→51` [`3N+3`]; `cast-choice search`: `5,074 plans`, `192,077 calls`, `142,189 states`) because `PaymentModel::payable` disables `direct_payable` whenever any source has activated abilities, so DFS chains all `N` `Three Tree Mascot` filters (listed before `Forest` on the board) even when untapped fixed sources already pay the cost. Profile (`changeling16.sample`, 935 samples): `PaymentModel::witness` **58.61%**, `mana_ability_actions` **23.64%**, `activate_mana_ability` **22.25%** (`can_activate_mana_ability` `9.95%`, `pay_mana_upfront` `12.19%`), `mana_choice_completes_payment` **21.93%**, `PaymentSearch::insert` **20.96%** (`canonicalize_projected` `9.30%`, `canonicalize_resources` `6.63%`).

### Planned performance Slice 16 (`managym/src/flow/layers.rs`, `managym/src/flow/statics.rs`, `managym/src/flow/mana.rs`, `managym/state/src/card.rs`)

16. **Resolve fixed-amount `AddOneColorManaValue` in layers, deduplicate/cache `useful_payment_taps`, add a `FixedOnly` fast-path for boolean `payable`, and prune symmetric DFS siblings / projection allocations.**
    - *Targets:*
      (a) Resolve `Effect::AddOneColorManaValue` with `EffectValue::Fixed(_)` once from `permanent.chosen` during `layers::Context::finish_object` so `has_dynamic_mana_output()` stays `false` for `Secluded Courtyard`, `Heraldic Banner` and `Uncharted Haven` (eliminating `Arc::make_mut` + `current_mana_output` on every `characteristics()` read).
      (b) Deduplicate `Game::useful_payment_taps` by fixed-source equivalence class (debug-asserting against exact `payable_after_tap`) and retain `self.cache` in `PaymentModel::after_tap` (already keyed by `sources`/`pool`/`obligations`/`life`/`policy`/`tap`).
      (c) In `PaymentModel::payable` (and boolean feasibility checks in `mana_ability_actions` / `mana_choice_completes_payment` when applicable), short-circuit `SourcePolicy::All` to `true` whenever `self.direct(&obligations, SourcePolicy::FixedOnly)` succeeds with `direct_payable == true` (O(N) arithmetic, zero DFS calls, preserving all witness order since only `bool` is returned).
      (d) In `PaymentModel::witness`, once `visited.groups` is active, skip trying a later source in the same equivalence group with identical `(present, untapped, used_mask)` after an earlier sibling in that node failed (preserving first concrete witness in source order), and fast-path uniform-restriction `PurposeProjection::project_choices` / restriction-free states in `canonicalize_projected`.
    - *Expected result & checks:* Keep identical `bench_steps` and `bench_search` digests with `bench_search` back ≤ `977B` instructions (recovering the +0.39% vs `7aed946d0d`) and `bench_steps` ≤ `45.9B`; slash `entry-choices` (`16.92B`) and `changeling-anthems` (`19.96B`) retired instructions and eliminate the superlinear `entry-choices` `priority step:activate` / `block search` growth; preserve legal actions, choices, state digests, and pass debug layer/mana/payment/card invariant oracles.
    - *Deferred to gate/CI:* Ubuntu `x86_64` CI Engine Budget (`engine-budget.yml`, no cross-architecture wall-time claim) and full workspace debug `nextest` / Python admission checks.

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
