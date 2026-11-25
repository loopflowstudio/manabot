# MTG-431 — Foundations stack copies, choices, costs and procedure tail

All 56 of the 56 scoped Foundations rows have directed implementation evidence.
Slices 1–12 are complete. Demonic Pact and Maze’s End already had working
definitions.

Every one of the 56 scoped ledger rows is `implemented`; Abyssal Harvester's
`blocked_by` is cleared. Slice 11's four concept defects and Slice 12's three
behavior-review defects are built and verified.

**Next: Performance Slices 13–14, then Gate / Integrated validation** (below).

## Engine measurement at `5f3d0d81b5` (Apple M4 Max, 16 cores, contended host load ~14)

### Uninstrumented release Engine Budget vs CI reference `7d3ac1eb1c` (C/R/R/C)

Both commits were built uninstrumented (`cargo build --release --locked`, without `--features benchmark`) using this branch's `bench_steps.rs`, `bench_search.rs`, and `curated/`, and measured with `/usr/bin/time -l` for load-invariant retired instructions and cycles alongside wall time:

| Benchmark | Metric | Reference `7d3ac1eb1c` | Candidate `5f3d0d81b5` | Delta |
|---|---|---:|---:|---:|
| `bench_steps` | Steps / digest | 118,559 / `10601622128491155944` | 118,559 / `10601622128491155944` | match |
| `bench_steps` | Instructions retired | 46.741B, 46.721B (mean 46.731B) | 47.755B, 47.731B (mean 47.743B) | **+2.17%** |
| `bench_steps` | Cycles elapsed | 10.968B, 10.989B (mean 10.978B) | 11.296B, 11.290B (mean 11.293B) | **+2.86%** |
| `bench_steps` | Wall time (median) | 2.897s, 2.936s (2.916s; 40,651/s) | 3.152s, 3.055s (3.103s; 38,207/s) | +6.4% (contended) |
| `bench_search` | Seed 1–3 digests | `13638875039089371439`, `5466143068363283977`, `9998925852575961750` | identical (0 cap hits) | match |
| `bench_search` | Instructions retired | 1,027.48B, 1,027.71B (mean 1,027.59B) | 1,063.97B, 1,063.91B (mean 1,063.94B) | **+3.54%** |
| `bench_search` | Cycles elapsed | 233.11B, 237.56B (mean 235.33B) | 246.57B, 245.86B (mean 246.21B) | **+4.62%** |
| `bench_search` | Wall total / worst | 60.307s, 78.591s (contended) / 1.421s, 2.224s | 64.067s, 61.696s / 1.552s, 1.473s | wall noisy under load |

CPU sampling profiles (`/usr/bin/sample`, 1ms interval: 11,343 cand vs 10,540 ref samples on `bench_search`; 2,694 cand vs 2,296 ref samples on `bench_steps`) show the entire regression concentrates in `layers::Context::evaluate` (+2.79pp in `bench_search`, +3.41pp in `bench_steps`), specifically:

1. `layers::Context::finish_object` (`7.64%` vs `6.07%`, **+1.57pp** in `bench_search`, 99.4% called from `evaluate` rather than `seed`): `CreatureTypes::among` is **5.93%** of all `bench_search` (+1.53pp), with `is_creature_type` / `subtype_category` at **4.29%** (+1.07pp) and `memcmp` at **4.21%** (+0.99pp; +1.01pp in `bench_steps`) after `SUBTYPES` expanded to all 367 CR 205.3m creature types.
2. `layers::Context::matches` jumped **14×** from `0.11%` to **`1.51%`** (+1.39pp in `bench_search`; `0.04%` → `0.74%` in `bench_steps`), lifting `layers::instructions` (+0.59pp), `layers::targets` (+0.54pp), and `permanent_matches` (+0.74pp).

### Large-board stress states (`bench_states --release --features benchmark`)

Fixed three stale `benchmark_game_mut` calls in `managym/examples/bench_states.rs` and added the four missing MTG-431 stress shapes (`entry-choices`, `changeling-anthems`, `genesis-wave`, `ventmaw-pools`; both `bench_states` self-tests pass). Key measurements:

- **Flat / linear (no bottleneck):**
  - `storm-copies` (`n=4,8`): `retarget-1..8 step:choice` 0.01 → 0.02 ms; `retarget-8 search` 0.88 → 2.55 ms; `actions` 0.00 → 0.01 ms.
  - `genesis-wave` (`n=4,8,16`): `resolution step:resolve` 0.01 → 0.01 → 0.02 ms; `after-resolution step:choice` 0.01 → 0.01 → 0.02 ms; `after-resolution search` 0.95 → 1.23 → 1.99 ms.
  - `ventmaw-pools` (`n=4,8`): `pooled actions` 0.00 ms; `pooled step:cast` 0.01 ms; `pooled search` 0.42 → 0.25 ms.
- **Profiled bottlenecks:**
  - `changeling-anthems` (`n=4,8,16`, `N` `Three Tree Mascot`s + 8 `Adaptive Automaton`s + 2 `Secluded Courtyard`s): `characteristics` is linear and tiny (`0.03 → 0.07 → 0.10 ms`), but `cast-choice step:choice` grows 28× (`0.21 ms` / 118 calls → `1.17 ms` / 354 calls → `6.00 ms` / 1,210 calls) and `cast-choice search` hits **`2,080.77 ms`** (19,310 plans, 616,291 calls). `sample` (5,984 samples): **30.0%** self time in `Sip13Rounds`/`DefaultHasher` hashing `Arc<PaymentSearchState>`, **~17%** in `PaymentSearch::canonical` + `RawVecInner::finish_grow` + `insertion_sort_shift_left`, **8.8%** in `PaymentModel::witness`, **8.2%** in `memcmp`.
  - `entry-choices` (`n=4,8,16`, `N` distinct `Secluded Courtyard`s + 8 `Heraldic Banner`s + 8 `Adaptive Automaton`s): `priority step:activate` grows 60× (`0.13 ms` / 155 calls → `0.61 ms` / 384 calls → `7.80 ms` / 944 calls) and `cast-choice search` hits **`789.64 ms`** (21,280 plans, 186,044 calls). `sample` (1,806 samples): **~26%** self time in `ManaRestriction::allows` (6.81%) + `SpendPredicate::matches` (6.59%) + `SpendingObject::has_subtype` (3.60%) + `memcmp` (8.97%), plus **16.6%** in `Sip13Rounds`/`DefaultHasher` and **6.4%** in `optimistic_pool` / `cost_coverable`.

### Planned performance slices

13. **Skip unchanged `finish_object` work and guard `PredicateSource` in `layers.rs` (Engine Budget).**
    - *Hypothesis:* In `layers::Context::evaluate`, objects modified only in Layers 5–7 (counters, P/T anthems, color/keyword grants) already have final `is_animated`, `creature_types`, and `SpendingContext::Activate` from `seed()`, so tracking type/subtype/ability-grant mutations skips `CreatureTypes::among` (its 392-entry `SUBTYPES` binary searches and `Vec<String>` allocation/sort/dedup) on the vast majority of objects. Guarding `PredicateSource` construction in `layers::Context::matches` (and `Game::object_matches_with_source`) on `predicate.reads_source()` avoids a redundant `BTreeMap` lookup (`self.objects[&s]`) and `live_permanent(s)` fetch on every non-source-reading static target check. Together these recover the +2.2%–3.5% retired-instruction regression and are expected to bring `bench_search` below reference `7d3ac1eb1c`.
    - *Deletes:* Unconditional `finish_object` pass over Layer 5–7-only modified objects in `Context::evaluate`; unconditional `PredicateSource` construction in `Context::matches`.
    - *Focused check:* Debug layer/oracle and entry-choice/Changeling tests (`CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2 --manifest-path managym/Cargo.toml -E 'test(layer) | test(changeling) | test(courtyard) | test(automaton) | test(banner) | test(spyglass)'`), preserving uncached debug-oracle equivalence; paired uninstrumented release `bench_steps` and `bench_search` via `/usr/bin/time -l` verifying identical baseline digests and instructions retired `<=` reference `7d3ac1eb1c`.

14. **Eliminate per-source `Vec<bool>` allocation in `PaymentSearch::canonical`, cache `ManaRestriction` hash digests, and collapse repeated `class.allows` scans (`changeling-anthems` / `entry-choices`).**
    - *Hypothesis:* (a) `PaymentResources::used_on(index)` allocates a `Vec<bool>` on every source comparison inside `PaymentSearch::canonical` (`windows(2)` and `sort_unstable_by_key`); replacing it with allocation-free comparison over `resources.used` removes the ~17% canonicalization/allocation hotspot in `changeling-anthems`. (b) Caching a precomputed `u64` digest on `ManaRestriction` / `ManaClass` (like `SpendingObject::digest`) removes repeated `SipHash` traversal of `BTreeSet<SpendPredicate>` strings (30.0% / 16.6% of `changeling-anthems` / `entry-choices`). (c) Combining `optimistic_pool` and `usable_total` (or pre-evaluating `class.allows(&cost.purpose)` once per `PaymentModel` ability/class for the current `cost.purpose`) in `PaymentModel::cost_coverable` cuts the ~26% `ManaRestriction::allows` / `SpendPredicate::matches` / `memcmp` hotspot on boards with many distinct `Secluded Courtyard` restrictions.
    - *Deletes:* Per-source `Vec<bool>` allocation in `PaymentResources::used_on`, per-visit string-set hashing of `ManaRestriction`/`ManaClass`, and duplicate `usable_total` scans over ability outputs in `PaymentModel::cost_coverable`.
    - *Focused check:* Debug payment/mana tests (`CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2 --manifest-path managym/Cargo.toml -E 'test(mana) | test(payment) | test(courtyard) | test(mascot)'`), `cargo test --release --features benchmark --example bench_states`, and re-running `bench_states entry-choices changeling-anthems --sizes 4,8,16` plus uninstrumented `bench_steps` and `bench_search`, preserving complete choices, native skipping, observation capacities, and exact `bench_states` / Engine Budget digests.

Note on CI runner hardware: Ubuntu `x86_64` CI Engine Budget timing cannot be measured on this macOS ARM64 host; once Slices 13–14 land, run the CI Engine Budget check (or `gh workflow run ci.yml --ref jack/mtg-431-close-foundations-stack-copies-choices-costs -f measure=$(git rev-parse HEAD)`) with no cross-architecture wall-time claim.

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
