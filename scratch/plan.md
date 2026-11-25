# MTG-430 — Planeswalkers, loyalty, empowerment and emblems (FDN + FRA)

Base `cf73d9778f`; synced main through MTG-420 (#593, `45ddf3bca7`) and
later main (`964699a20b`). MTG-420 supplies exact object identity for stacked
abilities. Live layers and triggers also resolve exact command-zone sources.

## Done (detail in commits)

- Slices 1–8 and initial compress landed planeswalkers, fixed/variable
  loyalty, attack targets & Tomik, command-zone emblems, empower Jace, Way
  grants, zero-loyalty SBA exemption, loyalty activation events/turn-state,
  divided damage, keep-one-of-each-type, delayed attack triggers, FRA lands,
  7 planeswalkers, 28 adjacent cards, unified `Fight` / `RevealUntil`, and
  closed `fdn.jsonl` / `fra.jsonl` rows (`b17ed61f62`..`bc184fc81f`).
- Pending on other authorities (ledger rows carry them): Countersculpt and
  Theorist's Sanctum (behold), Tam (resumable proliferate), Gideon's Memorial
  and Chandra, Chill of Compliance (`ManaUse` by card type), Return to the
  Light Realms (CR 303.4f non-cast Aura attachment), Loot (negative-power
  damage assignment). Open defect: no replacement diverts a
  battlefield-to-exile move, so Overwrite's diverted case is untested.

- Pass 1 (`1a2ba2c1a3`): target division separated from `X`
  (`division_remaining`, `EffectValue::TargetDivision`, v11 `action::DIVISION`
  column 31), `SpellTargeting` factored, `−X` offers filtered by target
  availability; directed witnesses in `flow/target_authority_tests.rs`.
- Pass 2 (`18e676eb64`; `sba::tests` verified in Pass 3, full suite rerun in Pass 4): removed `FactKind::EmblemAbility` and `"Emblem::Static"` /
  `"Emblem::Triggered"` record tokens (emblems observed via `agent_cards` /
  `opponent_cards` `ZONE_COMMAND` + `semantic_cards` → `catalog_tokens`);
  projected `StaticAbility::StateBasedActionDoesntApply { action: ZeroLoyalty, .. }`
  onto `Characteristics.exempt_from_zero_loyalty` in `flow/layers.rs` (`Layer::Ability`,
  covering battlefield and command-zone sources) and restored the zero-allocation
  streaming `layer_views` loop in `flow/sba.rs` with unit tests in `sba::tests`.
- Pass 3 (`c8fff2997e`): one `CombatState.attack_targets` (player or
  permanent defender, `Option` for none); attack defenders bound only in
  `structured_offer.rs`; `ZoneEntry.attack_target` replaces the `PendingEvent`
  field and `create_token` shortcut (`create_token_attacking`); death-to-exile
  and finality rewrites live in `collect_replacements()`; per-defender refusals
  in `combat_preview.rs`; Academy drafts attackers at the defending player with
  inline planeswalker retargeting. Ran `managym --lib`, `rules_tests`,
  `match_state_hash_tests`, `managym-cards --test cards` (gtc/fdn/kld/fin/fra)
  green; Academy `npm run check`/`vitest` not run (no `node_modules`).

- Pass 4 (`d2d0d8ae2e`): CI contract fixes (legacy `CARD_DIM` slice, emblem
  `registry_name()`, v11 width assertions, test widths, Farm Animals decline
  case), `tests/public_api.json` recorded, Academy fixtures refreshed,
  `-planeswalkers` conformance/authority/parity receipts. Verified: academy/ui
  svelte-check + vitest (653), debug nextest (2,184), cp313 rebuild, pinned
  pilot zero drift, Python Academy/model/world/contract suites, release
  `bench_steps` digest `10601622128491155944` unchanged.

- Behavior review at `cba01358b6` (after compress `f47bb9b457`): debug
  workspace nextest 2,184/2,184 and Academy board vitest 494/494 green; FDN's
  five planeswalkers and 99 FRA rows `implemented`, seven FRA rows pending on
  named authorities. No defects found.

- Coverage decomposition (`9b19a44635`, `92aac08cde`, `7b5536b37c`):
  delayed attack watchers on `Whenever(TriggerCondition)` with two-axis
  defender matching; impulse exile as `LibraryTop(n)` + `SelectCards` +
  `GrantPlay`; legend-rule and zero-loyalty exemptions unified as
  `StateBasedActionDoesntApply` over `Characteristics.exempt_sbas`
  (`LegendRuleDoesntApply` retired in `token_renames.rs`). Focused debug
  nextest, pinned pilot (zero drift), Academy loyalty/choice tests and
  conformance/parity receipts green (`4e4be788f7`).

- Decomposition review at `ac9ea8b330`: delayed watchers (Garruk, Curse
  Breaker −4 per-batch at opponent; The Theorist −3 per-object at you or your
  planeswalkers) both resolve through `match_trigger_wrappers` +
  `collect_occurrences` in `flow/triggers.rs`; legend-rule and zero-loyalty
  SBAs read one `Characteristics.exempt_sbas`; printed (Garruk, Veiled
  Butcher) and resolved (Essence Burn) "would die → exile" feed one
  `ReplacementEffect::ExileIfWouldDie` in `collect_replacements()`;
  `MayCastExiled.if_declined` follows the existing `if_declined` decline
  pattern and is resolution-time casting, distinct from `GrantPlay`. Impulse
  predecessor variants and `LegendRuleDoesntApply` are gone (the latter only
  in `token_renames.rs`). No card-specific engine dispatch; the
  `empower_jace` match in `ability_description.rs` names the CR 701.71
  keyword action. Five legacy `DelayedEvent` variants stay a closed set for
  0.3.0 (see questions). No findings.

- Performance slices P1–P4 (`49393aef55`..`6a08a28b0e`): no-op combat
  defender cleanup skips invalidation, `Context::targets` resolves its source
  once per node, boxed `permanent_defenders` keeps `GameEvent` at 56 bytes,
  flat `attacks_this_turn` + sparse `attacked_permanents`. Focused debug
  nextest 149/149.
- Performance slices P5–P6 (this pass):
  - P5: `TrackedState::mark_dealt_damage`, `TrackedState::record_life_change`
    (discovered during measurement: `zones.rs` `emit` mutated `turn.history[player].life_lost`
    through `DerefMut` on every unblocked player-damage hit), and fine-grained
    `revisions.counters` / `TrackedState::change_counters` / `LayerBoard::reads_counters`
    (with six `u32` revision counters keeping `DerivedCache` / `TrackedState`
    within ceilings and shrinking each by 16 bytes; `ReadSet` tracks `no_counters`,
    `with_counter`, `ObjectProperty::Counters`, `SourceCountersAtLeast`,
    `AttachedCountersAtLeast`, `CountersOnLandsYouControl`, `CreaturesWithCounters`,
    `Except::Matching`, `AtMostHalfStartingLife`, and `OpponentLostLifeThisTurnAtLeast`).
  - P6: cached `source_permanent: Option<PermanentId>` directly on layer `Node`
    at creation (`Context::node`, `Node::resolved`), removing every `permanent_of`
    lookup from `exists`, `targets`, `instructions`, `buff`, `condition`,
    `comparison_value`, `select_permanents`, `matches`, and `apply`.
  - Verified: focused debug nextest 88/88 (`managym` `statics::`, `sba::`, `layer`,
    `emblem`, `counter`, `planeswalker`, `life`) and 266/266 (`managym-cards`
    `fdn`, `fra`, `zen`, `eoe`, `heartless_act`); `bench_states attack-planeswalkers`
    (N=4/8/16/32, identical digest `e6751d56...`): `damage | step:damage` dropped
    from `3 / 7 / 15 / 31` (`O(N)`) to **`1 / 1 / 1 / 1`** (`O(1)`), `block | search`
    dropped from `185 / 462 / 1,299 / 4,620` (`35.0` boards/sim at N=32) to
    **`137 / 248 / 366 / 714`** (constant `5.4` boards/sim at N=32; 6.47× fewer boards,
    2.49× fewer allocs), `damage | search` dropped from `119 / 148 / 144 / 272` to
    **`103 / 100 / 32 / 32`** (8.5× fewer at N=32); uninstrumented release
    benchmarks (identical digests & BLAKE3): `bench_steps` `46.49B` (`46.54B`, `46.43B`,
    **-0.65%** vs `main` `46.79B`), `bench_search --layers` `16.37B` (**-5.40%** vs
    `main` `17.30B`), `bench_search` `1,001.24B` (**-2.59%** / `-26.67B` vs `main`
    `1,027.91B`, `-13.10B` vs `09b0f89882`).
- Synced main `6203fff202` (#601, Academy deck pools/trophies; Python and UI
  only, no engine or protocol files) at `7941de98ae`. No checks run after the
  merge; they fall to gate.

## Engine measurement and performance review (at `09b0f89882` vs `main` `6203fff202`)

Measured on Apple M4 Max (contended host; load-invariant `/usr/bin/time -l`
instructions retired, cycles elapsed, and 1 ms `/usr/bin/sample` CPU profiles
govern). Base `6203fff202` is current `main` (identical simulation behavior to
committed `managym/perf/baseline.json` at `7d3ac1eb1c`). Both sides built as
uninstrumented release binaries (`cargo build --release --locked --example
bench_search --example bench_steps`, no `benchmark` feature) and run after the
P1–P4 compress commit (`6841e0df92`).

### Uninstrumented release benchmarks

| Benchmark | Metric | Base `main` `6203fff202` | Pre-P1–P4 `6dd3914c73` | Candidate `09b0f89882` | vs `main` |
|---|---|---:|---:|---:|---:|
| `bench_steps` (512 games, 118,559 steps, 0 capped) | Digest | `10601622128491155944` | `10601622128491155944` | `10601622128491155944` | unchanged |
| `bench_steps` (ABBA, median of 2) | Instructions retired | `46.79B` (`46.78B`, `46.81B`) | `50.92B` (`+8.8%`) | `46.57B` (`46.57B`, `46.57B`) | **-0.48%** (`-0.23B`) |
| `bench_steps` (ABBA) | Cycles elapsed / user | `12.31B` / `6.32s` | — / `7.70s` | `11.26B` / `5.04s` | **-8.50%** cycles |
| `bench_steps --layers` (16 games, 1,308 steps) | Digest / Instructions | `1064912348938184597` / `0.650B` | — | identical / `0.651B` | `+0.15%` (flat) |
| `bench_search --layers` (seed 244, 8 dec., 1,344 sims) | Digest / Instructions | `17627308883250227726` / `17.30B` | — | identical / `16.97B` | **-1.88%** (`-13.39%` cycles) |
| `bench_search` (seeds 1–3, 40 dec., 0 caps) | Digests | `13638875039089371439` / `5466143068363283977` / `9998925852575961750` | identical | identical (full BLAKE3 match) | unchanged |
| `bench_search` | Instructions retired | `1,027.91B` | `1,206.99B` (`+17.4%`) | `1,014.34B` | **-1.32%** (`-13.58B`) |
| `bench_search` | Cycles elapsed / user | `240.60B` / `88.29s` | `367.17B` / `156.03s` | `237.94B` / `79.32s` | **-1.10%** cycles (`-10.16%` user) |

### CPU profile (`1 ms /usr/bin/sample` on uninstrumented `bench_search`, 16.5k vs 16.2k samples)

- **Supported gains (P1–P4 + compress)**: `layer_board` `40.53% → 39.26%` (`-1.27pp`),
  `Context::evaluate` `35.25% → 33.57%` (`-1.67pp`), `Context::targets`
  `2.76% → 1.18%` (`-1.58pp`), `Context::instructions` `6.26% → 4.52%` (`-1.73pp`),
  `indexed_sources` `35.18% → 34.57%` (`-0.61pp`), `drain_pending_events`
  `34.09% → 33.84%`, `characteristics` `5.53% → 5.31%`, `legend_groups`
  `1.11% → 0.97%`.
- **Open profile bottleneck (P6)**: `Context::exists` grew `0.00% → 0.36%` inclusive
  (`0.32%` self, plus `+0.60pp` in `Context::evaluate`'s `SpecFromIterNested<Vec<usize>>`)
  because changing `Node.source` from `PermanentId` to `ObjectId` causes `exists()`,
  `targets()`, `instructions()`, and `apply()` in `managym/src/flow/layers.rs` to
  re-resolve `self.game.state.permanent_of(node.source)` on every layer pass.
- **Ruled out (negative results)**:
  - `indexed_sources`, `build_trigger_index`, `drain_pending_events`,
    `commit_proposed_event`, `collect_replacements` (`0.02% → 0.07%`), and
    `remove_invalid_defenders` (`0.00%`) show no curated-workload regression.
  - `permanent_has_modification` (`1.56% → 2.12%`, `+0.57pp`): no new caller was
    added on the priority hot path (`LoyaltyAtInstantSpeed` only runs for loyalty
    abilities in `play.rs:1814`) and `Modification` kept its boxed size
    (`Box<CardPredicate>`); this is inlining/layout movement within an overall
    `-13.58B` instruction reduction.

### Stress states (`bench_states --features benchmark`, at `09b0f89882`)

- **Flagged superlinear growth — `attack-planeswalkers` (N = 4 / 8 / 16 / 32)**:
  - **`damage | step:damage`**: `LayerBoard` rebuilds grow **`O(N)` (`N - 1`)** —
    `3 / 7 / 15 / 31` boards (`353 / 430 / 426 / 756` allocs; `0.060 / 0.033 / 0.035 / 0.188 ms`,
    `5.36x` ms from 16→32) vs constant `6 / 6 / 6 / 6` boards on `duplicate-creatures`.
  - **`block | search`**: **`O(N^2)` boards and `O(N^3)` work** — `185 / 462 / 1,299 / 4,620`
    boards (`9.2 / 12.8 / 19.1 / 35.0` boards/sim), `14,745 / 31,468 / 57,281 / 183,865`
    allocs (`3.21x` from 16→32), `3.5 / 7.7 / 14.4 / 46.9 MB` (`3.26x`),
    `1.50 / 2.14 / 4.39 / 23.09 ms` (`5.26x` from 16→32). Control
    `duplicate-creatures | block | search` stays linear at `5.0 / 5.2 / 5.1 / 5.1`
    boards/sim (`100 / 188 / 348 / 668` boards; `15,148` allocs at N=32).
  - **Root cause**: in `managym/src/flow/damage.rs:175` (`dealt_damage_this_turn = true`
    on non-creatures) and `managym/src/flow/proposed_event.rs:1918`
    (`commit_counter_change` for `-amount` loyalty counters), mutating a damaged
    planeswalker calls `self.state.permanents[permanent_id].as_mut()` through
    `TrackedState::deref_mut()`, unconditionally bumping `revisions.characteristics`,
    `revisions.mana_pools`, `revisions.tapped`, and `revisions.triggers` for every
    attacker in a simultaneous combat damage batch (`commit_batch`). Each next
    attacker's `source_keywords`/`layer_view` then rebuilds `LayerBoard` from scratch.
- **Ruled-out stress shapes (linear/constant)**:
  - `step:damage` (N = 4 / 8 / 16): `diverse-statics` `1 / 1 / 1` boards,
    `herald-statics` `2 / 2 / 2`, `mixed` `1 / 1 / 1`, `duplicate-creatures` `6 / 6 / 6`,
    `triggers` `6 / 6 / 6`, `entry-cascade` `0` boards; `fumigate-panda`
    `step:resolve` `3 / 3 / 3` boards.
  - `clone` at `damage`: `attack-planeswalkers` `32 / 32 / 34 / 36` allocs across
    N = 4 / 8 / 16 / 32 (zero per-attacker heap allocs); `diverse-statics` /
    `duplicate-creatures` restored to `38` allocs at N=8.
  - Fixed planeswalker shapes: `way-grants` (`actions` 25 / `126` allocs,
    `characteristics` 1 board / `187` allocs, `step:activate` 0 boards / `120` allocs),
    `divided-damage` (`target` 10 actions / `71` allocs, `divide` 7 actions / `84` allocs),
    `keep-one-each-type` (`keep-choice` 15 actions / `237` allocs, `step:choice` 0 boards).

### Debug invariants & behavior check (at `09b0f89882`)

- `cargo nextest run --manifest-path managym/Cargo.toml --lib --test rules_tests --test match_state_hash_tests --test conformance_tests --test semantic_conformance_tests --test scenario_tests`: **684/684 passed** (including `LayerBoard` and `TriggerIndex` `debug_assert_eq!` cache invariants).
- `cargo nextest run --manifest-path managym/Cargo.toml -p managym-cards --test cards -E 'test(fdn::) | test(fra::) | test(observability::)'`: **332/332 passed**.
- All uninstrumented `bench_steps` and `bench_search` digests match `main` `6203fff202` and `managym/perf/baseline.json` identically.

## Remaining

- **Gate / CI verification** on PR #607:
  full workspace debug nextest, Python extension rebuild +
  contract/Academy/model/pinned-pilot suites (including #601's new
  `tests/academy/test_deck_pools.py`), and clean Ubuntu runner Engine Budget
  via PR CI (or
  `gh workflow run ci.yml --ref jack-heart/7ff13d-implement-planeswalkers-loyalty -f measure=HEAD`).

## Target model (updated by concept review)

The core planeswalker rules (`PrintedCharacteristics.loyalty`, per-turn
`loyalty_activated_this_turn`, `LoyaltyCost`, `Modification::LoyaltyAtInstantSpeed`,
`Effect::empower_jace`, `ObjectKind::Emblem` + `EmblemDefinition` in
`ZoneType::Command` flowing through `layers.rs` and `triggers.rs`) hold.
Concept review (`scratch/concept-review.html`) identified six mismatches
between Comprehensive Rules / v11 / Academy concepts and the current branch
code; Passes 1–4 are complete.

## Consumer invariants (every pass)

Offers complete under `skip_trivial`; replay, forks and search clones carry
loyalty ledgers, attack targets and emblems; viewer privacy unchanged (all new
state is public); v11 columns appended, never reordered; frozen v9/v10 inputs
untouched; `protocol/*.schema.json` patched with each wire type change;
`tests/model/test_pinned_pilot.py` recorded with `--reason` if changed.
Engine Budget within 10%; rebuild the extension and run debug nextest before
publishing.

## Ledger

Update `fdn.jsonl` (`implementation.status`) and `fra.jsonl` rows as changes
land; pending rows carry their missing authority.
