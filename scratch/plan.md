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

- Performance slices P1–P4 (`49393aef55`..`6a08a28b0e`):
  - **P1** (`49393aef55`): `remove_invalid_defenders()` returns early before
    `self.state.combat.as_mut()` when no permanent defender is invalid,
    eliminating spurious `TrackedState` cache invalidations in combat SBA.
  - **P2** (`a9986c3aab`): `Context::targets()` resolves source permanent,
    controller, and name once per `Node`, fast-paths `This`/`Attached`/
    `SpellsYouControl`, and iterates `self.objects.iter()` directly (deleting
    per-target `Context::scope`).
  - **P3** (`ef7f6b59ac`): boxed `permanent_defenders` as
    `Option<Box<Vec<...>>>` on `GameEvent::AttackersDeclared` and
    `GameEvent::CombatAttackersDeclared` with a compile-time `<= 56` byte
    guard, restoring baseline `size_of::<GameEvent>()` with identical serde
    omission/encoding.
  - **P4** (`6a08a28b0e`): flattened `TurnState::attacks_this_turn` to
    `Vec<(ObjectId, u32)>` plus sparse `attacked_permanents` (borrowed
    directly in `CanonicalTurn`), eliminating per-attacker clone allocations.
  - Focused debug nextest: 149/149 passed (`attack_targets`,
    `presentation_events`, `combat_preview`, `cr_613`, `match_state_hash_tests`,
    `sba::tests`, `layers`, `observability`, `emblem`).

## Engine measurement (at `6a08a28b0e` vs `6dd3914c73` and base `7d3ac1eb1c`)

Measured on Apple M4 Max (load-invariant `/usr/bin/time -l` instructions
retired govern). Base `7d3ac1eb1c` is the committed `managym/perf/baseline.json`
reference (zero simulation diff from `main` `3be5a0cfb2`). Uninstrumented
release binaries (`cargo build --release --locked --example bench_search --example bench_steps`, no `benchmark` feature).

### Uninstrumented release benchmarks

| Benchmark | Metric | Base `7d3ac1eb1c` | Pre-P1–P4 `6dd3914c73` | Post-P1–P4 `6a08a28b0e` | vs Base |
|---|---|---:|---:|---:|---:|
| `bench_steps` (512 games, 118,559 steps, 0 capped) | Digest | `10601622128491155944` | `10601622128491155944` | `10601622128491155944` | unchanged |
| `bench_steps` | Instructions retired (median of 2) | `46.96B` (`46.99B`, `46.93B`) | `50.92B` (`51.03B`, `50.82B`) | `46.54B` (`46.55B`, `46.53B`) | **-0.89%** (`-0.42B`) |
| `bench_steps` | User CPU / wall | `7.51s`/`6.90s` | `7.70s`/`7.44s` | `2.93s`/`2.94s` (`40.3k steps/s`) | faster |
| `bench_search` (seeds 1–3, 40 dec., 0 caps) | Digests | `13638875039089371439` / `5466143068363283977` / `9998925852575961750` | identical | identical | unchanged |
| `bench_search` | Instructions retired | `1,034.82B` | `1,206.99B` (`+16.64%`) | `1,013.43B` | **-2.07%** (`-21.39B`) |
| `bench_search` | Cycles elapsed | `330.25B` | `367.17B` (`+11.18%`) | `233.53B` | **-29.29%** |
| `bench_search` | User CPU / wall | `150.71s` | `156.03s` | `58.62s`/`59.35s` (`worst 1.454s`) | faster |

### Stress states (`bench_states --features benchmark`, post-P1–P4 `6a08a28b0e`)

- **`step:damage` (size 8)**: `LayerBoard` rebuilds restored to exact baseline
  counts — `diverse-statics` `4 → 1` (`1,442 → 695` allocs), `herald-statics`
  `5 → 2` (`1,608 → 857` allocs), `duplicate-creatures` `10 → 6` (`516` allocs),
  `triggers` `10 → 6` (`1,119` allocs); `attack-planeswalkers` `10 → 7` (`430` allocs).
- **`clone` at `damage` (size 8)**: restored to exact baseline `38` allocs on
  `diverse-statics` (`46 → 38`) and `duplicate-creatures` (`46 → 38`);
  `attack-planeswalkers` across sizes 4/8/16 dropped from `36 / 40 / 50` to
  `32 / 32 / 34` (matching `block` `32 / 32 / 34`, zero per-attacker heap
  allocations).
- **`characteristics` (size 8)**: `diverse-statics` 1 board (309 allocs),
  `herald-statics` 1 board (311 allocs), `way-grants` 1 board (187 allocs).

## Remaining

All implementation and performance slices (Slices 1–8, Concept Review Passes 1–4,
Coverage Decomposition, and Performance Slices P1–P4) are complete.

- **Gate / CI verification**: full workspace debug nextest, Python extension
  rebuild + contract/Academy/model/pinned-pilot suites, and clean Ubuntu CI
  runner Engine Budget via PR CI (or
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
