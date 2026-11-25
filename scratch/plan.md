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

## Engine measurement (at `6dd3914c73` vs reference `7d3ac1eb1c`)

Measured on Apple M4 Max (contended host; wall time is noisy, so load-invariant
`/usr/bin/time -l` instructions retired and 1 ms `/usr/bin/sample` CPU profiles
govern). Base `7d3ac1eb1c` is the committed `managym/perf/baseline.json`
reference (zero simulation diff from `main` `3be5a0cfb2`). Both sides built as
uninstrumented release binaries (`cargo build --release --locked --example
bench_search --example bench_steps`, no `benchmark` feature).

### Uninstrumented release benchmarks

| Benchmark | Metric | Base `7d3ac1eb1c` | Candidate `6dd3914c73` | Delta |
|---|---|---:|---:|---:|
| `bench_steps` (512 games, 118,559 steps, 0 capped) | Digest | `10601622128491155944` | `10601622128491155944` | unchanged |
| `bench_steps` (C/R/R/C) | Instructions retired (median) | `46.96B` (`46.99B`, `46.93B`) | `50.92B` (`51.03B`, `50.82B`) | **+8.43%** (`+3.96B`) |
| `bench_steps` (C/R/R/C) | User CPU / wall (noisy) | `7.51s`/`6.90s` (`47.94s`/`21.66s`) | `7.70s`/`7.44s` (`56.03s`/`16.48s`) | +5.1% user |
| `bench_search` (seeds 1–3, 40 dec., 0 caps) | Digests | `13638875039089371439` / `5466143068363283977` / `9998925852575961750` | identical | unchanged |
| `bench_search` | Instructions retired | `1,034.82B` | `1,206.99B` | **+16.64%** (`+172.16B`) |
| `bench_search` | Cycles elapsed | `330.25B` | `367.17B` | **+11.18%** |
| `bench_search` | User CPU / wall (noisy) | `150.71s` (`784.14s`) | `156.03s` (`373.31s`) | +3.5% user |

### Stress states (`bench_states --features benchmark`)

- **Existing large-board shapes (size 8, candidate vs base)**:
  - Single-call `characteristics` and `observe` keep identical allocation counts
    (`+0.0%` allocs, `+0.8%–1.9%` bytes).
  - `step:damage` suffers massive extra `LayerBoard` rebuilds:
    `diverse-statics` `boards 1 → 4` (`647 → 1,442` allocs, `+122.9%`, `+118.3%` bytes);
    `herald-statics` `boards 2 → 5` (`801 → 1,608` allocs, `+100.7%`);
    `mixed` `boards 1 → 4` (`1,250 → 2,020` allocs, `+61.6%`);
    `duplicate-creatures` and `triggers` `boards 6 → 10`.
  - `clone` at `damage` grows `+21%–23%` in heap allocations (`38 → 46` on
    `diverse-statics`/`duplicate-creatures`, `39 → 48` on `herald-statics`,
    `40 → 49` on `mixed`) due to per-attacker `Vec<AttackTarget>` in `TurnState`.
  - `triggers` keeps identical allocation counts (`+0.0%`) with `+3.9%–13.2%`
    bytes (`+56.0%` on tiny `duplicate-creatures`) from wider `EventKinds` /
    `TriggerSourceEntry` / `TriggerSourceSnapshot`.
- **Planeswalker stress shapes (candidate)**:
  - `way-grants`: `actions` `0.048ms` (25 actions, 126 allocs), `characteristics`
    `0.028ms` (187 allocs, 164 KB), `search` `32.73ms` (100 sims, 1,023 boards, 204 MB).
  - `divided-damage`: `target` `observe` 10 actions (`0.013ms`), `divide` `observe`
    7 actions (`0.013ms`), `step:choice` `0.050ms` (16 boards).
  - `keep-one-each-type`: `step:activate` `0.470ms` (17 boards), `keep-choice`
    `observe` 15 actions (`0.105ms`), `step:choice` `0.028ms` (0 boards).
  - Added scaling **`attack-planeswalkers`** (`managym/examples/bench_states.rs`,
    N Gray Ogres vs 6 distinct opposing planeswalkers, sizes 4/8/16): `attack`
    `observe` stays 8 actions (`0.012 / 0.013 / 0.016ms`); `damage clone` grows
    linearly `+1` alloc/attacker (`36 / 40 / 50` vs `25` at priority); `damage`
    `step:damage` rebuilds `6 / 10 / 17` boards.

## Remaining

Four profile-backed performance slices before gate/CI. Every slice preserves
complete choices, observation capacities, native skipping, exact baseline
digests and debug cache assertions.

### Slice P1 — Avoid empty-removal `TrackedState` invalidation in `remove_invalid_defenders`
- **Profile evidence**: `perform_state_based_actions → legend_groups → layer_board`
  grew `0.93B → 122.91B` (`+121.99B`, `+11.8%` of baseline `bench_search`) and
  `0.01B → 2.62B` (`+2.61B`, `+5.6%` of baseline `bench_steps`); `bench_states`
  `step:damage` boards jumped `1 → 4` (`diverse-statics`/`mixed`), `2 → 5`
  (`herald-statics`), `6 → 10` (`duplicate-creatures`/`triggers`).
- **Hypothesis**: `remove_invalid_defenders()` (`managym/src/flow/combat_actions.rs:218`)
  unconditionally calls `self.state.combat.as_mut()` on every combat SBA pass
  even when `removed` is empty, triggering `TrackedState::deref_mut()` and
  invalidating `LayerBoard`, `TriggerIndex`, and `PaymentCache` right before
  `legend_groups()` reads `layer_views()`. Returning early whenever no permanent
  defender is invalid eliminates these spurious cache invalidations and their
  cascading rebuilds across `Context::evaluate`, `indexed_sources`, and
  `payment_model`.
- **Deletes**: Unconditional `self.state.combat.as_mut()` and eager `Vec`
  collection in `remove_invalid_defenders()` when no permanent defender is invalid.
- **Focused check**: `cargo nextest run -E 'binary(rules_tests) & test(attack_targets)'`,
  `sba::tests`, `bench_states diverse-statics herald-statics triggers duplicate-creatures attack-planeswalkers --sizes 8 --layers step:damage --allocations`
  (`boards` restored to `1, 2, 6, 6`), and `/usr/bin/time -l` release
  `bench_steps` + `bench_search` with unchanged baseline digests.

### Slice P2 — Resolve static source once per `Node` in `Context::targets`
- **Profile evidence**: `Context::instructions → Context::targets::from_iter`
  grew `10.61B → 27.09B` (`+16.48B`, `+155%`), and `Context::targets::FnMut`
  self instructions grew 5.5x (`3.12B → 17.03B` in `bench_search`; `0.03B → 0.34B`
  in `bench_steps`).
- **Hypothesis**: Moving `Node.source` from `PermanentId` to `ObjectId` caused
  `Context::scope` and `Context::matches` (`managym/src/flow/layers.rs`) to call
  `self.game.state.permanent_of(source)` up to four times per `(node, target)`
  pair inside the battlefield filter loop. Resolving `(source_permanent,
  source_controller, source_attached, source_name)` once per `Node` before
  scanning `self.objects` removes redundant per-target lookups while preserving
  exact battlefield and command-zone emblem scoping.
- **Deletes**: Per-target `permanent_of(source)`, `source_controller(source)`,
  and `attached_permanent` lookups inside `Context::scope` / `Context::matches`
  for `Context::targets`.
- **Focused check**: Debug nextest for `layers` and emblem tests (keeping
  `debug_assert_eq!` cache invariants), `bench_states diverse-statics herald-statics way-grants --layers characteristics`,
  and `/usr/bin/time -l` release `bench_steps` + `bench_search` with unchanged
  baseline digests.

### Slice P3 — Shrink `GameEvent` by boxing `permanent_defenders` payloads
- **Profile evidence**: `GameEvent::clone` grew 4.7x (`2.65B → 12.47B`, `+9.82B`
  in `bench_search`: `commit_proposed_event → emit` `+7.20B`,
  `perform_automatic_tap → emit` `+3.09B`), independent of `LayerBoard` invalidation.
- **Hypothesis**: Inline `permanent_defenders: Vec<...>` fields (+24 bytes each)
  on `AttackersDeclared` and `CombatAttackersDeclared` (`managym/src/flow/event.rs`)
  inflated `size_of::<GameEvent>()` for every event emitted, cloned twice in
  `Game::emit`, and stored across event logs. Boxing those rare payloads (e.g.
  `Option<Box<[...]>>` with identical serde omission when empty) restores
  `size_of::<GameEvent>()` to baseline.
- **Deletes**: Inline 24-byte `Vec` fields on `GameEvent::AttackersDeclared` and
  `GameEvent::CombatAttackersDeclared`.
- **Focused check**: `match_state_hash_tests`, `attack_targets` / combat-event
  rules tests, protocol/Academy serialization tests, and `/usr/bin/time -l`
  release `bench_steps` + `bench_search` with unchanged baseline digests.

### Slice P4 — Flatten `TurnState::attacks_this_turn` to remove per-attacker clone allocations
- **Profile evidence**: `bench_states` `clone` at `damage` grew `+21%–23%` in
  allocations (`38 → 46` at size 8; `36 / 40 / 50` at sizes 4 / 8 / 16 on
  `attack-planeswalkers` vs `25` at priority).
- **Hypothesis**: Changing `TurnState::attacks_this_turn` (`managym/src/flow/turn.rs`)
  to `Vec<(ObjectId, Vec<AttackTarget>)>` allocates one heap `Vec` per attacking
  creature and clones each on every `Game::clone()` in search. Storing flat
  `attacks_this_turn: Vec<(ObjectId, u32)>` plus sparse
  `attacked_permanents: Vec<(ObjectId, u32, AttackTarget)>` (empty for player
  attacks, matching `CanonicalTurn` in `managym/src/state/hash.rs`) eliminates
  all per-attacker clone allocations.
- **Deletes**: Nested per-attacker `Vec<AttackTarget>` in `TurnState::attacks_this_turn`
  and the flattening projection in `CanonicalTurn::from`.
- **Focused check**: `match_state_hash_tests`, `attack_targets` rules tests,
  `bench_states diverse-statics duplicate-creatures attack-planeswalkers --sizes 8 --layers clone --allocations`
  (`damage clone` allocs restored to `38`), and `/usr/bin/time -l` release
  `bench_steps` + `bench_search` with unchanged baseline digests.

### Post-slice re-measurement and deferred CI check
- After P1–P4, re-run uninstrumented release `bench_steps` and `bench_search`
  with `/usr/bin/time -l` and `/usr/bin/sample` against `7d3ac1eb1c` to confirm
  whether `Characteristics::clone_from_ref_in` (`Ability::clone`),
  `indexed_sources`, and `payment_model` recover once P1 stops combat SBA cache
  invalidation, or whether any residual hotspot warrants a follow-up slice.
- Deferred to clean GitHub-hosted Ubuntu CI runner (no wall-time speed claim
  from this contended macOS host): full workspace debug nextest, Python 3.12
  matrix, and same-runner Engine Budget via PR CI (or
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
