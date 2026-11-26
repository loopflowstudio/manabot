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
- Performance slices P5–P6 (`7955be5849`, compressed in `ba30625410`/`5a3416e943`):
  untracked `TrackedState::mark_dealt_damage` / `record_life_change`,
  fine-grained `revisions.counters` with `LayerBoard.reads` (`StateReads`),
  and cached `NodeSource` (`permanent: Option<PermanentId>`) in `flow/layers.rs`.
- Synced main through #610 (`f5684725db`, merge `fcc96be003`/`0c3821e06d`):
  Academy-side, plus one `experience.rs` field and matching protocol schema
  additions. No Python/Academy/schema checks run after the merge; they fall to gate.

## Engine performance review — Pass 2 (post-compress `5a3416e943` vs `main` `fcc96be003`)

Measured on Apple M4 Max (contended host; load-invariant `/usr/bin/time -l`
instructions retired, cycles elapsed, and 1 ms `/usr/bin/sample` CPU profiles
govern). Both sides built as uninstrumented release binaries (`--release --locked`,
no `benchmark` feature) on the same machine after the #610 merge and P5–P6
compress (`ba30625410`).

### Supported gains (uninstrumented release, identical digests)

- `bench_steps` (512 games, 118,559 steps, 0 capped, digest `10601622128491155944`):
  ABBA median `46.763B` (`46.814B`, `46.713B`) → `46.320B` (`46.322B`, `46.319B`)
  instructions retired (**−0.95%**, `−0.44B`).
- `bench_steps --layers` (16 games, 1,308 steps, digest `1064912348938184597`):
  `0.6487B` → `0.6403B` instructions (**−1.28%**, **−1.33%** cycles).
- `bench_search --layers` (seed 244, 8 dec., 1,344 sims, digest `17627308883250227726`,
  BLAKE3 `0eaa372c...`): `17.263B` → `16.333B` instructions (**−5.39%**,
  **−4.46%** cycles).
- `bench_search` (seeds 1–3, 40 dec. each, 0 caps, `7,680 / 8,064 / 7,424` sims,
  digests `13638875039089371439` / `5466143068363283977` / `9998925852575961750`
  and BLAKE3 `2f29dd10...` / `09de41e9...` / `9692421a...` unchanged):
  `1,028.10B` → `999.74B` instructions (**−2.76%**, `−28.36B`),
  `245.61B` → `238.50B` cycles (**−2.90%**), user `96.28s → 91.12s`.

### CPU profile & stress-state scaling

- **1 ms `/usr/bin/sample` on `bench_search`** (`10,781` base vs `10,702` cand samples):
  P6 eliminated `Context::exists` (`0.36%` inclusive pre-P6 → `0.00%`);
  `Context::evaluate` fell `46.91% → 42.48%` (`−4.43pp` inclusive), `layer_board`
  `42.79% → 40.87%` (`−1.92pp`), `Context::targets` `3.39% → 2.29%` (`−1.11pp`),
  `Context::apply` `10.00% → 9.19%` (`−0.81pp`), `drain_pending_events`
  `36.31% → 34.98%` (`−1.33pp`). `bench_steps` (`1,307` vs `1,668` samples) shows
  no branch-introduced regression.
- **`bench_states --features benchmark --allocations`**:
  - `attack-planeswalkers` (`N = 4, 8, 16, 32`): `damage | step:damage` boards
    `3 / 7 / 15 / 31` → `1 / 1 / 1 / 1` (`339 / 389 / 323 / 480` allocs,
    `0.047 / 0.053 / 0.056 / 0.128 ms`); `block | search` at `N = 32` dropped
    `4,620` → `714` boards (`5.41`/sim across 132 sims, linear like
    `duplicate-creatures` `668` boards / `132` sims = `5.06`/sim; 16→32 bytes
    `2.81x` vs `duplicate-creatures` `3.09x`).
  - `way-grants`, `divided-damage`, `keep-one-each-type`: 1 board per
    `characteristics`, 0 boards per `step:activate` on `way-grants`, 23–28 clone
    allocs. Existing guards (`duplicate-creatures`, `diverse-statics`,
    `herald-statics`, `triggers` at `N = 8, 16`) match baseline `step:damage`
    board counts (`6 / 1 / 2 / 6`) with linear N=8→16 growth.

### Invariants, negative results & ruled-out bottlenecks

- **Debug invariants verified (`debug_assertions` ON)**: audited `StateReads` /
  `ReadSet` / `TrackedState` (`mark_dealt_damage`, `record_life_change`,
  `change_counters`, `ActivatedAbilitiesCostLess` → `dynamic_activations`);
  passed `managym --lib` (413), `managym-state` (18), `managym-cards --test cards`
  (1,414), `rules_tests` (214, 2 ignored), `match_state_hash_tests` (11),
  `conformance_tests` (2), `semantic_conformance_tests` (7), and `scenario_tests` (20).
- **Ruled out / no open branch bottleneck**: P5 and P6 closed the two bottlenecks
  flagged in Pass 1 (`attack-planeswalkers` `O(N)` damage rebuilds and `Context::exists`
  `permanent_of` lookups). The remaining top `bench_search` shares (`Context::evaluate`
  42.5%, `priority_action_space` 24.2%, `_platform_memmove` 7.3%,
  `perform_state_based_batch` 7.1%) are shared baseline engine costs, all lower or
  flat in absolute instructions vs `main`; `finish_state_based_batch`'s `+0.42pp`
  relative sample share (`1.52% → 1.94%`) is sampling variance/shrunk-total shift
  (the only branch change there is a single enum arm `ObjectKind::Emblem => false`).
  No further MTG-430 optimization pass is warranted; advance to gate/CI.

## Remaining (Architectural Redo — 2026-11-25)

Requested after comparing against `phase-rs/phase` before shipping PR #607.
Keep all shipped planeswalker cards, catalog-keyed `EmblemDefinition`, composed
`Effect::empower_jace`, sequential combat/division offers, append-only v11
layouts, pinned pilot zero-drift, and Engine Budget intact. Do not adopt inline
ad-hoc emblems, dedicated per-keyword `Effect` state machines, or compound
batch `DeclareAttackers`.

- Slice R1 (`58e3e40618`): centralized `managym/src/flow/functioning_abilities.rs`
  gate (CR 702.26b phasing, CR 114.4 command-zone emblem admission, CR 113.6
  zone-of-function classifiers) routing `layers.rs`, `triggers.rs`,
  `proposed_event.rs`, `statics.rs` (deleted obsolete `static_sources()`),
  `zones::emblems()`, `damage.rs`, `tick.rs`, `sba.rs`, `mana.rs`, `play.rs`,
  `resolution.rs`, and `opening.rs` (also fixed missing phasing guards on
  damage replacements and `static_sources`, and main-merge `create_token`
  arity in `tla.rs`); added `AbilityProvenance` (`Characteristic` vs Layer-6
  `Granted`) + `LinkedAbilitySource { creator, characteristic_set }` latched
  onto `ExileLink` and emblem `ObjectForm` so Layer-6 grants fail closed on
  linked exile/emblem pairing (CR 607.1a/d, 607.5). Verified: `managym --lib`
  419/419 (including 5 new directed Slice R1 tests, `record_vocabulary`,
  `token_renames`, `sba::tests`), `managym-state` 18/18, `match_state_hash_tests`
  11/11, `rules_tests` 214/214, and `managym-cards` (`tla::koh`,
  `tla::boiling_rock`, `fdn::`, `fra::`) 248/248 green; Python extension
  rebuild + pinned pilot check deferred to Slice R2.

- Slice R2 (`735620f5a6`): added condition-independent `StaticSourceIndex`
  (plus `Layer` active bitmask and `pt_counters` pre-collection) in
  `managym/src/flow/layers.rs`, `statics.rs`, and `functioning_abilities.rs` so
  layer rebuilds, Layer-6 `discover()`, `compute_board`, and
  `for_each_functioning_static`/`has_functioning_static` scale with
  `O(|generators|)` instead of `O(|battlefield|)` while over-including
  conservatively and validating completeness under `debug_assertions`; added
  comment- and `#[cfg(test)]`-aware `managym/src/source_census.rs` architectural
  guard tests pinning single-authority counts across `managym/src/` (and fixed a
  main-merge `create_token` arity in `agent/observation.rs`). Verified:
  `managym --lib` 428/428 (including `static_source_index_scales_with_generators_and_is_condition_independent`
  and 7 `source_census` tests), `match_state_hash_tests` 11/11, `rules_tests`
  214/214 (2 ignored), `managym-cards` (`fdn::`, `fra::`) 243/243, `cargo fmt --check`,
  release `bench_steps` digest `10601622128491155944` (`46.157B` instructions,
  −0.35% vs Pass 2), `bench_steps --layers` digest `1064912348938184597`,
  `bench_search --layers` digest `17627308883250227726` / BLAKE3 `0eaa372c...`
  (`16.336B` instructions), `bench_states --features benchmark --allocations`
  linear scaling (`duplicate-tokens` 10/13/19 allocs, `attack-planeswalkers`
  10/11/14 allocs at N=8/16/32), cp313 `maturin develop` rebuild, and
  `tests/model/test_pinned_pilot.py` (2 passed, 1 skipped, zero drift).
  Refreshed `scratch/tour.yaml`; ready to advance to `card-review` → `card-perf`
  → `compress` → `pr-publish` → `ship-review`.

## Prior PR publish verification (baseline before redo)

- **PR publish verification** on PR #607 (`a9b5039ab8`):
  removed redundant unread v11 `action::ATTACK_TARGET` column (`action_focus[1]`
  already carries the planeswalker defender, matching `DeclareBlocker`;
  `actions` width 31 with `DIVISION` at column 30), updated
  `test_bucketed_shapes.py` to assert `attacked_permanent` remapping alongside
  `attached_to`/`blocking`, updated `test_native_observation_contract.py` v11
  action width to 31, and updated `test_catalog_naming.py` to pick an in-leaf
  peer for `"Gray Ogre"` instead of hardcoded `"Glory Seeker"`. Verified
  `public_api.py check`, `experience_protocol_tests`, Academy UI (`svelte-check`
  + 660 vitest + `play-session.spec.ts` mana-float e2e), full Rust debug
  nextest + benchmark tests, pinned pilot, and Python model/contract/Academy
  suites. Clean Ubuntu `Engine Budget` passed on PR #607 CI.

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
