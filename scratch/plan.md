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
- Performance slices P5–P6 (`7955be5849`): untracked
  `TrackedState::mark_dealt_damage` / `record_life_change`, fine-grained
  `revisions.counters` with `LayerBoard::reads_counters`, and cached
  `Node::source_permanent` in `flow/layers.rs`. `attack-planeswalkers`
  `step:damage` boards `3/7/15/31` → `1/1/1/1`; `block | search` at N=32
  `4,620` → `714` boards (5.4/sim, linear like `duplicate-creatures`).
  Release vs `main` `6203fff202`, identical digests: `bench_steps` −0.65%,
  `bench_search` −2.59%, `bench_search --layers` −5.40% instructions.
  Focused debug nextest 88/88 (`managym`) and 266/266 (`managym-cards`).
- Synced main through #610 (`f5684725db`, merge `fcc96be003`/`0c3821e06d`):
  Academy-side, plus one `experience.rs` field and matching protocol schema
  additions. No checks run after the merge; they fall to gate.

## Remaining

- **Gate / CI verification** on PR #607:
  full workspace debug nextest, Python extension rebuild +
  contract/Academy/model/pinned-pilot suites (including #601's
  `tests/academy/test_deck_pools.py` and #610's Academy changes), protocol
  schema byte-equality after the #610 merge, and clean Ubuntu runner Engine Budget
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
