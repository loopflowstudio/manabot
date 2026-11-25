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

- Pass 4 (gate & contract fixes): fixed legacy `CARD_DIM` slicing in
  `observability.rs:2304`, emblem `registry_name()` in `env.rs:2174` and
  `python/bindings.rs::catalog_tokens`, v11 ABI width assertions in
  `observation_layout.rs` (permanents 60, actions 32, relations 52),
  `test_input_scales.py` (`player_turn_state` reading width 24),
  `tla::farm_animals_can_decline_target_and_still_pay_sacrifice` (added
  real Food target to decline plus no-target activation case), and
  `test_loyalty.py` (`len(bindings) == len(ask.declarable)`). Recorded
  `tests/public_api.json` (`Action.division_amount`,
  `ContinuationFrame.division_remaining`), refreshed Academy opening
  fixtures, and wrote `-planeswalkers` semantic-conformance,
  authored-match authority (174 commands), and parity (175 checkpoints,
  1,050 viewer checks, 0 divergences) receipts while keeping main's
  MTG-420 `-battlefield-history` receipts untouched. Verified:
  `academy/ui` (`svelte-check` 0 errors/warnings, `vitest` 40/40 files,
  653 tests), Rust debug workspace nextest (2,184 tests green), cp313
  extension rebuild & atomic install, `tests/model/test_pinned_pilot.py`
  green with zero drift (v9 & v11 unchanged), Academy/model/world/contract
  Python suites green (105 + 5 + 286 + 177 passed), and release
  `bench_steps` digest match (`10601622128491155944`, 118,559 steps,
  512 games). Same-runner Engine Budget comparison deferred to gate/CI.

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
