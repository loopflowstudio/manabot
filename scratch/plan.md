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

- Coverage decomposition slices 1–3 landed (`9b19a44635`, `92aac08cde`,
  `7b5536b37c`):
  - Slice 1 (`9b19a44635`): replaced `DelayedEvent::Attacks` /
    `DelayedTriggerEvent::Attacks` with `Whenever(Box<TriggerCondition>)`
    matched by the shared `match_trigger_wrappers` / `collect_occurrences` /
    `event_trigger_context` pipeline; shared `TriggerMember::AttackBatch`
    binds `ObjectSelector::Remembered` when `reads_remembered()` holds; folded
    `AttackedSubject` into two axes (`attacked: Actor`, `attacked_kind:
    AttackedKind`, defaults skipped so existing `Attacks` tokens stay
    unchanged); cut over Garruk, Curse Breaker −4 and Jace, Reality Sculptor
    −3.
  - Slice 2 (`92aac08cde`): parameterized `ObjectSelector::LibraryTop(usize)`,
    propagated `library_top_departure` through `Effect::Move`, cut Chandra,
    Flameshaper +2 over to `exile(LibraryTop(3))` + `SelectCards { from:
    Moved, min: 1, max: 1, effects: [GrantPlay { Selected, ThisTurn }] }`
    (giving `GrantPlay` its first card consumer), and deleted
    `ExileTopCardsToPlay.choose` and `Effect::exile_top_choose_to_play`.
  - Slice 3 (`7b5536b37c`): added `StateBasedAction::LegendRule` and
    `StateBasedActions` bitset on `Characteristics.exempt_sbas`; cut Sakashima
    of a Thousand Faces (and its `CopyException::GrantStatic`) over to
    `StateBasedActionDoesntApply { LegendRule, YouControl(any) }`; deleted
    `StaticAbility::LegendRuleDoesntApply`, `exempt_from_zero_loyalty`, and
    the `names.clear()` scan in `legend_groups`; retired
    `StaticAbility::LegendRuleDoesntApply` in `cards/token_renames.rs`.
  - Verified: focused debug nextest (`flow::sba::tests`, `token_renames`,
    `ability_description`, `record_vocabulary`, `cr_603_triggers`,
    `attack_targets`, `garruk_curse_breaker`, `jace_reality_sculptor`,
    `sanctum_lurker`, `chandra_flameshaper`, `kld`, `cmr`, `hob`,
    `observability`), `cargo fmt --all`, cp313 extension rebuild,
    `tests/model/test_pinned_pilot.py` (zero drift, v9 & v11),
    `tests/academy/test_loyalty.py` (5/5), Academy opening fixtures refreshed
    and `tests/academy/test_choice_navigation.py` (8/8), planeswalker
    conformance & authored-match parity receipts refreshed
    (`semantic_conformance_tests` 20/20, `RUL5_PARITY_OK` 174 commands / 1050
    viewers).

## Remaining

No slices remain. Deferred to gate/CI: full workspace debug nextest,
same-runner Engine Budget comparison, and the full CI matrix (incl. Python
3.12).

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
