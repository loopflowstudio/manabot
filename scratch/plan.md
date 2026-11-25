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
- Pass 3 (this commit): folded `attackers_without_defender` into
  `CombatState.attack_targets: Vec<(PermanentId, Option<AttackTarget>)>` and
  updated `CanonicalCombat` (`non_player_defenders`), `combat_actions.rs`,
  `combat_preview.rs`, and `tick.rs`; removed the `Action::DeclareAttacker`
  bypass in `decision.rs::bind_frame_candidates` and bound permanent attack
  defenders in `structured_offer.rs::object_candidate_bindings` (updating
  `tests/academy/test_loyalty.py` accordingly); moved `attack_target` from
  `PendingEvent` onto `ZoneEntry` (applied directly in `zones.rs::commit_zone_move`),
  deleted `finish_created_token`, and replaced the player-hardcoded
  `tapped_and_attacking: bool` shortcut on `zones.rs::create_token` with
  explicit `create_token_attacking(key, controller, target)`; moved
  `Modification::ExileIfWouldDie` and finality-counter exile rewriting from
  `normalize_proposal()` into `collect_replacements()`; updated
  `combat_preview.rs` to report defender-specific refusals (`"<creature> can't
  attack <defender>."`) and updated `academy/ui/src/lib/board/Board.svelte` +
  `combat.ts` to default drafted attackers to the defending player with compact
  inline and board-click planeswalker retargeting (`academy/ui/node_modules` is
  absent in this worktree, so `npm run check` / `vitest` run in Pass 4).
  Verified: `managym --lib` (18 passed: `proposed_event::tests`, `decision::tests`,
  `sba::tests`), `rules_tests` (22 passed: `attack_targets`, `combat_preview`,
  `structured_attacker_offers`, `engine_events`), `match_state_hash_tests`
  (11 passed), `managym-cards --test cards` (104 passed: `gtc`, `fdn`, `kld`,
  `fin`, `fra`).

## Target model (updated by concept review)

The core planeswalker rules (`PrintedCharacteristics.loyalty`, per-turn
`loyalty_activated_this_turn`, `LoyaltyCost`, `Modification::LoyaltyAtInstantSpeed`,
`Effect::empower_jace`, `ObjectKind::Emblem` + `EmblemDefinition` in
`ZoneType::Command` flowing through `layers.rs` and `triggers.rs`) hold.
Concept review (`scratch/concept-review.html`) identified six mismatches
between Comprehensive Rules / v11 / Academy concepts and the current branch
code; Passes 1–3 repaired all six. Pass 4 (the gate) is next.

### Pass 4 — Gate & contract fixes

- Fix the pre-existing CI failures carried from `d1ef91d814`:
  - `observability::` legacy card-row parity (`observability.rs:2304` slices
    `legacy.agent_cards` with `CARD_DIM` (57) instead of `config.card_dim()` (56)).
  - `observation_layout::named_layout_matches_every_encoder_width` (update ABI
    width assertions in `observation_layout.rs` to match final post-Pass-1/2
    v11 widths).
  - `content_pack_contract_covers_env_roots_siblings_and_rollout_slots`,
    `definitions::every_authored_subtype_has_a_category`, and frozen
    semantic-conformance cases.
- Run debug nextest (`CARGO_BUILD_JOBS=2 cargo nextest run --profile ci
  --test-threads 2`), rebuild the Python extension, run Academy / Python
  tests, check Engine Budget (within 10%), and check
  `tests/model/test_pinned_pilot.py` (record with `--reason` if the attack
  row or Way/loyalty tokens moved). Keep main's MTG-420 receipts at their
  original revision.

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
