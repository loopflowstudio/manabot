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

- Pass 1 (separation of target division from `X` and `−X` target authority):
  removed `PendingChoice::DivideAmount` from `Game::announces_x()`, added
  `ContinuationFrame.division_remaining` (encoded on header `relation::DIVISION`)
  while preserving `frame.x`, added `ActionOption.division_amount` /
  `action::DIVISION` (v11 column 31), added `EffectValue::TargetDivision` and
  switched Chandra, Flameshaper's `−4` to it so `DealDamage` only divides when
  authored with `TargetDivision`, factored `SpellTargeting` (mirroring
  `ActivatedTargeting`) to eliminate the 7-field copy and `TargetSelection`,
  and filtered `legal_activation_x_values` by `activation_targets_available`
  (using `min_for_x`/`max_for_x`) with priority checking non-empty legal X
  values. Verified with directed witness tests (including `ObservationEncodeError::Capacity`
  when X offers exceed `max_actions`), Chandra `−4`, encoder, continuation, and
  observability field-coverage tests.

## Target model (updated by concept review)

The core planeswalker rules (`PrintedCharacteristics.loyalty`, per-turn
`loyalty_activated_this_turn`, `LoyaltyCost`, `Modification::LoyaltyAtInstantSpeed`,
`Effect::empower_jace`, `ObjectKind::Emblem` + `EmblemDefinition` in
`ZoneType::Command` flowing through `layers.rs` and `triggers.rs`) hold.
Concept review (`scratch/concept-review.html`) identified six mismatches
between Comprehensive Rules / v11 / Academy concepts and the current branch
code; repair them before gate in three focused passes. Pass 1 is done; build
Pass 2 next.

### Pass 2 — Remove derived `FactKind::EmblemAbility` and move zero-loyalty SBA exemption into `layers.rs`

- **Emblem v11 observation:** Delete `FactKind::EmblemAbility` and the
  `"Emblem::Static"` / `"Emblem::Triggered"` record-vocabulary tokens from
  `observation_encoder.rs`, `observation_layout.rs`, `record_vocabulary.rs`,
  and `observability.rs`. Emblems are already observed via their
  `agent_cards` / `opponent_cards` row (`card::ZONE_COMMAND = 1`) joined via
  `semantic_cards` to `catalog_tokens`, which carries their full static and
  triggered token streams. (Choice made for review in
  `scratch/concept-review.html` Q1.)
- **Zero-loyalty SBA exemption via `layers.rs`:** Project
  `StaticAbility::StateBasedActionDoesntApply { action: ZeroLoyalty, scope }`
  onto `PermanentView` in `layers.rs` (`Layer::Ability`, covering both
  battlefield permanents and command-zone emblems). Restore
  `sba.rs::perform_state_based_batch` to a zero-allocation streaming loop over
  `self.layer_views(...)`, deleting its per-SBA `views: Vec` and
  `loyalty_exemptions: Vec` allocations.

### Pass 3 — Unify combat-defender state, candidate binding, `ZoneEntry.attack_target`, death-to-exile replacement, and Academy drafting

- **Single `CombatState` defender list:** Replace
  `CombatState.attack_targets: Vec<(PermanentId, AttackTarget)>` +
  `attackers_without_defender: Vec<PermanentId>` with one field:
  `attack_targets: Vec<(PermanentId, Option<AttackTarget>)>` (`None` = CR
  506.4c defender left combat). Update `CanonicalCombat`, `combat_actions.rs`,
  `combat_preview.rs`, and `tick.rs`.
- **Single candidate-binding owner:** Remove the raw-`ActionSpace`
  `Action::DeclareAttacker` bypass in `decision.rs::bind_frame_candidates` so
  object candidate bindings come solely from `structured_offer.rs`.
- **Entry metadata on `ZoneEntry`:** Move `attack_target: Option<AttackTarget>`
  from `PendingEvent` (`proposed_event.rs`) onto `ZoneEntry`, and remove the
  player-hardcoded `tapped_and_attacking` shortcut on `zones.rs::create_token`.
- **Single death-to-exile replacement pipeline:** Move
  `Modification::ExileIfWouldDie` and finality-counter exile rewriting from
  `normalize_proposal()` into `collect_replacements()` alongside
  `ReplacementEffect::ExileIfWouldDie` so CR 614.1a / 616.1 death replacements
  share one ordering authority (single candidates still auto-apply).
- **Academy combat drafting & Tomik refusals:** In `combat_preview.rs`, report
  defender-specific refusals (`"<creature> can't attack <defender>."`) instead
  of `"<creature> can't attack now."`. In `academy/ui/src/lib/board/Board.svelte`,
  default newly drafted attackers to the defending player (when legal) with
  compact inline retarget controls for planeswalkers, instead of blocking Space
  behind mandatory blank `<select>` dropdowns. (Choice made for review in
  `scratch/concept-review.html` Q3.)

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
