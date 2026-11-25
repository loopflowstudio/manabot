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

## Target model (updated by concept review)

The core planeswalker rules (`PrintedCharacteristics.loyalty`, per-turn
`loyalty_activated_this_turn`, `LoyaltyCost`, `Modification::LoyaltyAtInstantSpeed`,
`Effect::empower_jace`, `ObjectKind::Emblem` + `EmblemDefinition` in
`ZoneType::Command` flowing through `layers.rs` and `triggers.rs`) hold.
Concept review (`scratch/concept-review.html`) identified six mismatches
between Comprehensive Rules / v11 / Academy concepts and the current branch
code; repair them before gate in three focused passes, starting with the
slice that validates the announcement/observation contract soonest.

Behavior review at `196a17675e` confirmed every Pass 1–3 defect is still
in code: `announces_x` lists `DivideAmount` (`flow/action.rs:39`),
`continuation.rs:874` writes `frame.x = *remaining`,
`legal_activation_x_values` (`flow/play.rs:1860`) returns a range without a
target-availability filter, `FactKind::EmblemAbility` (`observation_encoder.rs:76`),
`loyalty_exemptions: Vec` (`sba.rs:166`) and `attackers_without_defender`
(`flow/combat.rs:36`). No pass has been built yet; build Pass 1 next.

### Pass 1 — Separate target division from `X` and unify `−X` target authority (build first)

- **Continuation & action separation:** Stop returning `true` from
  `Game::announces_x()` for `PendingChoice::DivideAmount` (`flow/action.rs`)
  and stop overwriting `frame.x = *remaining` in `agent/continuation.rs`.
  Preserve the parent's real announced `X` (`cast.x` or `targeting.x`) on
  `frame.x` and carry the remaining division budget in its own continuation
  column (e.g. `frame.division_remaining` / header `relation::DIVISION`,
  distinct from `relation::X`) and action representation so a policy never
  aliases dividing a total with announcing `X`.
- **Single source of truth for divided amounts in `Effect`:** Add
  `EffectValue::TargetDivision` so `Effect::DealDamage` explicitly reads each
  target's announced `slot.division` when authored with
  `EffectValue::TargetDivision`, instead of silently overriding any
  `Effect::DealDamage.amount` inside a clause whose targets have `division > 0`.
  Update Chandra, Flameshaper's `−4` (`fdn.rs`) to pass
  `EffectValue::TargetDivision` (declaring `8` once on
  `TargetRequirement.division`). Factor `SpellTargeting` (mirroring
  `ActivatedTargeting`) so `PendingChoice::ChooseTargets` and
  `DivisionParent::Spell` stop duplicating seven fields.
- **`LoyaltyCost::MinusX` offer/completion authority:** Filter
  `legal_activation_x_values` (`flow/play.rs`) by
  `activation_targets_available(..., x)` so uncompletable `X` values (e.g.
  `X = 0` for `min >= 1` with `division: X` or `total_mana_value_x`) are never
  offered; have priority (`flow/action.rs`) check
  `!self.legal_activation_x_values(...).is_empty()` for `announces_x()`
  abilities instead of probing only `x_min`. Ensure the encoder raises a typed
  capacity error rather than silently dropping `X` offers past `max_actions`.
- **Witnesses:** Directed tests proving (1) a `−X` ability with `min = 1` and
  `division: Some(EffectValue::X)` is offered at priority, omits `X = 0`, keeps
  `frame.x == announced_x` during `DivideAmount`, and resolves exact shares;
  (2) a second fixed `DealDamage(1)` in the same divided `TargetClause` still
  deals 1 rather than re-dealing divided shares.

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
