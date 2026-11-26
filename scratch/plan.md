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

## Engine performance review — Pass 2 (`5a3416e943` vs main `fcc96be003`)

M4 Max, uninstrumented release, instructions retired govern; all digests
unchanged. `bench_steps` −0.95% (46.763B → 46.320B), `--layers` −1.28%;
`bench_search` seeds 1–3 −2.76% (1,028.10B → 999.74B), `--layers` −5.39%.
`bench_states` stress cases (`attack-planeswalkers`, `way-grants`,
`divided-damage`, `keep-one-each-type`) scale linearly; existing guards match
baseline. Remaining `bench_search` hot spots (`Context::evaluate`,
`priority_action_space`, memmove, SBA batch) are shared baseline costs. No
further MTG-430 optimization pass warranted.

## Remaining (Architectural Redo — 2026-11-25)

Requested after comparing against `phase-rs/phase` before shipping PR #607.
Keep all shipped planeswalker cards, catalog-keyed `EmblemDefinition`, composed
`Effect::empower_jace`, sequential combat/division offers, append-only v11
layouts, pinned pilot zero-drift, and Engine Budget intact. Do not adopt inline
ad-hoc emblems, dedicated per-keyword `Effect` state machines, or compound
batch `DeclareAttackers`.

- Slice R1 (`58e3e40618`): one functioning-abilities gate
  (`flow/functioning_abilities.rs`: CR 113.6 zones, 114.4 emblems, 702.26b
  phasing) routes layers, triggers, replacements, statics, damage, SBA, mana,
  play, resolution and opening (fixed two missing phasing guards); added
  `AbilityProvenance` + `LinkedAbilitySource { creator, characteristic_set }`
  on `ExileLink` and emblem `ObjectForm`. Debug `managym --lib`, rules, hash
  and fdn/fra/tla card tests green.
- Slice R2 (`735620f5a6`): `StaticSourceIndex` (`flow/layers.rs`) — a
  per-`Context::evaluate` filter that still scans every layer object; only the
  gathers after it (Layer-6 `discover()`, `compute_board`,
  `for_each_functioning_static`) scale with generators. Added
  `source_census.rs` text-count tests. Release digests unchanged
  (`bench_steps` `10601622128491155944`, `46.157B` instr), pinned pilot zero
  drift, cp313 rebuilt.
- Concept review of R1–R2 (`09df882062`, `scratch/concept-review.html`):
  debug workspace nextest 2,205 passed / 3 skipped at `e4a02d45cb`; no v11
  column, token, Command or offer shape changed in R1–R2. Findings F1–F6 are
  Slice R3; F7 is deferred.
- Slice R3 (`00151c7d69`): `AbilitySet { Printed(CardDefId), Acquired(GrantKey) }`
  and equality pairing in `linked_ability_source` (CR 607.5); Koh/Boiling Rock
  Rioter test in `tla.rs`; deleted `source_census.rs`, `FunctionZones::Command`,
  `emblems_linked_to`, `ObjectForm.emblem_source`. Focused debug suites and
  pinned pilot green.
- Concept review of R3 (`14a7fb0306`, `scratch/concept-review.html`): groups
  A (planeswalkers) and B (functioning gate) line up. At `bc0486e957`
  `tests/academy/test_authored_match_authority.py` fails (33 digest fields at
  stack-holding decisions 8–9, 30–33; offers/commands/labels identical) and
  debug workspace nextest is 2,243 passed / 14 failed / 3 skipped — all 14 are
  `semantic_conformance_tests frozen_*` trace hashes. Cause: printed abilities
  now serialize `ability_set: Printed(def)` into the state hash.
  Dispositions: G1, G2 → Slice R4; G3 adopted for review (accept the
  `ExileLink.ability_set` hash move, say so in the PR; passed over: auditing
  retained TLA receipts first, since no checked-in receipt failed for it);
  G4 (was F7) stays deferred.

## Next pass — Slice R4: printed abilities stop writing their set

- **G1.** Replace `ability_set: Option<AbilitySet>` / `AbilitySet` on every
  carrier (`ActivatedAbilityDefinition` `state/src/card.rs:1096`,
  `ActivatedAbilityOnStack` / `TriggeredAbilityOnStack`
  `state/src/stack_object.rs:113,180`, `EffectFrame` `flow/decision.rs:239`,
  `PendingTrigger` `flow/trigger.rs`, `EffectiveTrigger`) with
  `acquired: Option<GrantKey>`, `#[serde(skip_serializing_if = "Option::is_none")]`.
  `AbilitySet` survives only in `LinkedAbilitySource` and `ExileLink`, built in
  one derivation in `linked_ability_source` (`Some(key)` → `Acquired`, `None` →
  `Printed(ability_face …)`); delete the three fallbacks and `play.rs`
  `unwrap_or(Printed(..))`. Sites: `flow/triggers.rs:1148–1172,1715`,
  `flow/layers.rs:964,974,1851–2089`, `flow/play.rs:2270–2340`,
  `flow/resolution.rs:678–695,754,806,2298`.
- **G2.** Drop `EffectiveTrigger.clock`; derive `GrantClock` from
  `(acquired, ability_index)` where once-each-turn applies
  (`flow/layers.rs` clock closure, `flow/triggers.rs` `used_trigger_grants`).
- Acceptance, on unmodified evidence: `uv run --extra dev python -m pytest
  tests/academy/test_authored_match_authority.py` and debug
  `cargo nextest run --profile ci -p managym --test semantic_conformance_tests`
  pass; Koh CR 607.5 test in `tla.rs` still passes; debug workspace nextest,
  `size_of::<Game>() <= 5248`, cp313 rebuild, pinned pilot zero drift, release
  `bench_steps` digest `10601622128491155944`.
- Teaches: whether any digest still moves beyond exile links (G3), before ship.

## Deferred findings

- **G4 (was F7) replacement source order.** `ReplacementSourceOrder::{BySeat,
  SlotOrder, ControlledBy}` (`flow/proposed_event.rs` 4 sites,
  `flow/mana.rs:3449`) keeps three call-site traversal orders; CR 616.1 makes
  order the affected player's choice. Collapsing could change which
  replacement a "first applicable" path picks and move recorded games. Needs
  its own CR 616.1 Task: one order plus the choice where two apply.

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
