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

- Slice R4 (`38c9318125`, post-MTG-439 sync receipts `57f07018df`): replaced
  `Option<AbilitySet>` / `AbilitySet` on every ability carrier with
  `acquired: Option<GrantKey>` (omitted from serialization when `None`;
  `AbilitySet` now lives only in `LinkedAbilitySource` and `ExileLink`,
  derived once in `Game::linked_ability_source`); dropped
  `EffectiveTrigger.clock` and derived `GrantClock` from
  `(acquired, ability_index)` via `EffectiveTrigger::clock()`. Added
  serialization/clock regression test in `match_state_hash_tests.rs`.
  Note: sync merge `134d4c07bd` (main `d9751886b6`: `9f20178dbf` Academy
  history + `f4b272349b` MTG-439 auto-skipped forced dual-land payments) had
  left the `-planeswalkers` receipts at pre-sync counts (174 commands / 193
  commands on `ur-vs-gw-5eed`), so `57f07018df` refreshed them to the
  post-sync line (`170` deliberate commands, identical `TERMINAL_STATE`
  `c3947b7f2c1cc675a34b8884daa2a5efe280ba937c970faf0cad8d5ea974492d`, all 14
  `semantic_conformance` command counts matching main's
  `evidence-forced-mana-v1.json`). Verified: `match_state_hash_tests` (12),
  `functioning_abilities` (4), Koh CR 607.5 & `power_batch_grantors` card
  tests, `size_of::<Game>() <= 5248` (compile-time const), debug workspace
  `cargo nextest run --profile ci --workspace`, release `bench_steps`
  (`115,941` steps, digest `4186394851247720253` matching
  `managym/perf/baseline.json`), cp313 rebuild, `test_pinned_pilot.py` (zero
  drift), `test_authored_match_authority.py` + `test_authored_match_parity.py`
  (8/8), and `scripts/public_api.py check`. All planned passes built.
- Concept review of R4 (`f1db71c3fa`, `scratch/concept-review.html`): groups
  A–C line up; `match_state_hash_tests` + `semantic_conformance_tests` 32/32
  and authored-match authority/parity 8/8 at `4211b4bac1`. Dispositions:
  G5 adopted for review — a Layer-6-granted ability on the stack or pending
  writes `"acquired": {grantor, timestamp, static_slot}` into the state hash
  (real CR 607.5 state, same class as G3; no checked-in receipt holds one);
  say so in the PR beside G3. Passed over: auditing retained TLA receipts
  first. G6 deferred (below). G7 fixed: comment in
  `Game::linked_ability_source` naming which frames lack `ability_face`.
  No remaining passes.

## Deferred findings

- **G4 (was F7) replacement source order.** `ReplacementSourceOrder::{BySeat,
  SlotOrder, ControlledBy}` (`flow/proposed_event.rs` 4 sites,
  `flow/mana.rs:3449`) keeps three call-site traversal orders; CR 616.1 makes
  order the affected player's choice. Collapsing could change which
  replacement a "first applicable" path picks and move recorded games. Needs
  its own CR 616.1 Task: one order plus the choice where two apply.

- **G6 two grant records.** `granted_by: Option<AbilityGrant>` (exact
  grantor `ObjectId` + `AttachedHas` index) sits beside `acquired:
  Option<GrantKey>`; `flow/play.rs` only `debug_assert!`s that the first
  implies the second. Fold into one `Acquisition { key, grantor,
  attached_index }` record in its own Task: it moves main's `granted_by`
  bytes and TLA equipment's catalog-program lookup, so it needs a
  pinned-pilot and receipt check.

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
