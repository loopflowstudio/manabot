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

## Remaining — coverage decomposition (after behavior review `e5254b3cb6`)

Review criterion: Garruk, Curse Breaker; Jace, Reality Sculptor; Chandra,
Flameshaper; Chandra, Torch of Defiance; Sanctum Lurker (zero-loyalty
exemption) and Sakashima of a Thousand Faces compose from small orthogonal
primitives, with no card-specific engine logic, multi-axis special variant or
second owner of one rule. Three slices; each cuts every consumer over and
keeps its directed tests on the surviving owner. Stored TLA token spellings
and the pinned pilot stay unmoved in every slice (zero drift, no
`pinned_pilot.py record`).

### Slice 1 — delayed watchers match through `TriggerCondition`

The branch added a third copy of attack matching: `DelayedEvent::Attacks`
(state) → `DelayedTriggerEvent::Attacks` (flow) → its own arm in
`delayed_triggers_for_event`, beside `collect_occurrences` and
`trigger_condition_fires`. Every further "until your next turn, whenever X"
would repeat that in five places although `TriggerCondition` already says X.

- Add `DelayedEvent::Whenever(Box<TriggerCondition>)` and flow
  `DelayedTriggerEvent::Whenever`, matched by the ordinary
  `trigger_condition_fires` / `collect_occurrences` / `trigger_context` with
  the watcher (source identity, controller) as the source snapshot.
  Description and learning tokens come from the `TriggerCondition` template.
- The ordinary `Per::Batch` attack occurrence must bind its matching attackers
  ("those creatures"), as the delayed arm does today through `remembered`.
  Put that binding on the shared matcher; Garruk reads `Remembered`.
- Fold `AttackedSubject` into two axes on the event: whose side is attacked
  (`Actor`, default any) and whether planeswalkers count
  (`Player` / `PlayerOrPlaneswalker`). `Any` stays skipped in serialization so
  existing `Attacks` tokens (FDN, GTC, TLA, RIX) are unchanged.
- Cut over Garruk, Curse Breaker −4 and Jace, Reality Sculptor −3; keep their
  `fra.rs` behavior tests and the attack-trigger tests in
  `tests/rules/cr_603_triggers.rs`.
- `Dies`, `CreatureDies`, `EndStep`, `YouCastSpell`, `Upkeep` keep their
  spelling: TLA tokens and retained state hashes name them. They are the
  closed legacy set; add no further `DelayedEvent` variant.

### Slice 2 — impulse play composes from exile, select and grant

`ExileTopCardsToPlay` gained a `choose` axis and builds its own
`Decision::SelectCards`, while `Effect::SelectCards`, `GrantPlay` and
`ObjectSelector::Moved` already express "choose one of them; you may play it".

- `ObjectSelector::LibraryTop` takes a count (branch-new variant; Chandra,
  Torch of Defiance is its only card user).
- Chandra, Flameshaper +2 becomes `exile(LibraryTop 3)`, then
  `SelectCards { from: Moved, min 1, max 1, effects: [GrantPlay { Selected,
  ThisTurn }] }`. `GrantPlay` gets its first card consumer.
- Semantic check first: the composed exile must mark each card's
  `library_top_departure` and leave viewer knowledge as `exile_top_cards`
  does; with fewer than two exiled cards the choice is forced or empty.
  A difference is a rules-contract fix before the cut-over.
- `ExileTopCardsToPlay { count, until, face_down, condition }` stays as the
  stored spelling for TLA and Flameshape.

### Slice 3 — one static for "state-based action doesn't apply"

`StaticAbility::LegendRuleDoesntApply` and the branch's
`StateBasedActionDoesntApply { action, scope }` are two variants for one axis.

- Add `StateBasedAction::LegendRule`; Sakashima and its `GrantStatic` copy
  exception use `{ LegendRule, YouControl(any) }`.
- `Characteristics.exempt_from_zero_loyalty` becomes one small set of exempt
  actions written by the single `Layer::Ability` projection; `legend_groups`
  and the zero-loyalty check both read it per permanent.
- Retire the `LegendRuleDoesntApply` token in `cards/token_renames.rs`; move
  its observability witness onto the surviving variant.

### Delete — do not maintain (deepest owner first)

1. The `DelayedTriggerEvent::Attacks` arm of `delayed_triggers_for_event` and
   its batch-binding block (`flow/triggers.rs`).
2. `attacked_subject_matches`'s four-variant match, `AttackedSubject` and its
   card-less `PlaneswalkerControlledBy`.
3. `DelayedTriggerEvent::Attacks`, `DelayedEvent::Attacks`, their lowering arm
   in `Effect::CreateDelayedTrigger` and description arm.
4. The inline `Decision::SelectCards` branch of `Effect::ExileTopCardsToPlay`,
   the `choose` field, its description branch and
   `Effect::exile_top_choose_to_play`.
5. `StaticAbility::LegendRuleDoesntApply`, the `names.clear()` scan in
   `legend_groups` and `Characteristics.exempt_from_zero_loyalty`.

### Already meets the bar (no slice)

- Loyalty: `LoyaltyCost` on the activation, one per-turn flag, timing as
  `Modification::LoyaltyAtInstantSpeed`; no planeswalker-specific resolution.
- Emblems: catalog `EmblemDefinition` rows whose statics and triggers run
  through `layers.rs` / `triggers.rs` like any source.
- Empower is a constructor over `If` + `create_tokens` + `PutCounters`.
- Death-to-exile: the `Modification`, the printed replacement and finality
  already share `collect_replacements`. A general zone-change replacement
  waits for a card that changes origin or destination (the open
  battlefield-to-exile defect is its first consumer).
- `Fight { first, second }`, `RevealUntil { to }`, `Except::Matching`,
  `TargetDivision`, `NextSpellYouCast` each hold one axis.
  `LibraryExceptBottom` has one consumer and no second axis to split.

After the slices: rebuild the cp313 extension (compress and these slices
change Rust after Pass 4's rebuild), debug nextest, pinned pilot, refresh
Academy fixtures if definition ids or tokens move. Deferred to gate/CI:
same-runner Engine Budget comparison and the full CI matrix (incl. Python 3.12).

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
