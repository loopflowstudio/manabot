# T5 (a7234f): permanent copying — plan and status

Status (2026-11-02, HEAD `628608323`, PR #405 open, mergeable): implemented,
demoed through native Commands and the LLM/Academy text path, and synced onto
main at `4c8324a59` (T3's squash, the characteristics cache and T4). All three
cards are defined and supported. The post-sync concept review found nothing to
change and no open product choice (below). Shipping is stopped per Jack Heart's
steer 6eae1230: the PR waits for Jack Heart's review; no merge.

Two accepted models from T3 (`08bf54de2:scratch/6875f0-tla-t3-characteristics.md`)
are the base this builds on: continuous effects as records
(`state/effect.rs`: `Modification` × `Affected` × `EffectDuration`), and open
vocabularies as v5 fact rows (`FactKind`).

## Design review (Jack Heart, 2026-10-30, comment 10a31948) — accepted, done

1. Stack on T3 #399 and drop the duplicate `sorcery_timing`. Done; T3's field
   and `check_activation_terms` are the one implementation.
2. Replace tensor-only copy identity with generic, composable
   modified-characteristics observation (copy + general exceptions + layered
   effects, tensor and tokens). Done as below.
3. Sacrifice history on `TurnState`, shared with T4 raid. Done in T4's shape.
4. Define Phoenix Fleet Airship. Done.

## Cards

| Card | State |
| --- | --- |
| Joo Dee, One of Many | supported; surveil, token copy, sorcery timing |
| Ran and Shaw | supported (out of `tla::INCOMPLETE`); nonlegendary copy |
| Phoenix Fleet Airship | supported; end-step self-copy after a sacrifice, eight-namesake creature, crew 1, flying |

## Model as implemented

- **Copy effect (layer 1) as data.** `Card.copy: Option<CopyEffect { of,
  exceptions }>`. `of` is the printed definition at the root of the copy
  chain; `exceptions: Vec<Modification>` accumulate down the chain
  (CR 707.9b). `Game::create_token_copy` (`flow/zones.rs`) is the one
  authority: it materializes copiable values into the card's definition via
  `Modification::apply_copiable`, so every layer reader starts from them, and
  records the inputs. A departed source is copied from its retained card
  (LKI). The state hash includes `copy` when present.
- **Exceptions share T3's vocabulary.** `Modification::Nonlegendary` is either
  a copy exception or a record; the legend rule reads it through
  `Game::permanent_is_legendary` (`flow/sba.rs`). `apply_copiable` returns
  false for rules effects; a debug assert guards authoring.
  `Effect::CreateTokenCopy { exceptions: Vec<Modification<EffectValue>> }`.
- **Observation.** `PermanentData.copy: Option<CopyData { registry_key, name,
  exceptions }>`; exceptions are `ContinuousEffectData` entries. v5 adds
  `FactKind::Copy` (7) and `FactKind::CopyException` (8, encoded like `Effect`
  rows); native and Python encoders match. The permanent row's definition
  index stays the copied definition. Exceptions tokenize through
  `CreateTokenCopy`'s serialized `Modification` values. LLM text: "copy of
  D<n>, except isn't legendary"; Academy note: "copy of <name>, except isn't
  legendary". Protocol: `copy_of: CopyView` (pydantic's `BaseModel.copy`
  shadows `copy`).
- **Turn history (T4's).** `PlayerTurnHistory { attacked,
  permanents_sacrificed, land_entered, cards_drawn }` on `TurnState.history`,
  `PLAYER_SACRIFICED_INDEX` after attacked (player width 67), always hashed.
  `sacrifice_permanent` increments it. Observed in `PlayerData.turn_history`,
  the protocol (`PlayerTurnHistoryView`), LLM text ("This turn: sacrificed 1
  permanent.", singular) and Academy's PlayerArea.
  `Condition::SacrificedThisTurnAtLeast` reads it.
- **Airship.** `StaticAbility::CreatureWithNamesakes(8)` is computed inside
  the layer stages (private `name_stage` and `ability_loss_timestamp`, never
  the cached view) and counts the controller's permanents with this
  permanent's name, so no name enters the tokens. The end-step trigger uses
  T4's `intervening_if` with `SacrificedThisTurnAtLeast(1)` (CR 603.4).
  Folding `CreatureWithCounters` and `CreatureWithNamesakes` into one variant
  waits for a third self-animation condition (review of `00724828c`).
- **Decision codes.** `continuation::decision_code` is the only mapping from
  obligation to code, bounded by `MAX_DECISION_CODE` (22; Surveil is 22 after
  main's KeepCreatures 20 and Retarget 21). `bindings.rs` validates against
  and exports it; `test_every_native_decision_code_has_a_label` ties the Python
  label table (`_OBLIGATIONS`) to it. `_frame_decision` labels a frame's
  `decision` by frame kind (obligation for kind 6, ordering class for 7/8,
  placement group for 9, delayed-trigger event for 10).

## Concept review after the sync — 2026-11-02 (headless)

Checked the accepted model against the synced tree (`scratch/concept-review.html`).
Nothing to change; no product choice is open.

- One owner per fact survived the sync: a copy is `Card.copy` and the root
  definition; "sacrificed this turn" is T4's `PlayerTurnHistory`, read by
  `Condition::SacrificedThisTurnAtLeast` and `EffectValue::PermanentsSacrificedThisTurn`;
  every sacrifice (effect, cost, forced) increments it through
  `sacrifice_permanent`. No `StaticCondition`, `CreateNonlegendaryCopy` or
  second `sorcery_timing` remains.
- The three cards follow set conventions: firebending is the attacks trigger
  every TLA card uses; the copy exception is the shared `Modification`.
- Not adopted, as before: folding `CreatureWithCounters` and
  `CreatureWithNamesakes` (waits for a third self-animation condition).
  Mixed `"legendary"` / `"Legendary"` supertype casing predates T5 (11 cards
  on main; readers compare case-insensitively, and tokens carry `is_legendary`,
  not the string); a cosmetic cleanup for any batch, not a T5 decision.

## Sync onto main — 2026-11-02 (steer 6eae1230)

- Merged with T3's head `a18902e0e` as the base (main's T3 is its squash), so
  only T5's own diff met the cache and T4: ~60 hunks instead of ~1465.
- Dropped T5's own `PlayerTurnHistory`, player column, `update_turn_history`
  (engine undo is gone) and `StaticCondition` for T4's; Ran and Shaw keeps
  T4's `intervening_if` with `CreateTokenCopy`.
- Re-recorded in the sync: protocol schemas, semantic-kernel w17 evidence
  (14 cases / 3396 commands), belief v3 receipt pins.
- Integration (main) on #405 (`552636009`, 2026-10-30) failed only the four
  `test_policy_choice_can_learn_either_preference[*-target-*]` cases. Cause:
  the branch then stacked on T3 before `a18902e0e`, which erases the v5
  timestamp-rank column with the definition in `verify/action_cases.py`'s
  target tie. Main's T3 squash carries that change; the file passes on the
  synced tree (25 passed, 2026-11-02). CI on `1e9fd2d7c` is running.

## Remaining

- Gate (re-gate per steer 6eae1230): re-record whatever the sync left stale
  among the authored-match receipt and its SHA pins in both parity modules,
  authored-match parity, public-commitment parity, Academy demos, choice
  navigation and trigger-order fixtures; then debug `cargo test`, clippy,
  focused Python suites and frontend checks. Confirmed stale on 2026-11-02 (digests only): the
  authored-match receipt (`test_checked_authored_match_parity_receipt`), the
  trigger-order root digests (`test_survival_orders_cross_consumers`, 2 cases)
  and the Academy precision demos (`test_decision_surface`, 3 cases). Choice
  navigation, authored-match authority, Learn evidence, belief state and
  learning, and continuation input already pass; public-commitment parity
  is unchecked since the sync. CI on `1e9fd2d7c` (2026-11-02): Rust, Python
  unit, conformance and Clean-machine Play passed; **Academy UI failed** on the
  three Playwright `decision-surface.spec.ts` Precision cases because
  `play_demo.py` rejects the diverged precision demo digest before any
  decision is offered, so the decision surface never appears. Same cause as
  the three stale `test_decision_surface` cases; re-recording the demos fixes
  both. Integration (main, long) were still running.
- Jack Heart's review of PR #405. No merge.
- Known, not fixed: `tests/sim/test_rollout.py::TestBatchedSampler::test_selects_valid_actions_only`
  fills every float buffer with noise, which the fact and definition checks
  reject. CI does not select the file.

## Delete — do not maintain

- `Effect::CreateNonlegendaryCopy` (main) and `CreateTokenCopy { nonlegendary }`:
  replaced by `{ exceptions }`. Done.
- The branch's duplicate `ActivatedAbilityDefinition.sorcery_timing`: T3's is
  the one field. Done.
- The tensor-only "copy identity" model: superseded by `CopyEffect` and its
  fact rows. Done.
- T5's own `PlayerTurnHistory` and `StaticCondition::SacrificedThisTurnAtLeast`:
  T4's. Done at the sync.

## Checks

- 2026-11-02 (realign, HEAD `1e9fd2d7c`): `uv run --extra dev pytest
  tests/model/test_action_identity.py -q` passed (25), confirming the CI
  diagnosis above. The receipt suites
  (`tests/academy/test_authored_match_parity.py test_authored_match_authority.py
  test_choice_navigation.py test_trigger_order.py test_learn_evidence.py
  test_decision_surface.py tests/belief/test_state.py test_learning.py
  tests/verify/test_continuation_input.py`) ran 55 passed, 6 failed, all six
  the stale digests named under Remaining.
- 2026-11-02 (sync): debug `cargo test --no-fail-fast` (all targets after the
  conformance re-record), text observation, observation parity, `tests/env/`,
  belief state, experience protocol, vitest permanent notes and GameBoard
  passed.
- 2026-10-30 (gate, pre-sync tree): `cargo fmt --check`, debug `cargo test`
  (21 targets, 0 failed), clippy `-D warnings` (`PYO3_PYTHON` must be
  absolute), `semantic_conformance check`, the focused Python suites listed in
  `AGENTS.md`, vitest (22) and `npm run check` passed.
- 2026-10-30 (demo, headless, record in `scratch/t5-demo.md`): the first
  pass found the text-path decision-code gap (fixed at `2963f8f87`, regression
  `test_joo_dee_surveil_and_copy_play_through_the_text_decision`); the second
  and third passes showed Joo Dee, Ran and Shaw and the Airship through the
  text path.
