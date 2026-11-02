# T5 (a7234f): permanent copying — plan and status

Status (2026-10-30): implemented, demoed through native Commands and the
LLM/Academy text path (after the `2963f8f87` capture fix), and stacked on T3 #399
at its head `a18902e0e` (merged at gate; it erases the v5 timestamp rank in the
target-tie case, which clears the 4 inherited `test_action_identity` failures).
All three cards are defined and supported. T4 488a40 published as draft PR #407
(on T3, failing CI, still moving); this branch has not synced onto it.

T3's plan, inherited through the stack, is at
`08bf54de2:scratch/6875f0-tla-t3-characteristics.md`. Two of its accepted
models are the base this builds on: continuous effects as records
(`state/effect.rs`: `Modification` × `Affected` × `EffectDuration`), and open
vocabularies as v5 fact rows (`FactKind`).

## Design review (Jack Heart, 2026-10-30, comment 10a31948) — accepted, done

1. Stack on T3 #399 now; drop the duplicate `sorcery_timing`. Done: the
   merge kept T3's identical field and check (`check_activation_terms`).
2. Replace tensor-only copy identity with generic, composable
   modified-characteristics observation (copy + general exceptions +
   layered effects, tensor and tokens). The shape was proposed in scratch
   first, then implemented as below.
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
  chain. `exceptions: Vec<Modification>` accumulate down the chain
  (CR 707.9b). `Game::create_token_copy` (`flow/zones.rs`) is the one
  authority. It materializes the copiable values into the card's definition
  via `Modification::apply_copiable`, so every layer reader starts from them.
  It also records the inputs. A departed source is copied from its retained
  card (LKI). The state hash includes `copy` when present.
- **Exceptions share T3's vocabulary.** The new `Modification::Nonlegendary`
  is either a copy exception or a record. The legend rule reads it through
  `Game::permanent_is_legendary` (`flow/sba.rs`). `apply_copiable` returns
  false for rules effects; a debug assert guards authoring.
  `Effect::CreateTokenCopy { exceptions: Vec<Modification<EffectValue>> }`.
- **Observation.**
  - `PermanentData.copy: Option<CopyData { registry_key, name, exceptions }>`.
    Exceptions are `ContinuousEffectData` entries (Indefinite, the copy's
    timestamp, the copied definition as source).
  - v5 adds `FactKind::Copy` (7: id `name_value("CopyEffect")`, source = the
    copied definition + 1, order = the copy's timestamp rank) and
    `FactKind::CopyException` (8: encoded like `Effect` rows through
    `effect_facts`). Native and Python encoders match (`FACT_KINDS` 8).
  - The permanent row's definition index stays the copied definition.
  - Exceptions tokenize through `CreateTokenCopy`'s serialized
    `Modification` values.
  - LLM text: "copy of D<n>, except isn't legendary". Academy note: "copy of
    <name>, except isn't legendary".
  - Experience protocol: `copy_of: CopyView`. The wire name avoids pydantic's
    `BaseModel.copy`.
- **Turn history.**
  - `PlayerTurnHistory { permanents_sacrificed }` on `TurnState.history`
    uses T4's names. Writes go through `Game::update_turn_history`
    (journaled), incremented in `sacrifice_permanent`, and the history resets
    each turn.
  - It is hashed only when non-empty, so sacrifice-free state hashes are
    unchanged.
  - Observed in `PlayerData.turn_history`, a v5 player column
    (`PLAYER_SACRIFICED_INDEX`, width 63 → 64), the protocol
    (`PlayerTurnHistoryView`), LLM text ("This turn: sacrificed N
    permanents.") and Academy's PlayerArea.
  - `StaticCondition::SacrificedThisTurnAtLeast` reads it.
  - `cards_drawn_this_turn` stays main's field. T4 folds it into the struct.
- **Airship.** `StaticAbility::CreatureWithNamesakes(8)` is read in
  `permanent_is_animated` and counts the controller's permanents with this
  permanent's `permanent_name`, so no name enters the tokens. The trigger is
  `ActiveIf { SacrificedThisTurnAtLeast(1), BeginningOfYourEndStep }` with an
  `IfCondition` recheck (CR 603.4).

## Concept review (headless, 2026-10-30, HEAD `187d4e0d7`) — one change

The card, copy, exception, turn-history and Airship model still match Jack
Heart's design review (comment 10a31948). No product choice is open. The
`00724828c` review stands: folding `CreatureWithCounters` and
`CreatureWithNamesakes` into one variant waits for a third self-animation
condition.

The decision-code fix (`2963f8f87`) gives Rust one owner, `decision_code`,
bounded by `MAX_DECISION_CODE`. Python keeps the labels (`_OBLIGATIONS` in
`manabot/semantic/text_observation.py`; text-rendering vocabulary). The bound
is the shared fact: the native module exports `MAX_DECISION_CODE` beside
`WORLD_VERSION`, and `test_every_native_decision_code_has_a_label` asserts
`len(_OBLIGATIONS) == MAX_DECISION_CODE + 1`, so a new code fails in CI
rather than mid-game. Done 2026-10-30; extension rebuilt; that test,
`tests/semantic/test_text_observation.py` (38 passed) and debug
`cargo test --lib agent::continuation` (5 passed) are green.

## Concept review (headless, 2026-10-30, HEAD `a907cbe22`) — nothing to change

The decision-code direction above is implemented as recorded: `continuation.rs`
owns `decision_code` and `MAX_DECISION_CODE`, `bindings.rs` validates against it
and exports it, and the Python label table is tested against it. The card, copy,
exception, turn-history and Airship model is unchanged since the previous
review. No product choice is open, and the `CreatureWith*` fold still waits for
a third condition.

## Remaining

- Done at publish (2026-10-30): w17 evidence re-recorded in place for T5's
  content digest. Semantic kernel: same 14 cases / 3401 commands, only
  state hashes and content digests moved. Also re-recorded: authored-match
  receipt, with its SHA pinned in both parity modules; authored-match parity
  (142 commands); public-commitment parity (69 commitments, no gaps); Academy
  demos and choice navigation (same decision revisions); trigger-order
  digests; belief v3 receipt pins.
- Sync onto T4 once #407 leaves draft with its CI settled. Gate left it out on
  2026-10-30: #407 is a draft with failing Rust, Semantic Kernel, Integration
  and Academy UI checks and a running Task, and a dry-run merge conflicts in
  about 25 source files and every evidence replay. On that sync:
  - `PlayerTurnHistory` takes T4's other fields.
  - `cards_drawn_this_turn` moves into the struct.
  - The player column moves to T4's `PLAYER_SACRIFICED_INDEX` position, after
    attacked.
  - w17 evidence is re-recorded for the combined content digest.

## Sync onto main (T3 squash, characteristics cache, T4) — 2026-11-02

Jack Heart's steer 6eae1230: T3 and T4 merged; sync and fold sacrifice history
into T4's `PlayerTurnHistory`.

- Resolved against T3's head `a18902e0e` as merge base (main's T3 is its
  squash), so only T5's own diff met the cache and T4: ~60 hunks instead of
  ~1465.
- Turn history is T4's: `PlayerTurnHistory { attacked, permanents_sacrificed,
  land_entered, cards_drawn }`, `PLAYER_SACRIFICED_INDEX` after attacked
  (player width 67), always hashed. T5's struct, column, `update_turn_history`
  (engine undo is gone) and `StaticCondition` are dropped; the Airship trigger
  uses T4's `intervening_if` and `Condition::SacrificedThisTurnAtLeast`, and
  Ran and Shaw keeps T4's `intervening_if` with `CreateTokenCopy`.
- Decision codes keep main's numbering: KeepCreatures 20, Retarget 21;
  Surveil moves to 22 (`MAX_DECISION_CODE` 22). The text labels follow, and
  kind-10 waiting delayed triggers label their event through
  `_frame_decision`.
- `CreatureWithNamesakes` is computed inside the layer stages, so it reads a
  private `name_stage` and `ability_loss_timestamp`, never the cached view.
- Wording: "sacrificed 1 permanent" / "drew 1 card" singular in LLM text and
  Academy (main's test expected "drew 1 cards").
- Re-recorded in the sync: protocol schemas, semantic-kernel w17 evidence
  (14 cases / 3396 commands), belief v3 receipt pins. Taken from main and
  still to re-record at gate: authored-match receipt and its SHA pins in both
  parity modules, authored-match parity, public-commitment parity, Academy
  demos, choice navigation and trigger-order fixtures.
- Checked: debug `cargo test --no-fail-fast` (all targets pass after the
  conformance re-record), text observation, observation parity, `tests/env/`,
  belief state, experience protocol, vitest permanent notes and GameBoard.

## Delete — do not maintain

- `Effect::CreateNonlegendaryCopy` (main) and `CreateTokenCopy { nonlegendary }`
  (this branch's first slice): replaced by `{ exceptions }`. Done.
- This branch's duplicate `ActivatedAbilityDefinition.sorcery_timing`: T3's
  is the one field. Done.
- The tensor-only "copy identity" model (shared key plus card-row bits as the
  only signal): superseded by `CopyEffect` and its fact rows. Done.

## Checks

- 2026-10-30 (realign, HEAD `552636009`): `uv run --extra dev pytest
  tests/semantic/test_text_observation.py -q` passed (38, including
  `test_every_native_decision_code_has_a_label`); `managym.MAX_DECISION_CODE`
  imports as 20. Full debug `cargo test` and CI on the fix commits are left
  to gate/CI.

- 2026-10-30 (gate, tree after merging T3 `a18902e0e`):
  - `cargo fmt --check`, debug `cargo test` (`CARGO_PROFILE_DEV_OPT_LEVEL=1`,
    21 targets, 0 failed), `cargo clippy --all-targets --all-features -D
    warnings`, and `semantic_conformance check` (14 cases / 3401 commands)
    passed. Clippy's `PYO3_PYTHON` must be absolute; pyo3's build script does
    not resolve `../.venv` from `managym/`.
  - Python focused suites passed: the world/observation list in `AGENTS.md`,
    `tests/model/` (including `test_action_identity`), text observation,
    observation parity, continuation input, belief state, heuristic, Academy
    navigation, Learn, experience protocol, authored-match parity, play modes,
    active training, `test_server -k protocol_v1`, and public-commitment parity
    (69 commitments, no gaps).
  - Frontend: vitest (22 passed), `npm run check` (0 errors).
  - Re-recorded the authored-match parity receipt. Only its source hashes moved,
    for `continuation.rs` and `bindings.rs`, after the decision-code fix.
  - Not fixed: `tests/sim/test_rollout.py::TestBatchedSampler::test_selects_valid_actions_only`
    fails. Its synthetic batch fills every float buffer with noise, which the
    fact and definition checks reject. CI does not select this file.
  - Fixes: "sacrificed 1 permanent" singular in LLM text and Academy; the
    leftover closure in `modificationPhrase` removed.
  - Browser suites and clean-machine Play ran only in CI, and passed on
    `552636009`. The gate commits are unpushed, so CI has not run them.

## Demo (headless, 2026-10-30) — text-path gap fixed at `2963f8f87`

This demo ran headless. The agent stood in for Jack Heart; no human watched it.
The full record is in `scratch/t5-demo.md`.

- The first pass at `edc1fabd8` found a text-path gap. Capture failed
  everywhere text is captured: LLM runner and session, `verify/llm_game.py`,
  and Academy's `study_protocol.py`. `parse_continuation_frames_json` rejected
  `decision > 13`, which ruled out main's codes 14–18 as well as T5's 19
  (DistributeCounters) and 20 (Surveil). Joo Dee's surveil hit "invalid
  continuation frame relationship". The tensor path was unaffected.
- Fixed:
  - `continuation::decision_code` is the only mapping from obligation to code.
  - `MAX_DECISION_CODE` (20) bounds the validator in `bindings.rs`.
  - `every_decision_code_is_within_the_validated_bound` builds every
    `Decision` variant.
  - `text_observation.py` labels a frame's `decision` by frame kind
    (`_frame_decision`), and its obligation table now covers 14–20.
  - Regression test:
    `test_joo_dee_surveil_and_copy_play_through_the_text_decision`.
- Second pass, rebuilt extension, text path:
  - Joo Dee: surveil, then copy, then sacrifice. The result shows "copy of D1"
    and "sacrificed 1".
  - Ran and Shaw: copying a cast Ran and Shaw makes the token nonlegendary.
  - Airship: the end-step self-copy follows a sacrifice.
  - Scripts: `scratch/tools/demo_t5.py` and `demo_t5_more.py`.
- CI on PR #405, before the fix: only `Integration (main)` failed, on the 4
  inherited `test_policy_choice_can_learn_either_preference[*-target-*]`
  cases. CI has not yet run on the fix commits.
