# T5 (a7234f): permanent copying — plan and status

As of 2026-11-02, local and published PR #405 head is `1de85f7aa`;
main `29660d00c` is merged and remains the upstream head. Joo Dee, One of Many,
Ran and Shaw, and Phoenix Fleet Airship are defined and supported. Implementation and the
native Command / LLM text demos are complete. The choice-navigation repair
and updated PR description are published. Gate is complete: CI run 41541743395
passed at that head, and the local regression/CI-selection additions passed
their focused checks. Those gate additions remain uncommitted; runtime code is
unchanged. See the gate results below for exact scope and skips.

**Shipping is stopped.** Jack Heart's latest steer f39fef34 says to run the
Graph through gate, then stop for review: no merge. The Task's recorded
2026-11-02T21:35:52Z auto-ship approval and “running from land --wait-and-fix”
Graph label do not override this explicit direction.

Prior review narratives and detailed check history:
`bbb5101eb:scratch/jack-heart/a7234f-tla-t5-permanent.md`.
The implement note at compression entry confirmed no source
changes since `924e520b4`; its corrected characteristic authority is preserved
here and the completed deletion list is retained at `933798396`. Its original diff is also in
`.lf/tmp/t5-compress-input.patch`.

## Accepted design

Jack Heart, 2026-10-30, comment 10a31948: stack on T3 #399 and reuse its
sorcery timing; observe copies and general exceptions through composable
modified-characteristic facts and tokens; share sacrifice history with T4;
define Phoenix Fleet Airship. All four are implemented. T3's continuous-effect
records and v5 fact rows are the substrate
(`08bf54de2:scratch/6875f0-tla-t3-characteristics.md`).

Acceptance remains all three cards' allocated obligations, directed debug
rules witnesses, current-world support, and viewer-safe complete observations
and actions through Academy and training. Passing a shared consumer path alone
does not establish rules correctness. Jack Heart's review is still required.

Engine scope is copiable-value authority and dynamic ability inheritance for
these three self-copy cards; arbitrary face-aware consumers finish in T6.
Allocation and obligations remain in `docs/rules/tla/allocation.jsonl`.

## Concept review with Jack Heart — decisions 1a/2a, 2026-11-02

Jack Heart selected both changes before merge:

- **1a:** Legendary is a layer-4 `Characteristics.is_legendary` value:
  copiable supertypes minus `Nonlegendary` records. The legend rule reads
  that view; remove `permanent_is_legendary`.
- **2a:** Every battlefield `CardData` field stating a mutable characteristic
  reads `Game::characteristics()`; off the battlefield it reads the printed
  card. Classify every field as layered or printed by definition in
  `tests/cards/observability.rs`, with an exhaustive destructure. Add missing
  characteristics to the view rather than another printed reader.

Implemented in `16e28c304`, simplified in `924e520b4`, evidence realigned in
`889948346`. The guard classifies all 20 fields (9 layered, 11 printed with
reasons). `Characteristics.types` owns type flags (`is_creature()` and
`is_land()` derive from it); `ward` disappears under ability loss. Card rows
use struct update over `printed_card_data`, with `CardTypeData::from_types`
as the shared type builder and `Card::is_legendary` as the printed reader.
The legend-rule/column-39 witness, all-layered-fields witness, off-battlefield
witness and text-renderer test cover the split. Widths and protocol fields
are unchanged. Battlefield card text now says “keywords,” not “printed
keywords.”

The headless concept review after 1a/2a found no further T5 product choice
(`scratch/concept-review.html`):

- Text box and mana cost stay printed because no current modification changes
  them; copies materialize both in the definition. The first text-changing or
  cost-setting effect must add the value to `Characteristics` and reclassify
  it as layered.
- `PermanentData.name` and `creature_types` are Aura-set markers for fact rows;
  the card row's name is the actual layered name. Keep the permanent columns
  consumed by the tensor, Python and text paths.
- `base_power` / `base_toughness` now equal layered P/T on the battlefield.
  A printed-P/T accessor or protocol removal is a separate decision (below).
- Do not fold `CreatureWithCounters` and `CreatureWithNamesakes` before a
  third self-animation condition justifies it. Mixed supertype casing predates
  T5; readers are case-insensitive and tokens use the legendary flag.

Rechecked at `933798396` on 2026-11-02 after compression: **no design change
and no open T5 product choice**. Production behavior has not changed since
the preceding concept review. The normal interaction remains activate →
private surveil choice → copy enters → choose a sacrifice; committed costs
and revealed information survive interruption. Recovery must return the actual
pending decision through the existing revision-bound Command contract, never
restart the ability or choose a sacrifice in Academy. A departed copy source
retains its copiable definition; counters and temporary grants must not leak
into the copy. These are the counterexamples that would invalidate the design,
not new product choices. Existing directed witnesses cover private surveil,
copy inheritance and departed sources; this review adds no runtime proof or
new reconnect claim. The next proof remains current-head gate acceptance.
The deferred printed-P/T alternatives below remain unapproved and outside T5;
no documentation or skill-instruction redesign is needed for this outcome.

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
  a copy exception or a record; the legend rule and battlefield card row read
  `Game::characteristics(id).is_legendary`. `apply_copiable` returns
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

## Evidence and failure context

- Sync used T3's pre-squash head `a18902e0e` as merge base, avoiding replay
  of its squashed changes. T4 owns turn history and the sacrifice column;
  Surveil appends code 22 after KeepCreatures 20 and Retarget 21.
- Initial Integration (main) failure on #405 (`552636009`, 2026-10-30)
  was four target-choice separability cases. T3 `a18902e0e` clears the v5
  timestamp rank with the target's definition. Synced
  `uv run --extra dev pytest tests/model/test_action_identity.py -q` passed 25.
- The first demo failed because capture rejected decision codes above 13.
  The native shared bound and kind-specific text labels repaired it;
  `scratch/t5-demo.md` retains the negative result and subsequent demos.
- Re-gate after sync re-recorded content-bound w17 receipts and precision
  demos; the authored-match authority receipt remained byte-identical.
  Release Playwright passed 25, skipped 2 environment-gated cases. The
  subsequent 1a/2a CI run 41536286824 at `e8fd88eed` failed Academy UI's
  CHOOSE_TARGET and DISCARD cases: retained `base_power` / `base_toughness`
  still held printed values after card rows became layered. Release-packages
  run 41536286630 succeeded; it was not the failing CI run.
- `889948346` re-recorded choice navigation and preserved recorded offer ids
  when labels repeat. Only projection P/T, match ids, frame hashes and source
  commit changed; state digests and offer order stayed fixed. The kernel,
  authored-match, demo and trigger-order receipts did not move. Academy UI
  passed in CI run 41539137476 at `bbb5101eb`, verifying the repaired live
  cases. That run's Integration (main) was still running when inspected.
- `933798396` checkpoints the subsequent test-helper and note compression;
  production behavior is unchanged. PR #405 now describes layered legendary,
  the CardData rule and the repaired fixture. No description refresh remains.

## Remaining

- Gate completed at published head `1de85f7aa`: all required CI jobs and
  aggregate Tests Result passed in run 41541743395. Local additions improve
  copy-fact parity coverage and select that bounded test in CI; their focused
  checks passed. Preserve these additions and the preceding demo evidence in
  the next authorized checkpoint/publication. No production repair remains.
- Open for Jack Heart: the projection's `base_power` / `base_toughness`
  were documented as printed values for a "4/4 (2/2)" display; with the
  card row layered they equal `power` / `toughness` on the battlefield, and
  printed P/T has no other Python-visible source (the pack manifest carries
  ids and names only). No UI component reads them today. Choices: (i) keep
  them as the card row's value (current; the comment now says so), (ii)
  add a printed-P/T accessor to the engine and keep the field printed, or
  (iii) drop the fields (protocol change). (ii) and (iii) are follow-up
  Tasks, not T5 blockers.
- Then Jack Heart's review of PR #405. No merge: steer f39fef34 is the latest
  direction, and the Task record's auto-ship approval (21:35:52Z) does not
  override it.
- Known, not fixed: `tests/sim/test_rollout.py::TestBatchedSampler::test_selects_valid_actions_only`
  fills every float buffer with noise, which the fact and definition checks
  reject. CI does not select the file.

Obsolete copy/legend readers, duplicate T3/T4 mechanisms, and spent migration
and capture-probe scripts are removed. Their deletion history is retained at
`933798396`; there are no remaining deletion targets in the accepted design.

## Validation handoff

Reusable evidence (2026-11-02): debug `cargo test --test cards --test
rules_tests` passed 501 + 141 and clippy `--all-targets --all-features -- -D
warnings` passed at `924e520b4`; extension rebuilt for `e8fd88eed`.
The 1a/2a focused Python observation, parity, Academy, content and forced-choice
suites passed 75. After fixture re-record, `uv run --extra dev pytest
tests/academy/test_choice_navigation.py -q` passed 7 (5 failures beforehand).
Full prior output and commands remain at the history reference above.

Compression keeps all behavioral tests, shares the Ran and Shaw graveyard
setup, uses `Characteristics::pt()` in T5 assertions and names the injected
renderer test for what it proves. Production source and receipts are unchanged.
Compression's recorded focused checks passed 390 TLA tests and the injected
text-renderer test; format/lint were clean. Reconciliation changes context only
and adds no new runtime claim.

Check: `git diff --check` — passed; `lf context` — memory and scratch within both token and byte limits; current-head acceptance deferred to gate/CI.

## Gate at `1de85f7aa` — 2026-11-02

Comparison: checkout base/main `29660d00c` through HEAD, plus the existing
demo note and its untracked evidence. The latest commit changes context only;
native inputs still match the rebuilt extension at `e8fd88eed` (the Python
projection has only a comment change since that build).
The repository selector (`git diff --no-renames --name-only -z 29660d00c HEAD |
bash .github/ci/select-tests.sh`) selects all eight CI job groups. It has no
local result cache. CI run 41541743395 is exercising this exact published head;
full native suites, full model/integration suites and headless release browser
coverage remain with that run rather than being duplicated locally.

Planned focused checks, selected before execution:

- `managym/src/{flow,state,agent}` and native tests: retain the same-head demo's
  51 observability and 13 card witnesses; run continuation unit tests and
  `cr_613_continuous_effects` in optimized debug, then cargo fmt and all-target,
  all-feature clippy with two build jobs. These cover the continuation bound,
  layered legend rule and static correctness beyond the card demo.
- `manabot/env/observation.py`, native bindings, text, experience projections,
  fixtures and changed belief test: run focused observation/parity/continuation,
  content, forced-choice, protocol, choice/trigger, Learn setup and belief
  tests. Retain the same-head demo's text-consumer evidence. Add the mandated
  current-world admission/monitoring/choice-separability and active-training
  checks because rules and policy-input facts reach those consumers.
- Academy notes, types, player history and decision labels: run focused Vitest
  notes/protocol/label tests and Svelte type checking; CI owns the built release
  browser cases, including the previously failing choice-navigation cases.
- Changed Python and stubs: Ruff lint and format checks; final diff whitespace
  check. No configured Python type-check command is present in CI.

No merge, Task completion, new work, or Wave-memory change is part of this gate.

Review found no production defect. The all-fields encoder-parity witness did
not exercise copies: extend it with a nonlegendary exception and a two-keyword
exception, asserting the copy's object/source joins, indefinite duration and
timestamp rank alongside native/Python equality. This is an injected boundary
test, complementing the native copy rules witnesses rather than claiming a
second rules proof. CI previously omitted this module: the integration workflow
now selects this single bounded test by node id. Runtime code, extension inputs
and retained evidence are unchanged.

Local results (current tree, no timeout or deferred local check):

- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 --lib agent::continuation` — 5 passed. Its first invocation's completion was not captured; the immediate cached rerun above supplies the result.
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 --test rules_tests cr_613_continuous_effects` — 6 passed, debug assertions enabled.
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --manifest-path managym/Cargo.toml -j 2 --all-targets --all-features -- -D warnings` — passed.
- `cargo fmt --manifest-path managym/Cargo.toml --check` — passed.
- `uv run --extra dev --extra notebook --extra artifacts --extra reports pytest tests/env/test_observation.py tests/verify/test_observation_parity.py tests/verify/test_continuation_input.py tests/semantic/test_current_content.py tests/env/test_skip_forced_decisions.py tests/academy/test_experience_protocol.py tests/academy/test_choice_navigation.py tests/academy/test_trigger_order.py tests/academy/test_learn_setup.py tests/belief/test_state.py tests/semantic/test_elemental_world.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py tests/model/test_choice_separability.py tests/training/test_active_training.py -q --durations=10` — 218 passed in 102.85s, including the expanded copy-fact parity witness. Longest case: ordinary native/Python observation parity, 28.47s; no timeout.
- `npm --prefix academy/ui test -- --run src/lib/permanent-notes.test.ts src/lib/decision-labels.test.ts src/lib/protocol-conformance.test.ts` — 14 passed.
- `npm --prefix academy/ui run check` — 0 errors, 0 warnings (Node emitted a dependency deprecation notice).
- `uv run --extra dev ruff check manabot/env/observation.py manabot/managym/__init__.pyi manabot/managym/experience.py manabot/managym/experience_projection.py manabot/semantic/text_observation.py tests/belief/test_state.py tests/semantic/test_text_observation.py tests/verify/test_observation_parity.py` — passed.
- `uv run --extra dev ruff format --check` with those same eight paths — passed after formatting only the expanded parity test.
- `uv run --extra dev pytest tests/infra/test_ci_selection.py -q` — 40 passed after the workflow selection edit.
- `uv run --extra dev pytest --collect-only -q tests/env/test_observation.py tests/verify/test_observation_parity.py::test_v5_encodes_every_observed_permanent_and_card_field_in_parity` — 55 collected, including the new CI selection alongside its observation consumer suite.

The same-head demo retains 51 observability, 13 T5 card and 47 text/choice
consumer passes in `scratch/t5-demo.md`.

Final CI result: [run 41541743395](https://github.com/loopflowstudio/manabot/actions/runs/41541743395)
**passed at `1de85f7aa`**, including every selected job and Tests Result. Debug
Rust includes 501 card passes and 141 rules passes (2 existing ignored rules
tests). Academy's built, headless release browser suite passed 25 and skipped
2 cases requiring a local card-art cache or retained checkpoint; these skips
are not claimed as passes. All 22 frontend unit-test files and Svelte checking
passed. Both integration groups, evaluation Perf proof, training smoke,
protocol, semantic conformance and clean-machine play passed. Release package
and install checks also passed; package publication was skipped.

The main integration stage took 10m10s and its evaluation stage 9m04s. Both
completed successfully without a timeout; their slower duration is not a
controlled performance comparison. The preceding heads' cancelled CI runs
remain historical, not substitute acceptance.

`git diff --check` and `lf context` passed; scratch and memory stay within their
configured limits. No server or watcher was started. All local test/build
commands have exited. This gate added only the parity witness, its CI node
selection and these notes; the existing uncommitted demo evidence was preserved.
The new test/workflow changes are locally verified, not part of the cited CI
head. No commit, push, merge, auto-merge, Task completion or Wave-memory edit
was performed. PR #405 remains open with `autoMergeRequest: null`, awaiting
Jack Heart's review.
