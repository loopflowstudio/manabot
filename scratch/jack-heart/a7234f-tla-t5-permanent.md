# T5 (a7234f): permanent copying — plan and status

As of 2026-11-02, local base `bbb5101eb`, PR #405 was last recorded at
`e8fd88eed`; main `29660d00c` is merged. Joo Dee, One of Many, Ran and Shaw,
and Phoenix Fleet Airship are defined and supported. Implementation and the
native Command / LLM text demos are complete. Gate owns current-head checks,
publication and the PR description refresh below.

**Shipping is stopped.** Jack Heart's latest steer f39fef34 says to run the
Graph through gate, then stop for review: no merge. The Task's recorded
2026-11-02T21:35:52Z auto-ship approval and “running from land --wait-and-fix”
Graph label do not override this explicit direction.

Prior review narratives and detailed check history:
`bbb5101eb:scratch/jack-heart/a7234f-tla-t5-permanent.md`.
The uncommitted implement note at compression entry confirmed no source
changes since `924e520b4`; its corrected characteristic authority and completed
deletion list are preserved here. Its original diff is also in
`.lf/tmp/t5-compress-input.patch`.

## Accepted design

Jack Heart, 2026-10-30, comment 10a31948: stack on T3 #399 and reuse its
sorcery timing; observe copies and general exceptions through composable
modified-characteristic facts and tokens; share sacrifice history with T4;
define Phoenix Fleet Airship. All four are implemented. T3's continuous-effect
records and v5 fact rows are the substrate
(`08bf54de2:scratch/6875f0-tla-t3-characteristics.md`).

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
  subsequent 1a/2a CI failure and its repaired fixture are detailed below.

## Remaining

- Gate at the head `e8fd88eed` (1a/2a implemented and compressed, main
  `29660d00c` merged). CI runs 41536286824 / 41536286630
  passed every job except Academy UI, whose two live retained
  choice-navigation cases (CHOOSE_TARGET, DISCARD) found the fixture's
  projection stale: `base_power` / `base_toughness` are the card row's P/T,
  layered since 2a. Realign re-recorded
  `academy/fixtures/choice-navigation-w17.json` (only those values, match
  ids, frame hashes and the source commit moved; every state digest and
  offer order is preserved) after fixing
  `scratch/tools/regen_choice_navigation.py` to keep a recorded offer id
  when its label repeats (the first-match rule had silently moved
  CHOOSE_TARGET to another copy of It'll Quench Ya!, changing its digest).
  The kernel, authored-match, demo and trigger-order receipts did not move.
  Gate at the new head: CI (the Playwright spec is the proof of the
  re-record), the local debug `cargo test`, clippy and the focused Python
  suites with the rebuilt extension.
- Open for Jack Heart: the projection's `base_power` / `base_toughness`
  were documented as printed values for a "4/4 (2/2)" display; with the
  card row layered they equal `power` / `toughness` on the battlefield, and
  printed P/T has no other Python-visible source (the pack manifest carries
  ids and names only). No UI component reads them today. Choices: (i) keep
  them as the card row's value (current; the comment now says so), (ii)
  add a printed-P/T accessor to the engine and keep the field printed, or
  (iii) drop the fields (protocol change). (ii) and (iii) are follow-up
  Tasks, not T5 blockers.
- PR #405's body predates 1a/2a: it says the legend rule reads
  `Game::permanent_is_legendary` (deleted) and does not state the layered
  `CardData` rule or `Characteristics.is_legendary`. Gate or pr-publish
  refreshes those sentences; the "314 cards" and copy-model paragraphs are
  current.
- Then Jack Heart's review of PR #405. No merge: steer f39fef34 is the latest
  direction, and the Task record's auto-ship approval (21:35:52Z) does not
  override it.
- Known, not fixed: `tests/sim/test_rollout.py::TestBatchedSampler::test_selects_valid_actions_only`
  fills every float buffer with noise, which the fact and definition checks
  reject. CI does not select the file.

## Delete — do not maintain

- Removed spent T3 rewrite scripts `scratch/tools/convert_continuous.py` and
  `scratch/tools/rust_followups.py`: the surviving engine already owns their
  migrations. Removed `scratch/add_fixture.py` (its scratch input is absent;
  Scryfall fixtures are maintained by the existing card tooling).
- Removed `scratch/tools/probe_t5.py` and `scratch/tools/demo_t5_contract.py`:
  probes for the repaired capture failure; the real text-path demos and
  `test_joo_dee_surveil_and_copy_play_through_the_text_decision` survive.
  Original scripts remain at `bbb5101eb:scratch/`; no live callers.

- `Game::permanent_is_legendary` and battlefield `CardData` reads of printed
  mutable characteristics: replaced by `Characteristics` and the exhaustive
  layered/printed field guard. Done in `16e28c304`, compressed in `924e520b4`.
- `Effect::CreateNonlegendaryCopy` (main) and `CreateTokenCopy { nonlegendary }`:
  replaced by `{ exceptions }`. Done.
- The branch's duplicate `ActivatedAbilityDefinition.sorcery_timing`: T3's is
  the one field. Done.
- The tensor-only "copy identity" model: superseded by `CopyEffect` and its
  fact rows. Done.
- T5's own `PlayerTurnHistory` and `StaticCondition::SacrificedThisTurnAtLeast`:
  T4's. Done at the sync.

No remaining deletion targets in the accepted design.

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
`CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 --test cards tla::`: 390 passed; `uv run --extra dev pytest tests/semantic/test_text_observation.py::test_card_line_uses_the_observed_flag_without_rederiving_it_from_effects -q`: 1 passed; format/lint and `git diff --check` clean. Gate retains current-head CI and acceptance checks.
