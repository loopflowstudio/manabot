# T5 (a7234f): permanent copying — plan and status

Status (2026-11-02, PR #405 open, no review yet, head `e8fd88eed`, main
`29660d00c` merged, 0 behind): implemented, demoed through native Commands
and the LLM/Academy text path, synced onto main (T3's squash, the
characteristics cache and T4). All three cards are defined and supported.
Jack Heart's concept-review decisions 1a and 2a (below) are implemented
(`16e28c304`) and compressed (`924e520b4`); CI on the head is the gate's
pending check (Remaining). Shipping is stopped per Jack Heart's latest steer
f39fef34 ("run the graph through gate. Shipping stays stopped: no merge").
The Task record shows "Auto-Ship approved by Session … for Jack Heart
2026-11-02T21:35:52Z" and the Graph "running from land --wait-and-fix"; the
steer is the later, explicit direction, so the PR waits for Jack Heart's
review and nothing merges.

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
Nothing to change; no product choice is open. Re-run at `39dd86190` after the
realign: no code or main change since, same outcome; the page's header and CI
line were refreshed.

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

## Concept review with Jack Heart — 2026-11-02 (at `2df025c20`, open)

Page: `scratch/concept-review.html`. The headless reviews found nothing to
change. This one checks the accepted decision 2 ("always see a permanent
having modified abilities") against how the code reads one characteristic.

- **Mismatch.** T3's `Game::characteristics()` (`flow/statics.rs:265`) is the
  layered view that the rules and the observation share. Name, subtypes,
  creature, animated, keywords, P/T and abilities are all in it. Legendary is
  not. Two readers decide it on their own:
  `Game::permanent_is_legendary` (`flow/sba.rs:16`) reads copiable supertypes
  minus a `Nonlegendary` record, and the legend rule uses it.
  `CardData.is_legendary` (`agent/observation.rs:1160`) reads printed
  supertypes only, and the tensor column 39, the text "legendary yes/no" and
  the tokens' `is_legendary` all use it.
- **Why it's consistent today.** The only source of `Nonlegendary` is a copy
  exception, which `create_token_copy` writes into the token's definition, so
  the two readers agree. Their answers split when a record (not a copy
  exception) says "isn't legendary": the legend rule spares the permanent, but
  the policy and the text still say legendary. The `Modification` vocabulary
  already allows this, since `Nonlegendary` is a layer-4 record.
- **Proposed (not accepted).** Add `is_legendary` to `Characteristics`,
  computed in layer 4 from copiable supertypes and records. The legend rule
  reads `characteristics(id).is_legendary`, and `permanent_is_legendary` goes
  away. `CardData.is_legendary` reads characteristics on the battlefield, the
  same way `is_ally` already does, and the printed value elsewhere. No column,
  token or text change. Proof: a directed test that applies a `Nonlegendary`
  record outside a copy and checks the legend rule, column 39 and the text
  line together.
- **Not in question.** Writing copiable values into the definition at
  creation is the rule (token copies don't track their source later), and
  `Card.copy` records its inputs for observation.
  `PlayerTurnHistory` has one owner.

**Decided by Jack Heart (2026-11-02): 1a, 2a.**

1a. Do it in T5, before merge. `Characteristics.is_legendary` (layer 4:
copiable supertypes unless a `Nonlegendary` record removes it) becomes the
one authority. Delete `permanent_is_legendary`; the legend rule reads
`characteristics(id).is_legendary`.

2a. Make it a rule: every `CardData` field that states a characteristic of a
battlefield card reads `Game::characteristics()`. Off the battlefield, fields
keep reading the printed card. `tests/cards/observability.rs` enforces this.
For each `CardData` field it records whether the field is layered (and reads
characteristics on the battlefield) or printed by definition (identity, token,
costs paid when casting), so a new field must be classified before it
compiles. Apply the rule to every current field, including subtypes such as
Lesson and abilities such as ward. When a characteristic has no layered value
yet, add it to `Characteristics` instead of reading the printed card.

**Implemented (2026-11-02, headless, after the decisions):**

- `Characteristics` (`flow/statics.rs`) gains `is_legendary` (layer 4:
  copiable supertypes unless a `Nonlegendary` record), `types: CardTypes`
  (layer 4: printed, static type grants, Creature when animated, Artifact for
  an artifact-creature record; `is_creature()` / `is_land()` are methods
  reading it, not separate fields) and
  `ward` (printed ward unless abilities are lost). `permanent_is_legendary`
  is deleted; `legend_groups` reads the view.
- `Observation::card_data` reads the view for every characteristic field of a
  battlefield card: name, power/toughness, legendary, Ally, Lesson, ward,
  card types and keywords (`card_data` overrides `printed_card_data` with
  struct update, so the split is visible in one place; `Card::is_legendary`
  is the one printed-supertype reader). Printed by definition: identity (zone, owner, id,
  registry key), token status, text box, mana cost, kicker, face-down,
  cast permission and X. `CardTypeData::from_types` is the one builder.
- `tests/cards/observability.rs`: `card_sources()` classifies every
  `CardData` field as `Layered` or `Printed(reason)` through an exhaustive
  destructure (a new field does not compile unclassified);
  `layered_card_fields_read_characteristics_on_the_battlefield` applies
  records (Nonlegendary, +1/+1, flying, Lesson, artifact creature, ability
  loss, Ally) and Honest Work, then requires every Layered field to move
  and no Printed field to;
  `card_rows_off_the_battlefield_read_the_printed_card`;
  `a_nonlegendary_record_clears_the_legend_rule_and_card_column_39_together`.
  `Modification::Nonlegendary` is now `Seen::Card(39)`. The Toph granted-land
  test no longer claims v4 is blind to the grant: card column 11 (land) shows
  it in every version, v5 adds the granted-permanent column.
- Python: the text's card line says "keywords" instead of "printed
  keywords" (its battlefield flags are layered now);
  `test_a_nonlegendary_record_reads_as_not_legendary_on_the_card_line` (a
  renderer check: the engine-side proof is the Rust test above);
  the `Card` stub, `managym/README.md` and `WORLDS.md` state the rule. No
  column, width, token or protocol change; the Python encoder reads the same
  rows, so parity holds without a change.
- The Yip Yip! card test and `PermanentData.keywords`' doc had stated the
  old contract (card entry keeps printed keywords); both now state the
  layered one.
- Evidence: a battlefield card row now equals its permanent's layered
  values (P/T with counters and pumps, granted keywords and types). Implement
  expected recorded observations with active effects to move; on
  `e8fd88eed` only the choice-navigation fixture's projection did
  (earthbent lands, anthem creatures), and realign re-recorded it
  (Remaining). The kernel, authored-match, demo and trigger-order receipts
  held.

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

- `Effect::CreateNonlegendaryCopy` (main) and `CreateTokenCopy { nonlegendary }`:
  replaced by `{ exceptions }`. Done.
- The branch's duplicate `ActivatedAbilityDefinition.sorcery_timing`: T3's is
  the one field. Done.
- The tensor-only "copy identity" model: superseded by `CopyEffect` and its
  fact rows. Done.
- T5's own `PlayerTurnHistory` and `StaticCondition::SacrificedThisTurnAtLeast`:
  T4's. Done at the sync.

## Checks

- 2026-11-02 (realign after 1a/2a, headless): CI on `e8fd88eed` passed all
  jobs but Academy UI (two choice-navigation projection cases). Extension
  rebuilt at HEAD; `uv run python scratch/tools/regen_choice_navigation.py`
  then `uv run --extra dev pytest tests/academy/test_choice_navigation.py -q`:
  5 failed before the re-record, 7 passed after. The Playwright proof is
  CI's at the new head.
- 2026-11-02 (compress, headless): debug `cargo test --test cards --test
  rules_tests` (501 + 141 passed), clippy `--all-targets --all-features -D
  warnings` clean. Observation output is unchanged, so no extension rebuild
  or Python rerun; gate owns those.
- 2026-11-02 (implement 1a/2a, headless): debug `cargo test --test cards
  observability` (51), `--test cards tla::` (390) and `stx::` (19),
  `--test rules_tests` legend / engine_events / cr_613 (24),
  `search_state_contract` and `possible_worlds_tests` passed; `cargo fmt
  --check` clean. Extension rebuilt; `uv run --extra dev pytest
  tests/semantic/test_text_observation.py tests/verify/test_observation_parity.py
  tests/academy/test_recorded_attempts.py tests/semantic/test_current_content.py
  tests/env/test_skip_forced_decisions.py -q`: 75 passed. Clippy, the full
  debug `cargo test`, conformance receipts and the evidence re-records are
  gate's.
- 2026-11-02 (gate at the green head `836042c07`, headless): the diff
  against main carries no leftovers (no debug output, TODOs or transient
  comments). The sync had duplicated three lines, which this gate removed:
  `Player.turn_history` in `manabot/managym/__init__.pyi`, the
  `turn_history` spread in `experience_projection._serialize_player` (the
  second entry overwrote the first, so serialized output is unchanged) and
  the `turn_history` label key in `tests/semantic/test_text_observation.py`.
  Rust and TypeScript reject such duplicates at compile time, and clippy and
  svelte-check pass in CI, so only Python carried them. The engine reports
  314 supported definitions in `world` (`set_cards` over every set file: 257
  TLA eligible, 4 TLA unsupported, 1 `ths` unsupported, 57 others), not the
  266 the branch had stated (main's 263 was stale before T5, below), so
  `WORLDS.md` and the PR body now say 314; the PR body's turn-history
  bullets now claim only T5's own change (the projection, Python
  construction and singular wording), since the field, column and
  `Condition` are main's. `WORLDS.md` no longer lists "a static condition
  on sacrifices this turn" as a T5 addition. Checks: `ruff check` and
  `ruff format --check` on the three Python files; `uv run --extra dev
  pytest tests/semantic/test_text_observation.py
  tests/verify/test_observation_parity.py
  tests/academy/test_recorded_attempts.py
  tests/semantic/test_current_content.py -q` (70 passed, 53 s). No Rust,
  fixture or frontend source changed, so the suites recorded below keep
  their results.
- 2026-11-02 (re-gate after the sync, headless): rebuilt the extension from
  the synced tree. The authored-match authority receipt regenerated
  byte-identical (`081bbbe7…`), so its SHA pins and `TERMINAL_STATE` stay.
  `python -m academy.authored_match_parity write` moved only `relevant_source`
  hashes; `public_commitment_parity verify` passed (142 commands, 69
  commitments, 0 gaps). Re-recorded `trigger-order-w17.json` (content digest
  `92b2f506…`, root `7073b445…`) from `manabot.verify.trigger_order.root` and
  the precision demos with `scratch/tools/regen_play_demos.py` (revisions 30
  and 66 unchanged; the manifest gains the three cards in name order).
  Release Playwright (`npm run test:e2e:release`, ports 8031/5191): 25 passed,
  2 skipped (deck-browser art cache and checkpoint cases, environment-gated),
  including the three decision-surface Precision cases CI had failed. The
  receipt suites (authored-match authority and parity, choice navigation,
  trigger order, Learn evidence and setup, decision surface, belief state and
  learning, continuation input): 72 passed.
- 2026-11-02 (realign after the concept review, HEAD `9bf9c1928`): `gh run
  view 41507203990 --json jobs` on `1e9fd2d7c`: Rust, Python unit, both
  conformance, Clean-machine Play, Academy Integration and Integration (long)
  passed; Academy UI and Integration (main) failed on the stale digests named
  under Remaining. Main is unchanged since the sync (`origin/main` =
  `4c8324a59`); no rerun of local suites.
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
