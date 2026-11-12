# MTG-320: runtime object kinds by zone

Design: `docs/engine.md#objects-and-zones` (CR term -> type table, zones, token moment, "What is
built"); `managym/README.md` links it. Assumptions and naming exclusions: `questions.md`.

## Built (HEAD 2a3eae3b0, synced with main a343904ee and MTG-298 15e50b205)
- Kind: `ObjectKind { Card, Token, SpellCopy }`, private `kind` on the runtime record, set at
  construction (`with_kind` for token copies and spell copies); `resolve_into_token()` in
  `commit_zone_move` when a spell copy enters the battlefield (CR 111.12). Runtime readers use the
  kind: SBA (CR 704.5d-e), predicate/layer `token` filter, exile cast grants, "card" selection,
  return-to-battlefield guard, main's `PutIntoGraveyard` (`moved.is_card()`). `become_copy_of`
  leaves kind alone. Remaining `is_token` reads are catalog rows, deck/sideboard admission, the
  observation column (`!is_card()`) and hash wire bytes.
- Rename (steers d57fc17e, 0779d3e9, e3b2a142, dbf13009): record `Card` -> `Object` in
  `state::game_object`, `CardId` -> `ObjectId`, `CardVec` -> `ObjectVec`, `GameState::objects`,
  `object_to_permanent`, old u32 `ObjectId` -> `PublicId`; object-level helpers `remove_object`,
  `object_zones`, `matches_object[_with_power]`, `object_power`, `object_mana_value`. Serde keeps
  wire names. Incoming MTG-298 tests ported in 2a3eae3b0.
- Byte identity: observation `is_token` = `!is_card()`; hash keeps `is_token: true` for non-card
  objects. `semantic-kernel-w17-v2` conformance reproduced after the rename (one serde-attribute
  drift on `ObjectLki.counters` found by diffing traces and fixed).
- Test: Azula instant copy is `SpellCopy` and ceases off the stack.

## Checks seen (before the MTG-298 15e50b205 merge and 2a3eae3b0)
- Debug `cargo nextest --workspace --profile ci`: 1166 passed; 4 timed out at load ~250-380 and
  pass alone (conformance 508 s; base revision 345 s at the same load). Clippy `-D warnings` clean.
- pytest env/belief/semantic/verify/academy + world-change list: 907 passed, 1 failed
  (`test_v10_vector_buffers_reach_the_model`, "policy forward requires an observation from
  admit_observation"; no Python differs from main here, so inherited), 1 deselected (5-min limit).
- After the rename's compress: `cargo build --tests --workspace` ok.
- QA (release extension): 60 random games, 150 Azula games, 4 Academy games; no token or copy seen
  off battlefield/stack at a decision except Treasures mid mana choice (CR 704.3, correct).

## Concept review direction (2026-11-12): built
Object names for `move_object_with_entry`, `capture_object_move`, `subject_matches_moved_object`
and the `assert_stack_consistent` locals; stale `move_card` mentions and the object-path doc
comments fixed; `docs/engine.md#objects` now gives the built meaning of `ObjectId` first and puts
the per-zone-change model under "Planned split". Card-named hand/library/graveyard/exile helpers
left on purpose. PR body drafted in `scratch/pr-491-body.md` (library/hand hold `ObjectId`s; steer
dbf13009 said "Card-only zones (library, hand, decklists) hold card ids", and the body explains
the difference: CR 111.7-111.8, 704.5d, byte identity).
Check: fmt, `cargo build --tests --workspace`, clippy `-D warnings` clean; debug nextest
cr_400/405/601/603/704 + scenario 30 passed, cards token/azula 41 passed.
Compress: `assert_stack_consistent` compares sorted object lists instead of a count map; clippy
clean, debug nextest stack/copy/counter/cr_6 160 passed.

For the later split Task, not here: whether library and hand become card-id lists or stay object
lists as now.

## Remaining
- CI run 43325427802 on 2a3eae3b0: all jobs passed (Rust Tests debug, Engine Budget, both
  conformance jobs, Python unit, Academy, Integration long/evaluation) except Integration core,
  still in progress when read again at the compress step. Commits after 2a3eae3b0 need their own run.
- PR #491's published body is stale; replace it with `scratch/pr-491-body.md` at publish, updating
  its CI line.
- Stack parent PR #476 (MTG-298) is open; it lands first.
- Left for later (steer allows "land the rename plus the typed ids here, and say what is left in
  the PR body"): separate `Card`/`CardId` store, new object per zone change, the CR 400.7
  carry-over record, measuring their layer/`TriggerIndex` invalidation cost.
