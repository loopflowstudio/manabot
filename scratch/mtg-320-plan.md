# MTG-320: runtime object kinds by zone

## Outcome and constraints

Runtime names and kinds follow the rules: tokens and spell copies are objects, never
cards. Keep observations, Python/PyO3 surfaces, serialized names and frozen evidence
byte-identical relative to MTG-298. Design: `docs/engine.md#objects-and-zones`.

The full design separates the physical card from each appearance in a zone. Steer
`dbf13009` permits staging it: "If the full split is too big for one PR, land the
rename plus the typed ids here, and say what is left in the PR body." The same steer
requires flat stores, a tiny Copy card record, an inline CR 400.7 carry-over record,
checked object-to-card conversion, and measurements of cache invalidation. Those
requirements remain for the split; this PR does not implement it.

## Built

At a45df9d4b:
- `state::game_object::ObjectKind { Card, Token, SpellCopy }` is private runtime
  state. `resolve_into_token()` changes a permanent spell copy at battlefield entry;
  `become_copy_of` preserves kind. SBA, token predicates, exile grants, card selection,
  return guards and `TriggerCondition::PutIntoGraveyard` read kind.
- Runtime `Card` -> `Object`, `CardId` -> `ObjectId`, `CardVec` -> `ObjectVec`,
  `GameState::cards` -> `objects`, `card_to_permanent` -> `object_to_permanent`;
  old public u32 `ObjectId` -> `PublicId`. Serde keeps wire spelling. Observation
  `is_token` and hash bytes still encode non-card objects as before. Token predicates
  now exclude spell copies; no defined card filters spells by token status.
- `ObjectId` remains a persistent storage row; `ObjectRef` (`EntityId`, `Incarnation`)
  names the appearance and goes stale. `ObjectLkiLog` retains departed facts.
- Every `ZoneManager` list holds `ObjectId`, including library and hand, so moved
  tokens remain there until SBA. Decklists and sideboards alone are card-only.
- Concept-review cleanup is done in 5dc528319: `move_object_with_entry`,
  `capture_object_move`, `subject_matches_moved_object`, stack locals, object-path
  comments and `move_object` accessor mentions. CLAUDE.md links AGENTS.md.
  `docs/engine.md#objects` presents the built identity first and labels the split
  planned. Card-named selection helpers and published observation names stay.
- d608ff5ba simplified `assert_stack_consistent` to compare sorted object lists.
  The a45df9d4b parent sync adds only scratch notes and QA artifacts; no runtime diff.
- Realign fixed README's stale `card_matches_predicate` reference to the actual
  `flow/statics.rs::object_matches_predicate`. No runtime changes in this pass.

The supplied iteration feedback predates 5dc528319 and d608ff5ba. Its code/doc work
is complete; its CI check and PR-body draft are accounted for below.

## Validation evidence

- CI 43325427802 on 2a3eae3b0: all 13 jobs passed, including debug Rust, both
  conformance jobs, Engine Budget, Clean-machine Play, Python Unit, Academy and
  Integration core/long/evaluation. Read on 2026-11-12; core is no longer pending.
- Same-run Engine Budget versus rebuilt 7ef6ce588: search total 173.303 vs 173.206 s,
  worst reply 5.102 vs 5.123 s; steps 12,174.4/s vs 12,398.8/s. Search/decision
  digests matched and the 10% budget passed. Source: run 43325427802, job
  131556645087. These numbers precede the helper cleanup.
- After 5dc528319: fmt, workspace test build and clippy `-D warnings` passed;
  focused debug zone/stack/trigger/SBA/scenario tests 30 passed, token/Azula 41 passed.
  After d608ff5ba: clippy passed, focused debug stack/copy/counter/cr_6 tests 160 passed.
- Earlier, before parent 15e50b205: debug nextest 1166 passed; four tests hit the
  limit under observed host load and passed alone (conformance 508 s, base revision
  345 s at similar load). These timings do not justify relaxing the five-minute limit.
- Earlier Python: 907 passed, 1 failed (`test_v10_vector_buffers_reach_the_model`,
  "policy forward requires an observation from admit_observation"), 1 deselected
  for the time limit. Python matched main then, but the claim that the failure was
  inherited was an inference, not an isolated cause reproduction. Later CI passed.
- Release QA: 60 random games, 150 Azula games and four Academy games finished.
  Treasures persisted in the graveyard during a mana choice, consistent with SBA
  waiting for priority. Academy permanent-spell-copy and spell-copy-to-hand paths
  were not reached. Semantic-kernel-w17-v2 reproduced after fixing a serde drift
  on `ObjectLki.counters`; no evidence regenerated for this rename.

## Remaining for this PR

- Confirm final-head checks. Release packages 43330293430 and CI 43330293531 on
  a45df9d4b were in progress when read on 2026-11-12; the older green run does not
  cover helper cleanup.
- Publish the corrected body from `scratch/pr-491-body.md`, preserving the existing
  task context block and refreshing CI status. PR #491's live body still says the
  runtime record is Card and the main readers have not moved; both are obsolete.
- Parent #476 remains open at 4bc48211e and must land first. Delivery stops at review;
  the task steer says "Stop at review. Do not enable auto-merge."
- Main advanced from a343904ee to 85ed9214e (CUDA layout defaults): the compare
  lists training/evaluation/docs/tests and `.lf/skills/qa.md`, no engine or evidence
  changes. No change to the object approach follows from it. Reconcile any required
  upstream sync through the delivery workflow; this pass did not merge it.

## Concept review, second pass (2026-11-12, a45df9d4b plus the README fix)

No design change for this PR. The remaining `card`-named helpers that return
`ObjectId` (`library_top_card`, `revealed_cards`, `foretold_cards`,
`exile_granted_cards`, `selection_zone_cards`, `priority_candidate_cards`,
`unseen_cards`, `pinned_library_cards`) read library, hand, graveyard or exile, where
the rules mean cards; the agent-side `card`/`cards`/`card_reference` writers are
published observation names. The one open choice below belongs to the split and does
not change what this PR builds.

Optional wording nit, not worth an iteration alone: `state/src/game_object.rs`
describes `EntityId` and `ObjectRef` in terms of "physical card" where a token
qualifies too.

## Later split and unresolved choice

Still required: a separate tiny Copy `Card`/`CardId` store; a fresh object row per
zone change; an inline carry-over record citing CR 400.7 sub-rules; and measurements
of layer-view/TriggerIndex invalidation cost, with Engine Budget passing. The current
incarnation/LKI machinery supplies the starting point, not the completed split.

Steer dbf13009 says "Card-only zones (library, hand, decklists) hold card ids."
Current library and hand retain ObjectIds because moved tokens occupy those zones
until SBA (CR 111.7-111.8, 704.5d), and byte identity forbids removing them early.
For the split, resolve whether those zones keep object lists or use card-id lists
with an explicit representation of transient non-card occupants. Neither alternative
is a settled product decision. Preserve this conflict for review/demo; the staged
rename can be reviewed with the current behavior documented.
