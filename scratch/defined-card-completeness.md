# Defined cards and completeness — 2026-11-03

## Current direction and recovery

Jack Heart clarified the intended policy: if a card was already deemed good
enough to play, known bugs should be categorized and tracked in Tasks rather
than blocking it from play. This supersedes the assistant's proposal below to
require complete behavior before permitting any production definition.
Jack then directed: "Anyway let's fix these guys and move on." Finish the five
fixes and validation; do not turn this into a broader policy redesign or create
additional Tasks merely from this discussion.
MTG-239's five fixes are implemented and integrated with MTG-238 at
`6d30c9013`; full gate and delivery remain open. Current implementation and
remaining checks live in [the plan](mtg-239-complete-five-defined.md).

## Evidence and causal chain

- Earlier curated-matchup implementations knowingly approximated rules.
  `fa5e13b23^:managym/src/cardsets/tla.rs:302` explicitly says Accumulate
  Wisdom randomizes the remainder because no reorder decision exists. The
  same file's Suki definition documents other then-existing shortcuts. The
  relevant early commits are `d0705c49b` and `bbcd730a2`; their commit account
  metadata does not establish who made or authorized these choices.
- Tests could preserve the approximation: at Task base `22e4aab71`,
  `managym/tests/cards/tla.rs:216` accepts either remainder order and expects
  priority immediately after selection. It does not require the player's
  ordering choice. Passing this test cannot establish that rule's correctness.
- The October 29 one-definition migration deliberately retained 14
  setup-used incomplete definitions and excluded them only from arbitrary
  deck admission. See `fa5e13b23:scratch/questions.md` and the opening of
  `fa5e13b23:managym/src/cardsets/tla.rs`. The migration preserved existing
  setups while exposing known gaps; it did not originate every gap.
- At Task base `22e4aab71`, `docs/rules/tla/README.md` assigns these five
  missing behaviors to T2/T3 (Rallier crosses both). Those batch commits are present at the Task base,
  yet all five entries remain in `tla::INCOMPLETE`. Other entries were removed
  as later batches delivered behavior. The evidence proves the obligations
  survived those deliveries; it does not establish the exact review decision
  or whether they were consciously deferred versus missed.
- The Task-base runtime contract institutionalized the distinction:
  `world.rs` says incomplete definitions still run in fixed setups; `cards/pool.rs` marks
  their descriptor rows Unsupported; deck admission rejects those rows.
  AGENTS.md explicitly permits the exception to “support is definition.”

The practical purpose was continuity for old curated setups plus rejection
of new imported decks. Jack Heart rejects that inconsistent availability:
known defects in previously playable cards should be tracked as bugs.
Agreement across consumers and green tests of existing behavior do not prove
fidelity to card rules; playability does not imply absence of known bugs.

## Superseded assistant proposal

The assistant proposed requiring complete rules implementation before any
production definition could be played and removing the fixed-setup exception.
Jack Heart rejected that admission policy on November 3. The full proposal is
retained at `60b8071f8:scratch/defined-card-completeness.md`; it is not remaining
work. These five rules fixes and their whole-deck proof remain required.
