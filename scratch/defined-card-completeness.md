# Defined cards and completeness — 2026-11-03

## Current direction and recovery

Jack Heart clarified the intended policy: if a card was already deemed good
enough to play, known bugs should be categorized and tracked in Tasks rather
than blocking it from play. This supersedes the assistant's proposal below to
require complete behavior before permitting any production definition.
Jack then directed: "Anyway let's fix these guys and move on." Finish the five
fixes and validation; do not turn this into a broader policy redesign or create
additional Tasks merely from this discussion.
MTG-239's existing implementation Graph remains live. Its five fixes are not
yet a verified delivery. Do not duplicate its implementation.

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
- `docs/rules/tla/README.md` assigns these five missing behaviors to T2/T3
  (Rallier crosses both). Those batch commits are present at the Task base,
  yet all five entries remain in `tla::INCOMPLETE`. Other entries were removed
  as later batches delivered behavior. The evidence proves the obligations
  survived those deliveries; it does not establish the exact review decision
  or whether they were consciously deferred versus missed.
- The runtime contract institutionalized the distinction: `world.rs` says
  incomplete definitions still run in fixed setups; `cards/pool.rs` marks
  their descriptor rows Unsupported; deck admission rejects those rows.
  AGENTS.md explicitly permits the exception to “support is definition.”

The practical purpose was continuity for old curated setups plus rejection
of new imported decks. Jack Heart rejects that inconsistent availability:
known defects in previously playable cards should be tracked as bugs.
Agreement across consumers and green tests of existing behavior do not prove
fidelity to card rules; playability does not imply absence of known bugs.

## Superseded assistant proposal

After completing these five behaviors, remove the incomplete-definition
exception and its fixed-setup privilege from the current content contract.
Define a production card only with its required rules implemented; keep
unfinished work on its implementation branch and preserve historical worlds
at their recorded revisions. Unknown cards remain rejected and tokens/fixtures
retain their distinct deck-entry rules. Do not remove admission protection
ahead of fixing the underlying behavior.

No type system can prove a card has no bugs. Enforcement means no intentional
partial-card category, behavior tests against the actual obligations, and
review of the remaining obligations before calling a card delivered. The
specific protocol/code cleanup was a proposal, not accepted direction. The
latest user direction above supersedes this proposed strict admission policy.
