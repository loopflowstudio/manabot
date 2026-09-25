# Learn/Lesson decisions and evidence

Accepted design: 2026-09-24. Working-interaction approval: 2026-09-25.
Implementation remains unfinished in
[Complete Learn/Lesson in the selected matchup · ETU-75](https://linear.app/loopflow/issue/ETU-75/complete-learnlesson-in-the-selected-matchup).
This is the bounded Rules change following
[PR #185](https://github.com/loopflowstudio/etude/pull/185),
not a restart of ETU-14/31/55 or evidence of completed replay certification.

## Accepted setup and interaction

The human chose formal sideboards: “We just need to give decks formal
sideboards.” The proposed lists were “totally fine for now,” and “Sideboards
should be open-decklist, as are the matches.” Keep the shipped main decks:
UR has 41 cards and GW has 40.

| Deck | Additional sideboard copies |
|---|---|
| UR Lessons | One each Firebending Lesson, It'll Quench Ya!, Accumulate Wisdom |
| GW Allies | One each Yip Yip!, Fancy Footwork |

These are accepted product lists, not a claim of Constructed legality. The
extra Firebending Lesson is the fifth combined copy. GW currently has no Learn
spell; its sideboard establishes symmetric setup and ownership. Between-game
sideboarding, hidden custom sideboards and a general format framework are out
of scope. Supported non-Lessons may be admitted to a sideboard but cannot be
retrieved by Learn. Invalid names, tokens and counts fail admission.

The human preferred “Probably first choice then select a card” and approved the
design with “approve design.” Present Take a Lesson, Discard and draw, or
Decline Learn, then a card selector with previews and Back for the first two.
The engine offers every complete atomic choice. Local navigation commits
nothing; selecting a card submits its original offer once. Empty modes explain
their absence, and reconnect or a changed decision resets the selector.

## Rules and information invariants

- Compiled semantic setup owns deck and sideboard lists. Named setup, seat
  reversal, scalar/vector reset and recorded roots carry both. Resolve within
  the pack: Jeong shares a deck name but has its own explicit empty sideboards.
- Sideboard copies are ordinary owned cards instantiated once outside all game
  zones. The immutable roster plus absence from zones determines availability.
  Retrieval reveals the definition, then uses ordinary movement from no zone
  to Hand. It neither allocates a copy nor draws, shuffles or casts. A consumed
  copy cannot be retrieved again from hand, graveyard or exile.
- Both players know setup and remaining definition counts. Only the owner sees
  outside candidate identities. Opponent hands and uncommitted candidates stay
  private; outside candidates are not fabricated in-game ObjectRefs.
- Public hand knowledge is a minimum multiset of definitions, never hand slots
  or physical-copy identities. Secret draws preserve it. A public departure
  decrements the known definition minimum regardless of which duplicate left.
  Private randomization must define invalidation at its rules operation.
- Determinization and possible-world materialization reserve known counts and
  sample residual Hand-plus-Library copies. Outside copies never join that
  pool; residual deals determine weights. New knowledge needs explicit
  hash/clone/undo coverage; scalar PlayerVitals journaling is insufficient.
- The suspended Learn frame resumes once after retrieval, discard-then-draw,
  or decline. No priority intervenes. Preceding draw/damage/bounce and ordinary
  source finalization remain authoritative.
- Matching dimensions or unchanged opcodes do not establish compatibility.
  Corrected Learn requires w3 and full setup/rules/schema/pack bindings in
  ordinary checkpoint writers and loaders. Preserve frozen w2 evidence.

## Evidence boundary at 2026-09-25

Native full Learn, finite ownership, reveal/knowledge, residual worlds and
nested rollback have focused debug evidence. Native/Python observations carry
owner outside programs and public definition counts. The live browser uses
mode-first controls; the bounded scalar/vector encoders preserve Learn kind,
outside location and focus, default to 64 action rows and reject capacity
excesses. Those rows do not yet join complete programs or known-definition
facts into actual policy input. Ragged catalog availability and matching
tensors alone do not establish that a configured policy consumes their meaning.

The real `./scripts/play --demo learn` launcher uses asset version 2.0.0 and a
checked legal GameSession prefix into **Divide by Zero** Learn at revision 29,
UR seat 0 against seeded Random GW. It is an unscored interaction fixture.
The controls review reports both Learn browser tests passing on development,
built preview and the actual wrapper: three outcomes, previews, Back, keyboard,
navigation atomicity and reconnect reset. This does not prove the proposed
Pop Quiz or reversed-seat Search demo.

In the existing live human session, after receiving the URL and walkthrough,
the human replied **“works. approved”**. This accepts the working interaction
and requests no control or sideboard changes. It is not an itemized report of
every outcome: that session collected no per-attempt Command tape or match
identity. Do not infer them, reopen the accepted interaction gate, or create a
duplicate human session. This approval does not authorize publication, merging
or Task completion.

The latest compression reports 22 pack/setup/demo tests passing and changed-file
Ruff/whitespace checks passing, with no Rust changes. The preceding controls
review reports 53 debug library and 5 protocol tests passing, plus focused
Python/browser checks. Full gates remain open: that review retains 17 unresolved
broad Python failures after its bounded repairs and two historical Rust
source-drift failures. These are dated reports, not tests rerun by this memory
update and not a green full-branch gate.

ETU-75 still owns complete policy inputs and bindings, ordinary w3 checkpoint
safety, configured Search and reversed-seat demonstration, remaining directed
rule/rejection/large-choice cases, and full-root live/headless/replay parity.
Before scoring, register seeds 0–7 in both seats, exact world/setup/source and
deterministic server-offer policy identities, and a 10,000-Command cap per game.
Retain all 16 attempts, including caps and errors; require zero same-tape
mismatches. Deliver the exact-source aggregate verifier. No numeric runtime
target, expensive training or strength claim is implied.

## Source history

The source documents below are retained in local checkpoint
`d1326f249609daa781e786f5ee103aaeea3e1651`; that checkpoint is not published as
of this update. The paths remain available in the working tree. Keep them while
implementation continues; this curated record does not replace the full design
or its detailed verification matrix.

- [Approved design and slice ledger](../../scratch/complete-learn-lesson-in-the.md)
- [Human session and exact approval scope](../../scratch/demo-human-learn.md)
- [Playable-controls review and failure inventory](../../scratch/review-slice3-controls.md)
- [Outstanding assumptions and evidence](../../scratch/questions.md)

`lf status rules --json` confirms ETU-75's ownership and this worktree, with an
unfinished working PR and no publication. Its current chapter is
`20260924-trained-challengers` (Project `ce74bb65-b93c-4ec1-b971-790222a2c0bc`);
all three KRs remain unproven, with no numeric targets or recommended Flow set.
The GOAL's older Project pointer differs from that operational record. The
mandate has not changed; this update neither changes it nor rotates the chapter.
