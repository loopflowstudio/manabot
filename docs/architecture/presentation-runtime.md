# Presentation runtime vertical slice

W2-184 carries one ordered, viewer-safe Lightning Bolt sequence from the
authority through live play and the persisted replay trace. It deliberately
does not generate presentation facts from observation diffs.

## W2-183 integration seam

W2-183 landed the authoritative `FrameUpdate.presentation` and
`RecoveryEnvelope.presentation_tail` fields in protocol v1. W2-184 now imports
that `PresentationEvent` type instead of maintaining a parallel interface.

The live socket path follows one ordering rule:

1. sequence-gate the `FrameUpdate`;
2. atomically commit its complete `ExperienceFrame`;
3. validate every event against the update's base and resulting revisions;
4. enqueue the ordered events into the live `PresentationPlayer`.

Recovery similarly commits the complete frame, cancels current theater, and
then loads the viewer-safe presentation tail at `presentation_cursor`. Every
event must be contiguous from that address; a gap is rejected and requests a
fresh envelope rather than being reordered or invented locally. Malformed
presentation clears the optional theater and reports an error without undoing
the authoritative frame. New games and failed resume attempts clear old
theater.

Replay frames carry the exact `PresentationEvent[]` persisted on the authority
transition's final trace step; selecting a replay frame loads those values into
a separate instance of the same player. The timeline does not advance to the
next authoritative frame until every semantic beat for the current frame has
played or been skipped. Traces written before this field existed correctly
produce an empty sequence rather than inferred snapshot-diff events.

The decision-inspector seam is `presentationInspectorRows`. It projects the
same canonical event objects used by the table and retains each original event
beside its accessible beat text, so later policy/search metadata can be joined
without reconstructing narration.

`presentationLabelsFromFrame` resolves object, player, and stack references
only through viewer-safe frames. The live path merges the previous and
resulting frame labels; replay does the same with adjacent observations, so a
creature that just died retains its name without treating disappearance as a
death fact. Labels are presentation context, not authority. The frame is
committed before theater starts, so skip, fast-forward, reduced motion,
unmount, or recovery cannot alter or delay canonical game state.

## Authority projection

The match-local Python `PresentationProjector` stages only exact identities
chosen through server-authored offers. This is necessary because target
selection can cast and resolve Lightning Bolt in one engine step, leaving no
post-step stack to inspect. Staging does not emit theater: facts become visible
only when the engine's committed event window contains the corresponding
`SpellCast`, `DamageDealt`, `SpellResolved`, and battlefield-to-graveyard
`CardMoved` records. The projector then emits, in order, `cast`, `targeted`,
`resolved`, `damage`, and `died`, validates them through the protocol-v1 model,
and binds them to the accepted command and revision transition.

The same list object is returned in `FrameUpdate.presentation` and persisted
in the trace. A scenario that changes zones without those domain events emits
nothing. Lethal damage produces `died`, never the rules-distinct `destroyed`.

## Integration boundary

The first `ExperienceFrame.projection` still reuses the legacy `Observation`,
which lacks object incarnations and separate stack render IDs. This narrow
adapter therefore certifies incarnation zero and reuses the visible spell card
ID as the stack render ID. The display-label bridge leaves other exact
references unnamed instead of guessing.

W2-203 extends this same projector with committed combat-domain facts. Native
rules sites add exact, viewer-safe identities to `recent_events` when attacker
and blocker declaration completes, combat damage is assigned, state-based
deaths commit, and a turn starts. The projector consumes those facts in native
order and emits `attack_group`, `blocked`, `damage`, `died`, and
`turn_started`. It does not compare observations. The fixed Rust and Python
learning encoders filter these additive presentation-only event types, so
policy inputs remain unchanged.

Combat prompts also expose native declaration metadata. This distinguishes
`Attack with X` from `Do not attack with X` without interpreting option
position in the client; blocker choices remain authority-published legal
offers. The same event dictionaries are returned live, persisted on the
transition's final trace step, replayed through the shared player, and mapped
to decision-inspector rows.

Recovery uses one complete `ExperienceFrame` plus a bounded, match-local ledger
of the same events returned by live updates and persisted in replay. The cursor
names the first returned event; an empty tail keeps that cursor unchanged. If a
requested address is outside the retained window or ahead of the authority,
the complete frame converges game truth and presentation restarts at the oldest
retained address. The ledger keeps the latest 256 events and is deliberately
not a durable cross-process checkpoint.

Upgrading the viewer projection to exact render identities is separate.
Converting that concern, recovery, or any combat fact into arbitrary
snapshot-diff text would break the contract.

## Readable consequences and current decision (ETU-76, 2026-10-04)

The shared live/replay Recent consequences component retains the latest 12 available semantic events in
an expandable, keyboard-scrollable Recent consequences list. Completion, Skip,
Fast-forward and Finish move the narration cursor without erasing this text.
Pause holds the current beat, including when another live update arrives.
Reduced motion removes movement without shortening reading time; explicit
Fast-forward still changes speed. The board remains at the authoritative current
position, as the history explains. Replay replaces the list when seeking to another
frame; a frame without semantic events says that narration is unavailable.

The live action panel names the current prompt actor and instruction separately
from narration and active turn. Waiting for a Command, disconnection, spectator
access, isolated Study and game-over have distinct status text. No presentation
control changes an offer, Command or authoritative frame.

At base `3f2975334e6fa2f1a36683d3686a7ecc2d68ffe9`, finishing either
`bolt-kills-ally-v1` (revisions 42–43, sequences 900–904) or
`ur-lessons-vs-gw-allies-combat-v1` (the checked-in
`frontend/src/lib/fixtures/curated-combat-to-turn.json`) removed all visible
semantic beat text; reduced motion allotted just 100 ms per beat. These are
reproducible fixture positions, not identified recordings of Jack Heart's report.
The regression tests retain Bolt casting/targeting/resolution/damage/death and
Allies/Lessons attack/block/damage/death/turn text after cursor movement. The
browser scenarios use a mocked authority and do not certify full-game behavior.

Coverage remains bounded. The authority projector still specializes spell facts
to Lightning Bolt, alongside combat/death/turn facts. Other spell families,
the original reported match/Command, human confirmation and full-game recurrence
remain open under ETU-76. A readable list cannot repair absent authority events.
Human acceptance is not inferred from fixture tests. No rules, world identity,
persisted trace format, or hidden-information boundary changed.

Validation on 2026-10-04: 49 focused presentation/store/socket/replay/protocol
tests pass, Svelte check has zero errors/warnings, and the production build
passes. Browser regressions were authored but not executed; the supplied run
had no rendering environment. Visual and browser acceptance remain unverified.

CI repair on 2026-10-05: PR #199 head
`b4e026f5bc8a88611c65d2cf44e00ec130f786b7`, run `37189676719`, failed
`board-combat` because the board grew from 1159 to 1419 pixels high. The
live grid stretched the board to the taller decision/history sidebar. Aligning
columns at the start preserves the board's content height without changing
visual references. A headless production-build regression reproduced sidebar
stretch (807 to 3000 pixels) before the fix and passes after it. Both narration
motion-mode scenarios, Svelte check and the build also pass locally on macOS.
The full release suite could not start without the local managym extension;
the pointer/keyboard scenario also needs backend HTTP routes. Linux screenshot
certification remains with CI, and human-play acceptance remains open.


Terminal capture repair on 2026-10-05: PR #199 head
`0a8a2a5b3a920bdc75ef924a62ce1018179ae494`, run `37409396849`, job
`112094186785`, reached `terminal-ur-lessons-loss.png` after the geometry repair.
Its successive screenshot diff changes the narration counter and damage text
behind the result dialog. Terminal capture omitted the Finish step used for
ordinary board references; disabling CSS animations does not stop JavaScript
beat timers, and reduced motion intentionally preserves reading time.
The shared capture helper now finishes narration at terminal without moving
result focus. Product narration, reference images and comparison tolerances
remain unchanged. A clock-controlled production-build browser test reproduces
changing terminal pixels before Finish, then verifies identical captures across
10 seconds of virtual time, retained consequences and Play Again focus.
Four focused browser checks, Svelte check and build pass on macOS. The local
managym extension is absent, so the full server-backed release matrix and pinned
Linux visual corpus remain CI responsibilities; this is not full-gate proof.
