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
then loads the viewer-safe presentation tail. Malformed presentation clears
the optional theater and reports an error without undoing the authoritative
frame. New games and failed resume attempts clear old theater.

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

Recovery continues to send an empty `presentation_tail`; checkpoint/tail
reconstruction is the explicit downstream recovery dependency, not part of
this sequence. Upgrading the viewer projection to exact render identities is
also separate. Converting either concern into arbitrary snapshot-diff text
would break the contract.
