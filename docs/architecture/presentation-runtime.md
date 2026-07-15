# Presentation runtime vertical slice

W2-184 adds the client-side consumer for ordered, viewer-safe semantic events.
It deliberately does not generate presentation facts from observation diffs.
The recorded Lightning Bolt fixture proves the narrow cast, target, resolve,
damage, and death vocabulary before server integration.

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

Replay frames carry the same optional `PresentationEvent[]`; selecting a replay
frame loads those values into a separate instance of the same player. The
timeline does not advance to the next authoritative frame until every semantic
beat for the current frame has played or been skipped. The trace writer does
not persist that array yet, so existing traces correctly
produce an empty sequence rather than inferred snapshot-diff events.

The future decision inspector should use `presentationInspectorRows`, adding
policy/search metadata beside those rows rather than reconstructing narration.

`presentationLabelsFromFrame` resolves object, player, and stack references
only through the viewer-safe frame. Labels are presentation context, not
authority. The frame is committed before theater starts, so skip,
fast-forward, reduced
motion, unmount, or recovery cannot alter or delay canonical game state.

## Known protocol mismatch

The Python authority currently emits empty presentation arrays. Its first
`ExperienceFrame.projection` also reuses the legacy `Observation`, which lacks
object incarnations and separate stack render IDs. The display-label bridge can
therefore name incarnation-zero objects and legacy stack card IDs only; it
leaves other exact references unnamed instead of guessing.

The `targeted`, `resolved`, and `died` variants make the requested Bolt beats
explicit alongside `cast` and `damage`. `died` records the resulting creature
death without incorrectly claiming that lethal damage was a rules-level
destroy action. The authority still needs to emit
those variants from domain events, upgrade its viewer projection to exact
render identities, and persist the same event values in traces. Converting any
of these facts to arbitrary snapshot-diff text would break the contract.
