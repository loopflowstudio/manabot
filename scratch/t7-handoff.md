# MTG-226 — TLA T7 launch direction

Jack Heart requested launch and opening the Task conversation in Rex on
2026-11-03. Continue existing MTG-226, not a duplicate Task. Publish one cohesive
PR for review; no merge or auto-merge is authorized by this launch.

## Outcome and scope

Implement T7 from `docs/rules/tla/delivery-plan.json`: player control and full-file
acceptance. The allocated card is Secret of Bloodbending. Its optional additional
waterbend changes combat-only control to whole-turn control of another player;
implement delegated decisions, permitted visibility, duration and self-exile
through shared native rules. Consult the printing, allocation and rules sources
before finalizing semantics. Do not silently approximate an optional choice.

Separate whose game decision is being made from who is authorized to choose it.
Academy, policy actions, text observations, replay and search must consume the
same authority. Identify delegated choices and configured automation explicitly.
Keep complete legal actions and viewer safety; do not add human-only semantics.
Prove activation/payment, both durations, delegation expiry, seat reversal,
appropriate private information, stale/duplicate rejection, reconnect and
deterministic replay with directed native debug and real consumer checks.
Use the established versioning policy for changed observation/action contracts;
new state must reach native/Python tensors, learning tokens and text together.
Keep one Rust CardDefinition representation and the printing's Scryfall id.

Full-file acceptance means reconcile all allocated TLA faces with definitions
and actual implemented obligations, identify any unallocated row explicitly,
and exercise valid TLA deck play through Academy and training consumers. Do not
equate admission or passing isolated card tests with full interaction coverage.
Preserve denominators and concrete uncovered cases. Current main admits defined
cards despite known bugs; do not reintroduce incomplete-card exclusion lists.
Do not launch paid training or inherit old experiment budgets for this task.

## Dependency handoff

Stack on T6 MTG-225 / PR #417's published head. T5, D1 decklist import and
MTG-235/236/237/238/239 foundations are already integrated there. T6 supplies
one global catalog, faces/Sagas, turn and combat scheduling, retained ability
state, Sozin payment/search authority, and referenced-row learner encoding.
Do not reimplement those foundations or revive authored pack multiplicity.

T6 head `7c790f78e` implements the accepted public schedule projection and
independent delayed event/frequency/expiry composition. Ordered extra turns/resume
and current/pending combat restrictions reach Academy; focused retained-session
recovery and native/Python/text/token proofs are recorded. Mutable-w17 receipt
refresh to hash contract 15 and combined gate validation remain outstanding.
Its implementation Graph is a31a0147-dab4-464d-8112-3cddf160516b. Start T7 from the published parent
only; never copy a live dirty T6 tree. Expect overlap in turn/decision state,
observation, experience projection and Academy. Incorporate T6's later published
completion through a supported sync before final combined validation. Landing
T7 remains dependent on T6; starting now does not waive this ordering.

Keep T7's working design in its own scratch. Inspect its selected code Graph and
existing repository guidance. Resolve routine design choices from accepted
requirements; ask genuine product questions in its opened conversation. Do not
open another conversation just to ask for review.
