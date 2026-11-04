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

T6 is still running a final accepted schedule-presentation completion: public
ordered extra turns/resume and current/pending extra combats with attacker
restrictions must reach Academy and retained-session recovery. Its implementation
Graph is a31a0147-dab4-464d-8112-3cddf160516b. Start T7 from the published parent
only; never copy a live dirty T6 tree. Expect overlap in turn/decision state,
observation, experience projection and Academy. Incorporate T6's later published
completion through a supported sync before final combined validation. Landing
T7 remains dependent on T6; starting now does not waive this ordering.

Keep T7's working design in its own scratch. Inspect its selected code Graph and
existing repository guidance. Resolve routine design choices from accepted
requirements; ask genuine product questions in its opened conversation. Do not
open another conversation just to ask for review.

## Design acceptance — 2026-11-03

Jack Heart approved the design summary in the primary Task conversation:
separate decision ownership from chooser authority; schedule next-combat or
next-turn control with optional waterbend and explicit expiry; share permitted
visibility through native authority across Academy, policy, text, search and
replay; identify delegated choices; validate payment, durations, privacy,
reconnect, stale Commands, deterministic replay and full-file TLA consumers.
Implementation may continue. Jack Heart explicitly permits publishing T7
stacked on T6; T6 need not land before T7 publication. Parent-first landing and
publication for review remain the boundaries; no merge or auto-merge is approved.

## Accepted API — Jack Heart, 2026-11-03

`ActionSpace.player` identifies the chooser. Native `Decision.player()` and
`Action.player()` identify whose rules decision/resources/actions apply;
`TurnState.active_player` / `Game::active_player()` retain turn ownership.
Resolve control once at publication; dispatch and offer/Command authorization
consume the published chooser without remapping it. Preserve legal actions,
resource ownership, visibility and replay for overlap and reversed seats.

This supersedes the earlier summary that used ActionSpace.player for the rules
subject. Jack Heart requested consolidation, not another identity field:
`acting_player` was an assistant suggestion, not a user requirement. The final
clarification reuses the existing native Decision/Action context.

## Working design — 2026-11-03

Implementation is based on parent `01ed2fbcb`. The code Graph owns this
compress step; later demo/delivery/review steps retain publication boundaries.

Rules sources: Wizards TLA release notes pp. 40–41
(`https://media.wizards.com/2025/downloads/TLA_Release_Notes_06HVqa76yI/EN_MTGTLA_ReleaseNotes_20250925.pdf`),
Scryfall printing `9bb928ae-f636-4aee-9146-a7885e6a8976` (TLA 69, read 2026-11-03).
The cost is UUUU, with optional additional waterbend 10 announced at casting;
the spell targets an opponent and exiles itself on resolution. Control applies
to the next actual combat or next actual turn, including its cleanup decisions.
The affected player stays active and retains their cards, abilities and costs.
The controller also makes their own decisions. No private sideboard inspection,
outside-game selection or concession on the affected player's behalf is allowed.

The existing optional additional-cost path owns payment. `ControlPlayer` over
an `ObjectSelector` schedules `PlayerControlDuration` on TurnState, in creation
order, and ordinary interval advancement activates/expires it. Last-created
active control wins. `ActionSpace.player` is the chooser; the existing Decision.player and
Action.player keep the rules subject. Publication resolves `decision_controller`
once from the legal actions; there is no duplicate identity field. Env rewards, structured/semantic offers, capture,
search and Academy read the published chooser directly.
In-game observations are the union of the viewer's and controlled player's
permissions; sideboard inspection stays with the original viewer. Learn's
native eligibility rejects outside-game selection while controlled. Known hand
definitions survive expiry; exact private card visibility does not.

Public v5 continuation kind 13 carries control's controller, affected player,
duration, pending/active state and creation order; kind 14 names a delegated
current decision. Existing native/Python relation encoders carry both without
widening rows. V1–v4 continue excluding kinds >=10. Learning tokens derive from
the rules enum; text and Academy retain the same native permissions. Hash
contract 14 binds the new mutable state. No frozen evidence is regenerated.

### Delete — do not maintain

Removed downstream control remapping and the partial `action_player` helper;
publish the chooser once on ActionSpace. Shared native information permission
replaces own-hand-only visibility, retaining private sideboard and closed-deck
sampling boundaries. No old player-control mechanic or card definition remains.

Removed `AtomicCommand::DeclarationSequence.player` and
`InternalOffer::DeclareBlockers.player`: their bound actions already carry the
rules subject, and exact legal-action equality checks it during lowering.
`remember_controlled_hands` delegates to existing `remember_hand`; it does not
maintain a second definition-count implementation. `Observation::add_card` is
inlined into ordered collection with one local seen set, avoiding repeated scans
of every accumulated row. Named StepKind values own duration boundaries instead
of numeric phase/step coordinates. No remaining deletion targets are identified;
retain structured binding, private visibility and deterministic order tests.

### Implementation outcome and remaining work

Implemented Secret of Bloodbending, both durations, optional additional waterbend,
self-exile, native delegated authority and inspection, Learn/sideboard exclusions,
ordered public control schedules and Academy presentation. The published chooser
comes from existing Action.player; no second ActionSpace identity survives.
The former partial action_player helper is removed. State hash contract is 14;
new public relation kinds 13/14 retain v5 widths and frozen v1–v4 exclusions.

Full-file reconciliation lives in `docs/rules/tla/acceptance.md`: 286 identities,
294 faces, nine fixed tokens, Copy marker and excluded Treasure; all definitions
are present with behavior witness links and explicit uncovered interactions.
Valid submitted 60-card TLA lists reach delegated cleanup in Academy and the
training Env; retained recovery preserves the actual offers and inspection.

Checks: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 -p managym-cards --test cards secret_of_bloodbending` — 10 passed; `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 -p managym --test rules_tests structured_` — 16 passed; `uv run --no-sync --extra dev --extra play pytest tests/semantic/test_player_control.py tests/academy/test_player_control_session.py tests/academy/test_named_search_recovery.py -q` — 5 passed. Release extension rebuilt and installed atomically; Rust formatting, changed-Python Ruff and diff checks passed. Earlier allocation, effect-token, Scryfall, protocol and Svelte checks remain applicable. Gate owns affected suites, mutable w17 receipts, arena/monitoring validation and CI Engine Budget.

Scratch parent notes were consolidated into `questions.md` with precise git
references after verifying their bytes matched the published parent. Wave memory
was read and remains unchanged; both context budgets pass `lf context`.

T6 is still live (`a31a0147-dab4-464d-8112-3cddf160516b`, implement loop 2 pass 2
on the latest status read). Its later schedule presentation and delayed-selector
completion must be incorporated from a published head before combined acceptance.
No dirty parent state was copied. Publication remains stacked review-only with
parent-first landing, no merge or auto-merge. No paid training ran.
