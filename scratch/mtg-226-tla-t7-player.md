# MTG-226 — TLA T7 implementation plan

Status, 2026-11-03: T7 implementation is merged with published T6
`16336f2a0f9d0273d7bb9959b0a6b2b407fdea1e` at `cd098d8e4`, including delayed
event/frequency/expiry separation. Local publication validation is complete;
CI, browser evidence and review remain pending. Jack Heart
authorized one cohesive PR for review, including stacked publication; landing
remains parent-first and requires review. No merge or auto-merge is authorized.

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

T7 began on published T6 `01ed2fbcb`. The merge at `20f42a09a` incorporates
published T6 `bf7631c6a`, including ordered schedule presentation/recovery,
creation-time delayed-selector capture, and main `3b122de90`'s v6 fact capacity.
Both schedule and player-control projections survive through native, Python,
TypeScript and Experience schemas. No dirty parent tree was copied.
T5, D1 decklist import and MTG-235/236/237/238/239 foundations are integrated.
One catalog, face/Saga state, scheduling, Sozin payment/search authority and
referenced-row learning remain the shared foundations.

T6 PR #417 remains open with its code Graph running. Jack Heart's later
2026-11-03 decision (comment `55355aab-4b5f-47e6-bcaf-2cb3326a3d08`) separates
delayed-event matching, once/repeat frequency and expiry. The publication sync
integrates that completion from `7c790f78e`; no dirty parent changes were copied.
Parent-first landing remains required.

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

Accepted design, reconciled against merged implementation `20f42a09a`.

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
active control wins. `ActionSpace.player` is the chooser; existing native
Decision/Action players keep the rules subject. Publication resolves
`decision_controller` once from legal actions; there is no duplicate identity
field. Env rewards, structured/semantic offers, capture, search and Academy
read the published chooser directly.
In-game observations are the union of the viewer's and controlled player's
permissions; sideboard inspection stays with the original viewer. Learn's
native eligibility rejects outside-game selection while controlled. Known hand
definitions survive expiry; exact private card visibility does not.

Public v5/v6 continuation kind 13 carries control's controller, affected player,
duration, pending/active state and creation order; kind 14 names a delegated
current decision. Existing native/Python relation encoders carry both without
widening rows. V6 is now the fresh default with configurable fact capacity
(default 512); v5 stays at 128 facts and v1–v4 keep their exclusions. Saved
configurations retain their version and capacity through `from_checkpoint`.
Learning tokens derive from the rules enum; text and Academy retain the same native permissions. Hash
contract 15 binds the combined mutable state. No frozen evidence is regenerated.

### Consolidation completed

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
of numeric phase/step coordinates. Structured binding, private visibility and
deterministic order tests remain.

### Implementation outcome and remaining work

Implemented Secret of Bloodbending, both durations, optional additional waterbend,
self-exile, native delegated authority and inspection, Learn/sideboard exclusions,
ordered public control schedules and Academy presentation. The published chooser
comes from existing Action.player; no second ActionSpace identity survives.
The former partial action_player helper is removed. State hash contract is 15;
new public relation kinds 13/14 retain v5/v6 widths and frozen v1–v4 exclusions.

Full-file reconciliation lives in `docs/rules/tla/acceptance.md`: 286 identities,
294 faces, nine fixed tokens, Copy marker and excluded Treasure; all definitions
are present with behavior witness links and explicit uncovered interactions.
Valid submitted 60-card TLA lists reach delegated cleanup in Academy and the
training Env; retained recovery preserves the actual offers and inspection.

The merged default-v6 policy path agrees with Python encoding, text and Academy.
Native observability checks retain explicit v5 evidence. This reconciliation also
regenerates the five replay/Study/advice schemas that omitted `player_controls`
from their shared projection type; historical fixtures and optional-field
semantics stay unchanged. No engine behavior repair was needed here.

Prior pre-sync evidence at `d91c5f011:scratch/mtg-226-tla-t7-player.md` includes
16 structured-action checks, named-search recovery, allocation/token/Scryfall
checks, protocol/Svelte checks and the atomic release-extension rebuild.
Those results apply to their tested inputs, not broad acceptance of this merge.
The inherited T6 notes are historical evidence; their older pending statements
do not override this plan. Engine memory now records the accepted chooser and
shared-visibility contracts; no child memory directories exist in this checkout.

Remaining work: final-head CI Engine Budget and other CI checks; complete
presentation review/demo and Jack Heart's PR review. Browser layout/interaction
evidence is unavailable in this headless run. No paid
training or new experiment authorization is implied. No new product decision is
needed; documented uncovered cross-card combinations remain coverage limits,
not grounds to remove defined cards from play.

Check: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 -p managym-cards --test cards secret_of_bloodbending` — 10 passed; `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --test canonical_replay_tests --test study_protocol_tests --test study_index_protocol_tests checked_in` — 3 passed; `uv run --no-sync --extra dev --extra play pytest tests/semantic/test_player_control.py tests/academy/test_player_control_session.py tests/academy/test_schedule_recovery.py tests/academy/test_experience_protocol.py -q` — 19 passed across focused invocations.

## Concept review — 2026-11-03

Outcome: **nothing to change and no open product choice**. Source review of
`20f42a09a` plus the existing reconciliation edits agrees with Jack Heart's
accepted chooser contract. This is concept-review evidence, not final combined
acceptance or authority to publish, merge or select the next Graph step.

The consequential mismatch was treating the chooser as the owner of the action's
resources. The implementation resolves it at `publish_action_space`: the legal
action retains its native player, the published space names its chooser, and
turn ownership remains independent. No `acting_player` alias or browser-owned
control lifecycle is needed. The ordered native control records retain pending,
active and overridden effects; their effective status is derived for presentation.

Usage wording, describing existing behavior: “When you control another player,
Academy says who chooses for whom. Choose from the offered actions using that
player's cards and resources. The control list says whether the effect is pending
or active and whether it lasts for a combat or turn. Reconnect to continue the
actual pending choice; retrying a committed Command does not make it happen twice.”
Existing AGENTS.md decision-ownership guidance already states the small API;
no skill or usage contract needs a new concept.

Follow-through: native inspection permission supplies policy, text, Academy and
search. Sideboards and outside-game choices retain their separate restrictions.
Control state participates in deterministic identity, continuation kinds 13/14
and the optional shared projection. Missing historical projection data stays
unknown. Retained recovery restores the same frame, offers and inspection;
there is no new persistence service or promise of process-crash recovery.

Counterexamples retained: a delegated cast cannot spend the chooser's mana;
controlling the chooser cannot apply authorization twice; an overridden record
cannot be discarded while it still has a lifetime; expiry removes physical
inspection without erasing known hand definitions. The normal cleanup path,
disconnect before commitment, duplicate retry and stale rejection have existing
directed witnesses. Repeated cleanup is still a declared integration coverage
limit: `turn_tick` repeats cleanup without advancing the interval, but that code
inspection is not an end-to-end Bloodbending proof.

Next proof remains the already-planned published T6 integration and combined gate,
including presentation and consumer validation. This review ran no new game tests
and adds no acceptance claim for the documented uncovered combinations.
The review surface is `scratch/concept-review.html`; its inherited T6 predecessor
is retained at `20f42a09a:scratch/concept-review.html`. No unresolved product
question or new implementation direction was found.

## Publication validation — 2026-11-03

Synced T6's delayed-lifetime implementation at `cd412daa3`, then its notes-only
update at `cd098d8e4`. The second merge retains T7's concept review; T6's review
is preserved at `16336f2a0:scratch/concept-review.html`. No runtime source changed
in that second sync.

The full debug workspace run (`CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2
--no-fail-fast`) had 958 passes and one stale conformance-receipt failure.
Refreshed only mutable w17 evidence; the failed conformance test then passed.
All 14 cases retain command counts and outcomes (3,473 Commands). Fixed three
test-only Clippy warnings; the 10 Bloodbending tests passed afterward, as did
all-target/all-feature Clippy, formatting, changed-file Ruff and diff checks.
The release cp312 extension was rebuilt and installed atomically.

Combined consumer/admission validation: 86 passed across player control,
schedule/named-search recovery, Experience protocol, Elemental admission,
evaluation identities/mirrors, monitoring and choice separability. The broader
model/semantic/Academy/belief/active-training/encoder/package-boundary run had
444 passes, one skip and one stale T6 text assertion. That fixture now names
event matching, once frequency and expiry separately; all 54 text tests pass.
Nine authority/public-commitment/trigger-order checks pass. New test module names
also collected with the combined suite (446 tests).

Rebound the authored receipt, both parity corpora, Academy fixtures and current
belief pins. All 135 authored Commands, semantic events (modulo catalog definition
IDs), 66 public commitments and presentation hashes remain. Bloodbending inserts
one name-ordered definition, shifting later IDs; kind-5 CARD_REVEALED sources
are definition IDs too. Authored parity passed 816 viewer checks; public
commitment parity reports zero gaps. Retrieved the unchanged RUL-9 input through
the artifact manifest; no artifact publication or historical evidence mutation.

Frontend: 122 tests, Svelte check and production build pass. No browser rendering
proof was run. CI Engine Budget still needs this PR's same-runner measurement;
no local speed claim or baseline adoption is made. Publication is for review,
stacked on T6, with no merge, auto-merge or Task completion.

Command logs are local `/tmp/mtg226-*.log`; the committed receipt writers and
tests reproduce the durable evidence. No servers or watchers were started.
