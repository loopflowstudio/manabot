# MTG-226 — TLA T7 implementation plan

## Reconciled implementation — 2026-11-04

Main `5dd11cec4` is integrated at `d10f4bc71` through `lf ops sync --manual`; merged T6
supplies the pilot fixture isolation repair. Synthetic scheduling/analysis uses
experiment protocol v5's identity and widths (policy input v7, 128 facts,
64 actions, 128 cards per player); actual runtime and schema rejection stay strict.
No frozen protocol, regime or experiment evidence is rewritten.

Implementation checkpoint `f052be4ad` contains accepted option 1a: neutral search heading and inspected-card labels
from native owner/zone, with unchanged viewer orientation and native delegation
instructions. Directed Islandcycling tests cover both seats/durations, intervening
own-player priority, exact search forks and library-inspection expiry. Svelte
checks cover the corresponding labels, resources, transitions and expiry. The
existing WebSocket test covers retained cleanup recovery and duplicate/stale
Commands; no search WebSocket recovery or browser geometry claim is added.
The release cp312 extension was rebuilt and installed atomically after sync.

Check: `git diff --check` — passed for this documentation reconciliation; reused `/tmp/mtg226-compress-*.log`: focused debug Rust 10 passed, Python control/session 8 passed, Vitest 13 passed, Svelte check and release build passed; `/tmp/mtg226-resume-consumers.log` records 41 consumer/pilot passes. Broad suites, mutable source fingerprints and final CI remain with gate; browser geometry remains with demo.

Remaining work:

- Gate: final CI and Engine Budget for the resumed combined tree; local
  publication validation and mutable parity reconciliation are recorded below.
- Demo/review: both seats and durations in a browser, search, alternating own
  and delegated decisions, reconnect and expiry. Existing headless proofs cover
  behavior and rendering, not browser layout or search WebSocket recovery.
- Publication: return the repaired change for review through MTG-226's delivery
  history. On 2026-11-04, GitHub reports #418 CLOSED at `02241d414`, still based
  on the former T6 branch; Task status reports its PR phase as open. Delivery
  still needs to reconcile the closed PR and base. T6 #417 merged as `5dd11cec4`.

No product decision remains. Jack Heart's review is required before merge;
merge, auto-merge, Task completion and experiment execution are not authorized.

### Resumed publication validation — 2026-11-04

`lf sync` found main already integrated and checkpointed the accepted
reconciliation at `54778e314`. Full debug Rust passed 963 tests (two ignored).
The consumer/model/admission/monitoring/pilot selection passed 338 (one skipped);
the additional contract/privacy/replay selection passed 153 with three stale
parity-receipt failures. Frontend tests passed 126; rustfmt and Clippy passed
(Clippy needed an absolute `PYO3_PYTHON` path). Logs are
`/tmp/mtg226-publish-{rust,python,contracts,ui,clippy}.log`.

Main's `shared_effects` observation field explains all 816 changed authored
viewer snapshots: removing only that field in a diagnostic comparison reproduces
every prior state and observation hash exactly. The regenerated mutable w17
authored receipt retains the entire Command tape, state transitions, consequences
and presentation hashes. Public-commitment regeneration changes only source
provenance and observation-bound prior/posterior/history digests; all other
fields, including 135 Commands, 66 commitments, support, privacy and atomicity,
match. Frozen receipts and protocols remain untouched. The focused parity
recheck passed all seven tests (`/tmp/mtg226-publish-parity.log`). Svelte checking
and the frontend production build also passed (`/tmp/mtg226-publish-check.log`
and `/tmp/mtg226-publish-ui-build.log`).

## Outcome and acceptance

Implement Secret of Bloodbending's optional waterbend, next-combat/next-turn
control and self-exile through shared native rules. Full-file acceptance
reconciles all allocated TLA faces and exercises valid submitted decks in
Academy and training. Definition remains support; known bugs and coverage gaps
never become admission exclusions. No paid training or inherited budget applies.

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

Accepted design; the current combined base is `d10f4bc71`.

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

## Accepted presentation — Jack Heart, 2026-11-03

Jack Heart selected option **1a**, stable viewer perspective, in the interactive
concept review. The earlier automated “nothing to change” conclusion was
superseded: `private_search` also describes other players' zones, so “Search
your library” made an unsupported ownership claim. Use “Search cards” and native
instructions; inspected-container placement grants permission, not ownership.
Option 1b (switching board perspective) was not selected. The accepted review
artifact `scratch/concept-review.html` remains unchanged as dated evidence;
its “not yet implemented” captions describe that earlier review, not this tree.

Usage: Your board stays in place while you control another player. Academy names
who you choose for. Use their cards/resources and inspect only what the engine
reveals. Reconnect resumes the actual pending choice; retry cannot commit twice.
No new identity, browser control lifecycle or recovery service is needed.

Headless concept review, 2026-11-04: no design change or open product choice.
The implemented search heading, owner/zone labels and Academy usage text agree
with Jack Heart's option 1a. Following delegated Islandcycling through an own
priority response preserves the distinction between chooser, rules player and
turn owner without a second identity or browser lifecycle. The retained cleanup
recovery witness covers reconnect, duplicate acknowledgment and stale rejection;
it does not establish search reconnect or crash recovery. Existing counterexamples
and the browser proof below remain required; this source review adds no execution
evidence. Preserve the dated accepted HTML review unchanged, as requested in the
resume direction. No further implement direction follows from this review.

## Evidence and remaining boundaries

`docs/rules/tla/acceptance.md` owns the full-file inventory and witness links:
286 identities / 294 faces, nine fixed tokens, Copy marker and excluded Treasure.
Native tests cover both seats/durations, payments, resource ownership, overlap,
extra turns/combats, visibility, sideboards/Learn and exact forks. Shared policy,
text and Academy tests cover the same authority. Submitted 60-card decks reach
delegated cleanup through training and retained Academy WebSocket sessions.

Coverage limits remain copied Bloodbending, retargeting, Sozin multi-zone search,
repeated cleanup triggers and the separate Kuruk extra-turn combination. Native
interval code suggests repeated cleanup preserves control, but that is not an
end-to-end witness. Search forks do not prove search WebSocket recovery; Svelte
server rendering does not prove browser geometry or crash durability.

Historical validation at `efef751ff` covered 958 debug passes plus a repaired
mutable conformance receipt, 86 combined consumer/admission checks, broader
444-pass/one-skip validation plus a repaired text assertion, and 122 frontend
tests/check/build. Mutable w17 refresh retained all 14 kernel command counts
(3,473 total), the 135-Command authored tape, 66 public commitments and
presentation hashes; catalog insertion shifted definition IDs. Frozen evidence
was unchanged. PR #418's `2ea2a208d` CI later passed all observed checks except
the heuristic inventory, fixed at `02241d414`; its later pilot failure is below.
Those results do not certify the resumed combined tree or a retained checkpoint.

Current cleanup keeps inspection ordering and native permissions intact while
flattening pending-decision collection, reads UI owner IDs directly and expresses
the training proof's action preferences once. No obsolete runtime path remains
to delete. Remaining validation and publication are listed above.

## Historical sources

Complete pre-curation notes, including the uncommitted resume evidence, are
preserved as git blobs `ea55409dcdcfd2e5694cb5d08c0c172db95ff1a1` (this plan)
and `ba6475cd6d25da3e2df093a5b909365653b00ba6` (questions); read with `git show`.
Committed T7 history is at `02241d414:scratch/mtg-226-tla-t7-player.md`;
pre-sync parent notes at `02241d414:scratch/`. T6 delayed-lifetime integration
is `7c790f78e` → `cd412daa3`, notes sync `16336f2a0` → `cd098d8e4`;
main/T6 integration is `5dd11cec4` → `d10f4bc71`. Parent-first landing is now
satisfied by merged T6, not permission to merge T7.

## Resume direction — Jack Heart, 2026-11-04

Jack Heart explicitly requested resume after being informed that PR #418 had
closed without merging and T6 #417 had merged. This authorized implementation
of accepted option 1a, integration with main, pilot CI repair and publication
for review in the existing Task. It superseded the concept-review-only boundary;
it did not authorize merge, auto-merge or experiment execution.

The earlier main-integration run 41789025455 / job 126437843412 failed 27 tests in
`tests/eval/test_tla_pilot.py` with "pilot runtime world differs from the
prospective protocol" (278 passed, one skipped). The earlier heuristic inventory
failure was repaired at `02241d414`. Merged T6 supplies the compatible-world
fixture for synthetic pilot outcomes, with separate runtime/schema rejection
tests; no further runtime repair was needed. All other observed checks,
including Engine Budget, passed on the historical head. Neither those results
nor the focused resume checks certify the resumed combined tree's full CI.
