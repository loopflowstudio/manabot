# MTG-226 — TLA T7 implementation plan

## Resume implementation — 2026-11-04

Main `5dd11cec4` is integrated at `d10f4bc71` through `lf ops sync --manual`; merged T6
supplies the pilot fixture isolation repair. Synthetic scheduling/analysis uses
frozen v5 identity and widths; actual runtime and schema rejection stay strict.
No frozen protocol, regime or experiment evidence is rewritten.

### Delete — do not maintain

- Removed `private_search`'s viewer-owned library heading; use “Search cards”
  and preserve native instructions/subjects for specificity.
- Removed inspection-container ownership from `visibleCardLabel` and bare-name
  inspection buttons. Native `owner_id`/`zone` describe private cards; keep
  board regions viewer-relative and ordinary battlefield controller labels.
- Retained main's deletion of inherited T6 scratch. Exact pre-sync history is
  at `02241d414:scratch/`; accepted T7 review edits remain in this checkout.

Implemented accepted option 1a: neutral search heading and inspected-card labels
from native owner/zone, with unchanged viewer orientation and native delegation
instructions. Directed Islandcycling tests cover both seats/durations, intervening
own-player priority, exact search forks and library-inspection expiry. Svelte
checks cover the corresponding labels, resources, transitions and expiry. The
existing WebSocket test covers retained cleanup recovery and duplicate/stale
Commands; no search WebSocket recovery or browser geometry claim is added.
The release cp312 extension was rebuilt and installed atomically after sync.

Check: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 -p managym-cards --test cards secret_of_bloodbending` — 10 passed; release maturin build and atomic cp312 install passed; `uv run --no-sync --extra dev --extra play pytest tests/semantic/test_player_control.py tests/academy/test_player_control_session.py -q` — 8 passed; focused Vitest — 13 passed; Svelte check, Ruff and diff checks passed. Logs: `/tmp/mtg226-compress-*.log`. Prior resume's unchanged pilot fixture check passed (39 tests); gate owns broad suites and mutable source fingerprints, demo owns browser geometry.

Remaining: gate owns affected suites, mutable parity source fingerprints and
final CI. Demo owns browser geometry; publication remains a separate Graph step.
No merge, auto-merge, Task completion or experiment execution is authorized.

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
artifact `scratch/concept-review.html` remains unchanged as dated evidence.

Usage: Your board stays in place while you control another player. Academy names
who you choose for. Use their cards/resources and inspect only what the engine
reveals. Reconnect resumes the actual pending choice; retry cannot commit twice.
No new identity, browser control lifecycle or recovery service is needed.

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
to delete. Gate owns broad suites, changed mutable source fingerprints and final
CI/Engine Budget; demo owns browser geometry; publication is a later Graph step.

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
closed without merging and T6 #417 had merged. Resume the existing Task in this
checkout, preserve the accepted concept-review edits, and return the remaining
work to review. The prior concept-review-only boundary no longer prevents
implementation. No merge or auto-merge is authorized.

Apply accepted option 1a: stable viewer perspective, neutral search heading
("Search cards"), explicit native player/resource context, no new identity
aliases. Complete the planned presentation proof with scoped existing browser
fixtures where feasible; retain honest coverage limits.

Use supported Loopflow sync to reconcile against main after T6's merge, preserving
T7's changes and frozen evidence. The last observed PR #418 head is `02241d414`.
Its later main-integration run 41789025455 / job 126437843412 fails 27 tests in
`tests/eval/test_tla_pilot.py` with "pilot runtime world differs from the
prospective protocol" (278 passed, one skipped). The earlier heuristic inventory
failure is already repaired. Read `/tmp/mtg226-ci-current-failed.log` or fetch
that job's logs. Diagnose the pilot contract against current main; do not weaken
world admission or rewrite frozen protocol/checkpoint/experiment evidence to
make CI pass. All other observed checks, including Engine Budget, passed on
that head; they do not prove the resumed combined tree.

Publish the repaired change through this Task's existing delivery history and
stop for review. PR closure is now superseded by explicit resume authorization.
Do not replay completed implementation wholesale: implement the accepted review
correction, repair integration, verify affected paths, and publish. Reconcile
the closed PR/base through Loopflow's supported delivery operations.
