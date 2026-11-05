# MTG-241 — composable rules execution

## Outcome and boundary

Jack Heart's accepted directive (2026-11-04): one PR through the code Graph,
stopping at interactive review; no merge, auto-merge, training or retained-evidence
rewriting. The Task remains **incomplete**. No human judgment is missing.
Binding checkpoint: `91517292c`; MTG-244 published v8/layers integrated at
`d77bc8676` from `1930146b0`. Earlier execution and hidden-consumer work remains
in the ancestry through `f1ce0ce13`. Owned bindings are complete. The loop feedback
written against `f1ce0ce13` predates that repair: its binding, joint-assignment and
regression items are done and pass at `660021431`. Its remaining items stand, in the
order given under Remaining implementation.

The selected Wave is `.lf/wave/coverage/`; its GOAL and full MEMORY were read.
Jack Heart's dated directive supplies this execution scope without reopening the
older chapter or authorizing experiments. The approved PRINCIPLES/README guide
came through PR #425 and remains intact. Earlier implementation and validation
history is preserved at `f1ce0ce13:scratch/mtg-241-composable-rules-execution.md`.
Reconciliation on 2026-11-04 inspected the selected Wave directory; it has no
child memory directories. Its accepted chapter remains unchanged.

## Surviving ownership and contracts

Native `Effect` is the sole authored program. `EffectFrame`/`run_frame` own
execution; `Decision` owns the current question. `FrameCompletion` retains one
return destination: caller frame/decision, operation event, spell cleanup,
paid announcement, activated ability, rule continuation or priority. Iteration
retains remaining targets, creates isolated child results and defers parent
suffix/finalization; branches share their scope. `EffectFrame::new` initializes
fresh results. `ObjectSelector::Targets` applies one verb to all eligible slots
in one batch; `ForEachTarget` instead runs a sequential child program per slot.
Leaves from the Vine uses `Targets` for its simultaneous counter placement.

`EventBatch` retains proposals, stable original ordinals, exact replacement
identities (source, definition index and namespace), cursor, departure LKI,
trigger sources, results and consequent child batches. Applicability is rechecked
after every replacement; the affected player chooses the order. No member commits
until all members are ready. Linked returns finish before the root's results are
bound and never replace those results. `bind_event_results` owns instruction
postambles after completion; `run_frame` owns scheduling and resumption.

Snapshots are limited to batches with departures. Departures match pre-batch
sources; entries match the finished battlefield. LKI is captured before members
leave. Earlier paid tap/life events match before sacrifice can remove sources;
triggers remain queued during replacement choices. Lifelink and damage counter
loss commit within the damage batch, without another trigger pass. Entry counters,
including Saga lore, pass through replacement before entry; emitted counts are
transformed counts. `apply_immediate_event` uses this same executor with an explicit
no-question precondition, never a pending-as-success adapter. The batch owns trigger
processing, including for immediate zone moves.

Mana activation transfers its caller by ownership. `advance_mana_upfront` commits
the automatic mana prefix, tap and life costs, then retains output and original
payment before running sacrifice. Linked-entry replacement choices can suspend
that cost without exposing output, paying again or granting priority. Activated
source/target metadata is captured before sacrifices; card-ability source identity
binds after discard cost. Spell entry/finalization, combat damage, SBA departures,
token creation, Earthbend, library placement, chosen exile/sacrifice groups,
counter distribution and LookAndSelect battlefield movement use retained batches.
First-strike retains normal damage across its SBA interval; SBA progress is
Stable/Changed/Pending. Token/attachment cleanup waits for child work, and
`turn_tick` exposes pending turn actions before stabilization.

`SelectCards` now migrates public results and hidden hand/library selections.
`ZoneSelection` retains owner, zone and predicate on decisions and selected child
scopes. Both samplers remap unknown commitments across the saved caller chain
with one exact-object matching; viewer-known choices remain exact. Incompatible
restricted count worlds are excluded, and materialization rejects impossible
commitments instead of dropping them. Direct determinization retries bounded
compatible deals and returns an explicit error on exhaustion. Ordinary roots take
no additional full clone. Requirements are captured before zone reassignment;
remapping follows selected/remembered/result aliases across caller frames. Mixed
foretell allocations count only compatible ordered fills, rather than retaining
all their weight after a feasibility check. The directed two-selected-instant /
two-foretold-slot witness retains 14 of 20 physical fills and resumes both samplers.
The frozen count-only belief sampler explicitly rejects retained-selection and
foretold domains before inference; its saved schema is not silently extended.
Library inspection exposes all permitted members in stable definition/object order,
never library order. Pending battlefield entrants
are public and pinned even before their retained moves commit.

`OptionalDiscardDraw`, `MayDiscard`, `SearchBasicLand`, `SearchLibraryToHand` and
their dedicated continuations/helpers are deleted. Definitions use selection,
movement, conditional effects and shuffle; optional discards commit the completed
subset together, and draws read committed `Moved` results. Library-to-hand movement
reveals definitions and updates known-hand minima. `CardsThisWay` retains latest
mill/discard independently of later moves. Shared selection writes code 28 on every
version, which is the open frozen-code defect below. Existing mill consumers retain
their earlier contracts.

`LibraryEnd` now lives with the authored rules types. `Move.library_end` commits
explicit top/bottom placement with the movement, before linked-return questions;
shuffle remains a subsequent instruction. The actual Merchant program's constructed
linked-return witness covers bottom placement, deferred shuffle and completion.
Real Merchant tests exposed sacrifice trigger predicates being read after their
subjects departed: committed sacrifice facts now match all pre-move subjects before
batch zone moves, with trigger placement still deferred.

`OrderLibrary` owns top/bottom ordering. Bottom keeps immediate placement and
remaining-only private visibility; top keeps the full inspected prefix pinned and
visible until complete. Rotation preserves the remaining offer order. Surveil
retains its graveyard partition privately until selection finishes, moves it as
one retained batch, then orders kept cards. Scry/Surveil completion events wait for
all choices (including empty-library completion) and match sources before the
caller's suffix can remove them. Planetarium recognizes both keywords. No current
real definition prints Surveil above 1: constructed programs prove arbitrary
kept order, while Joo Dee proves the registered interaction.

`SuspendedResolution::continuation` owns the root-first borrowed walk through
frames and saved mana decisions. Projection, inspection and determinization share
that ancestry. Obligation codes: bottom 23; top 27 (mode = committed prefix length);
shared selection 28; replacement 29.
Labels, source focus, text and Academy name those choices. These surfaces **do not**
complete retained payload observability. Prospective entry conditions now use the
entering controller; SourcePower reads exact current/LKI power, never a returned
incarnation.

## Delete — do not maintain

Completed: local non-suspending iteration; independent parent/finalize state;
cloned mana callers; `FrameFinalize`; `OrderLibraryBottom`; `MillAndSelect`,
`ReturnMilledThisWay`, `MilledBonus`, `SelectMilled`; optional discard/draw and
basic/tutor search variants with their dedicated helpers; synchronous linked
returns, sacrifice/destruction wrappers and result postambles. Historical detail
and checks remain at the committed plan reference above.

`continue_resolution(frame, next)` now owns batch-before-decision scheduling;
`run_batch_before_decision` and LookAndSelect's duplicate scheduling branches are
removed. `apply_replacement` owns both transformation and applied-identity tracking
for automatic and chosen replacements. These cuts preserve frame payloads and
serialized state. The directed inspection witness covers entry replacement before
both another selection and remainder ordering, with Command replay/stale checks.

Completed binding deletion cut: bare remembered-reference transfers, frame-only
discovery as the exclusive binding authority, and the no-suspended-frame shortcut
for retained binding refresh are removed. Preserve public delayed
captures, exact incarnation semantics, existing hidden-selection predicates and
saved-frame aliases. The binding implementation below owns that behavior. Distinct-land/named searches, Scry and LookAndSelect retain distinct
visibility and ordering contracts; their overlap still needs an audit.

The restored completed-capture regression supersedes the old unapplied patch;
its original failing form remains at `f1ce0ce13:scratch/completed-private-capture.patch`.
Capture inspection now visits bindings directly instead of allocating a temporary
list. Hand partitioning derives fixed hand/library membership from one snapshot;
minimum-count checks read zone membership without sorting an unused offer.
Entry replacement discovery borrows shared static abilities and appends matching
grants directly after own replacements, preserving candidate order and identities.
These cuts remove repeated collection and program cloning, not legal choices;
they do not establish a speedup or complete the risk acceptance below.

## Ownership revision — 2026-11-04

The completed-private-capture witness is now restored as an ordinary debug
regression. `ObjectBinding` replaces bare remembered refs and owns the exact
incarnation, captured zone/predicate and per-seat exact knowledge. `ZoneSelection`
lives with it in `managym-state`, so stack objects and engine frames use one type.
Bindings transfer through waiting, queued, detached placement (remaining/placing),
stack and resolving occurrences. Both samplers collect one joint assignment across
all those owners and active selections. Known bindings pin exact copies; stale
incarnations impose no live membership and never follow a later incarnation.
Moved captures take their new zone, rather than inheriting the old selection zone.
A later `SelectCards(Remembered)` has no zone marker; the joint assignment now
updates its candidates and selected child anyway, and direct determinization
refreshes that offer. Its directed Command witness originally retained entity 64
while the capture remapped to 41; the marker-gated predecessor is removed.
Canonical state serialization includes binding provenance. Viewer-facing references
still use native visibility and do not expose private binding identities.

The original defect and evidence are retained at
`f1ce0ce13:scratch/mtg-241-composable-rules-execution.md` and
`f1ce0ce13:scratch/completed-private-capture.patch`. Its frame-only discovery and refresh
shortcuts have been removed. Focused checks now cover both viewers, six lifetimes,
alias agreement, privacy and subsequent zone movement. Full retained-payload
projection and policy admission remain unfinished; this change does not establish
learning completeness or all hidden-zone consumers.

## Event-model correction — 2026-11-04

The binding repair is implemented, but the complete real-card counter witness
exposes a second missing domain fact. `ProposedEvent::CounterChange` retains
`target/kind/delta/by`; none distinguishes an effect, a cost or a turn-based
placement. `ModifyCounters` applies to every positive matching placement.
Consequently it cannot author Doubling Season faithfully. This is a source-level
representation counterexample, not a newly failing real-card test: Doubling
Season is not registered, and the constructed 3/4 driver never claimed its full
semantics. No current test result proves that missing behavior.

The pinned 2026-09-25 CR text (git blob
`4dc4fead596288be1ca6aa9c913ea1739c680a29`) makes the distinction explicit:
614.16 admits a resolving effect and also a replacement/prevention effect even
when the original event was not an effect. 714.3a makes entry lore an intrinsic
replacement, while 714.3c is the precombat turn action. The
[Foundations notes](https://magic.wizards.com/en/news/feature/foundations-release-notes)
confirm that adding loyalty as a cost is not doubled. The
[April 2026 bulletin](https://magic.wizards.com/en/news/announcements/secrets-of-strixhaven-update-bulletin)
confirms the Saga-entry change. The original source of a placement alone is
therefore insufficient: replacement history can make a later effect applicable.

There is a second part of the same card contract: `CreateToken` currently calls
`prepare_token` for each token before creating a zone-move batch. That can replace
entry but cannot replace the number of tokens in the logical creation event.
A counter-only Doubling Season definition would be incomplete. Caradora's full
search trigger and creature/Vehicle predicate also remain required.

Revised internal cut, within the existing single-PR boundary:
- A retained proposal owns initial cause and the replacement-produced effect
  status needed by CR 614.16. Costs remain costs for completion/payment ownership;
  the replacement applicability question reads causal provenance, not `by`.
- The single `ModifyCounters` rules type gains an explicit applicability filter;
  it does not gain a card-specific twin. Rechecking after a transformation reads
  the updated provenance, preserving exact applied identities and chooser order.
- Token creation becomes a retained logical proposal before token allocation,
  retaining count, definition/copy terms, controller and entry/attachment terms.
  After creation replacement choices, allocate the final group and use the existing
  simultaneous entry batch. No second token executor or count-only wrapper.
- Delete the eager per-token allocation path in `Effect::CreateToken` and
  `CreateTokenCopy` when moving their consumers. Preserve copied exceptions,
  created-result ordinals, tapped/attacking and attachment behavior, entry choices,
  trigger timing and once-only completion. Do not repair that predecessor.
- Directed acceptance distinguishes effect placement, unmodified cost/turn action,
  a non-effect modified by another replacement, intrinsic Saga/loyalty/defense
  entry, and token multiplicity with retained entry terms, alongside the actual
  Leaves/Caradora/Doubling Season 3-versus-4 choice and complete definitions.

Dependent retained-payload projection is stopped at this correction: encoding
only the current amount/owner would preserve an insufficient event model. The
successor projection must include causal and logical-creation payloads after this
cut. No new schema was committed. The published MTG-244 v8 integration is retained;
its layout is not evidence of complete execution-state representation. No human
judgment or new authorization is missing, and no weakened card definition was
added. The Task remains incomplete and is not ready for review.

## Remaining implementation and acceptance

Order: sync #427's current head, then event model, real cards, frozen-input
boundary, projection and admission, risk measurements.

0. Sync published MTG-244 `beaacaa19` before the event-model cut. Its schema files
   are unchanged from `1930146b0`, but it rewrites entry-replacement discovery in
   `proposed_event.rs` over `layer_views`, which this branch also restructured, and
   touches `triggers.rs`, `sba.rs` and `zones.rs`. Resolve that once, before adding
   provenance there. Measuring first would also time a repaired-away regression.
1. Complete the event-model correction above. Source inventory at `660021431`:
   `CounterChange` built at `resolution.rs:593,612,1753` and `decision.rs:1965`
   (effects), `tick.rs:256` (turn action, with the same `by: Some(active)` as an
   effect), and the `by: None` helper at `proposed_event.rs:1085`, whose callers
   need classifying. Eager token allocation is at `resolution.rs:1301,1391` and
   `zones.rs:301`.
2. Add the real Leaves from the Vine / Caradora / Doubling Season witness, with
   complete definitions and learning/observation coverage. Neither new card has a
   set file: there is no `rav.rs` or Aetherdrift file, so confirm first printings and
   Scryfall ids before authoring. Current `ModifyCounters` drivers use
   `count * multiplier + additional`; no registered card uses that replacement yet.
   The registered Leaves chapter with constructed drivers proves simultaneous
   placement, not those two complete cards.
3. Finish the hidden-state audit for unknown legacy Scry/LookAndSelect callers,
   preserving viewer-known inspected prefixes. `selection_sampling.rs` names neither
   decision and has no witness for them; `possible_worlds.rs` refreshes LookAndSelect
   and refuses other prompts explicitly. Private remembered subjects after their
   frame ended are closed by owned bindings. No claim of complete hidden-world
   coverage follows; new support for the frozen learned sampler needs an explicit
   successor contract, not post-sampling rejection.
4. Project pending batches (including precommit entrants), iteration progress,
   selected/this-way results, completion events and paid/rule callers. Paired states
   with identical current offers must distinguish consequential viewer-safe payloads
   and hide unavailable bindings. Cover native/Python inputs, learning tokens,
   actual model input, LLM, Academy, replay/fork and reconnect. Reject unrepresentable
   older inputs before inference, subject to the frozen-code boundary below. Overflow
   of any new row must be the typed `ObservationEncodeError::Capacity`: #426's opt-in
   recovery handles only that class, leaving version admission fatal. Recovery also
   discards the affected game, so overflow biases training away from deep pending
   states; report the overflow rate rather than treating recovery as neutral.
5. Extend the integrated MTG-244 successor schema after the event-model correction.
   [PR #427's contract](https://github.com/loopflowstudio/manabot/blob/1930146b08a1f1481af6e3b46a20f87d4a26a8fa/AGENTS.md)
   at `1930146b08a1f1481af6e3b46a20f87d4a26a8fa` publishes **v8**:
   v7 object/relation/shared header/member widths, 4096 × 3 effect details,
   detail kinds 11–19 for learning-token kinds 1–9 with signed i32 values split
   into 16-bit halves, and local fact kinds 11–14 for copy-exception programs,
   changed colors, keyword-counter timestamps and captured continuation programs.
   Continuation programs join by frame index + 1 with token kind/value/ordinal;
   v8 defaults to 512 local facts, v7 retains 128. This published layer layout
   is concrete integration evidence, not proof that MTG-241 batches, result
   bindings or completion payloads fit it. Combined schema/admission work remains;
   no competing version or layout is assigned here. Preserve v1–v7,
   fresh v6, saved omissions, capacities, original action-prefix indices/Commands
   and omitted-overflow-is-error. World/hash, capture boundaries and new conformance
   corpus remain unfinished.
6. Complete Jack Heart's [performance/learning-risk acceptance](mtg-241-risk-validation.md).
   Implementation owns bounded ordinary-step/search probes, equivalent-work versus
   added-branching attribution, targeted mitigations and remeasurement, and paired
   policy/model and hidden-information witnesses alongside the new representation.
   Gate owns final-head same-runner measurements, full debug Rust/Clippy, affected
   Python/headless Academy, current-world admission and Engine Budget. These checks
   cannot substitute for the unfinished implementation or its targeted risk work.
   No speed or learning claim follows from the refactors; no experiment was launched.

The real-card counter witness, frozen-input boundary, retained-payload projection,
version admission and bounded risk measurements remain implementation work. This
Task is not ready for review or completion.

## Frozen obligation codes — open defect, 2026-11-04

Observed in source at `660021431`; no runtime witness yet. `main` assigns live codes
5 (`SelectMilled`), 6 (`SearchBasicLand`/`SearchLibraryToHand`) and 7
(`OptionalDiscardDraw`). This branch routes all of them to 28 and adds 27 and 29.
`validate_policy_input` never reads `frame.decision`, and `ContinuationEncoder` feeds
column 28 to a linear layer as a raw scalar, so nothing bounds it. Every v2–v7 input
therefore changes meaning on ordinary searches with no error, including fresh v6 and
the v7 TLA protocol. Only the handlers were retired, not the codes' frozen meaning.
Blanket rejection cannot repair it: searches are common, so rejecting code 28 would
make fresh v6 unplayable.

Draft, not accepted: frozen versions keep writing the code they always used for a
migrated selection, with the selection's legacy class retained on the decision, and
28 is written only by the successor. Obligations with no frozen equivalent (27, 29)
are rejected by admission on older versions. Add a paired witness per version.
Assumption and alternatives are in [questions](questions.md); review may overrule it.

## Non-Rust consumers — 2026-11-04

Effect migrations reach Python through `card_definitions_json`, with no compile-time
guard. `manabot/sim/heuristic.py` still matched the deleted `SearchBasicLand`, emptying
`abilities().prompt`, so the heuristic anchor stopped cracking Fabled Passage and
Prismatic Vista. CI-selected `test_cracks_fetch_lands_when_offered` passes on `main`
and failed here. Fixed and parametrized over both cards; `main`'s source yields exactly
those two. The heuristic's source digest moves, as its anchor identity intends. It is
the only live reader of a retired name and the only `card_definitions_json` consumer.
Any further `Effect` change, including the event-model cut, needs that test.

`tests/sim` also has three failures outside CI's selection that also fail on a `main`
baseline (`fe83a6b06` Python, its `.so` from the main checkout), so they are not
caused here: `test_rollout` feeds random `facts` validity, the seed-197 checkpoint
has no world binding, and `test_conditional_search`'s frozen digest had already
drifted. This branch moves that digest again, consistent with the serialized state
change. None was repaired; they are recorded so gate does not attribute them here.

The CI "main" pytest group has four open failures that would fail PR CI:
- `test_every_observed_field_has_llm_text[Permanent|ContinuousEffect]`: `colors`,
  `keyword_counter_timestamps`, `program` and `description` are missing from the test's
  `_TEXT_FIELDS`, also at `1930146b0` and `beaacaa19`. Inherited from MTG-244, whose CI
  had not completed there. Text renders `colors`, the timestamps and `description`;
  whether `program` is covered only through `description` is unchecked. If the sync
  does not bring a repair, this Task owns it for its own PR.
- `test_survival_orders_cross_consumers[*]`: `trigger-order-w17.json` expects `main`'s
  root digest. It is a refreshable fixture, not frozen evidence, and #427 moves it too,
  so `scripts/refresh_academy_fixtures.py` belongs after the final rules change.

## Integration and evidence

MTG-242 owns payment resources/search; MTG-243 naming; MTG-244 layers. Task reads
on 2026-11-04 (late) show MTG-242 implementing with no PR, #424 at `55cde4796` with
Integration failing, and #427 republished at `beaacaa19` with CI pending. MTG-242
stops on the MTG-241/244 dependencies after its independent work, so this Task is on
its path. Publication does not establish integration. Task comments
carry FrameCompletion, the paid-output protocol and
codes 27–29. MTG-244 received the schema coordination request and
`ZoneSelection`/`LibraryEnd`/public-entry interfaces (comment
`dffd3f70-4dc9-462c-8fbc-6706b162ef76`). Other checkouts were neither edited nor
copied. Main's casting-cost/X fixes and buffer/overflow contracts remain intact;
the MTG-213 first-64 pilot exception does not authorize truncating new choices.

New shared evidence changes the integration risk, not this Task's ownership.
Jack Heart's MTG-244 CI-repair direction reports matching checked counts/digests
but search total 168.192s → 453.756s, worst reply 4.909s → 15.349s and throughput
12817 → 5654.8 steps/s at `1930146b0`. These are reported sibling measurements, not
an independently reproduced MTG-241 cost, and `beaacaa19`'s repair has no CI timing
yet. Combined-head measurements must separate inherited cost from this branch's
overhead. Only a comparison against main establishes net cost; comparisons between
already-modified cuts cannot. No baseline reset, accepted cost or speedup claim is
implied.

Jack Heart has since raised the speed bar on both siblings: MTG-244's direction makes
10% a minimum guard rather than the objective, and MTG-242's (2026-11-05 UTC) says a
6.7–8.0% slowdown is "not good enough". Neither was given to this Task, so MTG-241's
acceptance stays the configured budget plus attempted mitigation, but being merely
inside 10% may not satisfy review. Report the net number rather than the pass/fail.

Jack Heart's publication handoff makes [#425](https://github.com/loopflowstudio/manabot/pull/425)
the source of the approved PRINCIPLES.md/README guide, now integrated through
`d83376181`. The published guide supersedes the older embedded draft and links PRs
#42, #348, #353, #289, #412 and the speed investigation. Research publication does
not complete this Task. Its [milestone](../docs/research/standard-2026-11/milestone.md)
places execution/results beneath later costs, history and object-lifetime work;
those proposed cohorts and separately authorized training Tasks do not expand this
Task into another executor, content rollout or experiment campaign.

Rules evidence previously checked:
- [Guilds release notes](https://magic.wizards.com/en/news/feature/guilds-ravnica-release-notes-2018-09-20):
  surveil triggers after completion, including an empty library.
- [Aetherdrift notes](https://magic.wizards.com/en/news/feature/aetherdrift-release-notes)
  and [Foundations notes](https://magic.wizards.com/en/news/feature/foundations-release-notes):
  affected-controller counter ordering, including entry; (1 + 1) × 2 = 4 vs
  (1 × 2) + 1 = 3. These are behavior requirements, not complete card proofs.

Historical focused checks, exact diff hashes and their limits are retained at
`f1ce0ce13:scratch/mtg-241-composable-rules-execution.md`: nested iteration,
retained batches, hidden consumers and constructor compression; extension rebuilds,
text/parity and sampler rejection. They establish those slices, not full-head
acceptance, complete private-binding support or runtime/learning performance.

Earlier binding/schema checks remain at `d7ad17df1` and at `660021431:scratch/mtg-241-composable-rules-execution.md`;
they do not certify later edits. Gate owns full suites and Engine Budget after
implementation completes.

Proposed Wave memory, not written (`.lf/wave/` is read-only here): effect migrations
need a Python-consumer check, since CI catches them only through selected tests; and
frozen meaning covers scalar vocabulary values as well as widths and embeddings,
because admission that checks only payload presence cannot see a remapped code.

Check (2026-11-04, `660021431` plus the heuristic fix): debug `cargo test --lib hidden_selection` → 11 passed; `tests/sim` → 164 passed, 3 failed (all also fail on `main`); CI "main" pytest group → 1067 passed, 1 skipped, 4 failed (above); Academy, long group, Rust suites and Engine Budget deferred to gate.
