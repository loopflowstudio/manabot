# MTG-241 — composable rules execution

## Outcome and boundary

Jack Heart's accepted directive (2026-11-04): one PR through the code Graph,
stopping at interactive review; no merge, auto-merge, training or retained-evidence
rewriting. The Task remains **incomplete**. No human judgment is missing.
Reconciled implementation head: `f1ce0ce13`, incorporating main `ee85d6fc8`,
the hidden consumer migration at `724d7437f`, and the batch handoff/replacement
bookkeeping simplification. The active implementation replaces frame-only private binding ownership;
its correction and focused evidence are below.

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
mill/discard independently of later moves. Shared selection remains code 28;
6/7 join the retired codes. Existing mill consumers retain their earlier contracts.

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
that ancestry. Existing obligation codes remain fixed: bottom 23; top 27 (mode =
committed prefix length); shared selection 28 (legacy 5 retired); replacement 29.
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
Canonical state serialization includes binding provenance. Viewer-facing references
still use native visibility and do not expose private binding identities.

The original defect and evidence are retained at
`f1ce0ce13:scratch/mtg-241-composable-rules-execution.md` and
`scratch/completed-private-capture.patch`. Its frame-only discovery and refresh
shortcuts have been removed. Focused checks now cover both viewers, six lifetimes,
alias agreement, privacy and subsequent zone movement. Full retained-payload
projection and policy admission remain unfinished; this change does not establish
learning completeness or all hidden-zone consumers.

## Remaining implementation and acceptance

1. Owned-binding correction implemented; finish its successor projection/admission
   with the retained payload cut below. Broader hidden consumer audit remains open.
2. Complete the real Leaves from the Vine / Caradora / Doubling Season witness,
   including complete definitions and learning/observation coverage. Current
   `ModifyCounters` drivers use `count * multiplier + additional`; no registered
   card uses that replacement yet. The registered Leaves chapter with constructed
   drivers proves simultaneous placement, not those two complete cards.
3. Finish the hidden-state acceptance audit beyond the migrated consumers.
   Current witnesses cover partial hand/library selections, restricted incompatible
   deals, saved callers/inherited bindings, public precommit entrants, both samplers,
   Commands/replay/stale rejection, divergent forks, frame remembered aliases,
   conditional foretell weights and materialization. Count-only learned sampling
   now rejects unsupported domains before inference. Audit private remembered
   subjects after their selecting frame has ended and unknown legacy
   Scry/LookAndSelect callers, preserving viewer-known inspected prefixes. No claim
   of complete hidden-world coverage follows; new support for the frozen learned
   sampler needs an explicit successor contract, not post-sampling rejection.
4. Project pending batches (including precommit entrants), iteration progress,
   selected/this-way results, completion events and paid/rule callers. Paired states
   with identical current offers must distinguish consequential viewer-safe payloads
   and hide unavailable bindings. Cover native/Python inputs, learning tokens,
   actual model input, LLM, Academy, replay/fork and reconnect. Reject unrepresentable
   older inputs before inference; obligation codes alone are insufficient.
5. Reconcile the successor schema with MTG-244 through ordinary integration.
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

Continue with retained-payload projection and version admission. The real-card counter witness and bounded risk measurements
remain implementation work. This Task is not ready for review or completion.

## Integration and evidence

MTG-242 owns payment resources/search; MTG-243 naming; MTG-244 layers. Shared
Task reads on 2026-11-04 show all three implementing, with #424 and #427 open
and no MTG-242 PR. Publication does not establish integration. Task comments
carry FrameCompletion, the paid-output protocol and
codes 27–29. MTG-244 received the schema coordination request and
`ZoneSelection`/`LibraryEnd`/public-entry interfaces (comment
`dffd3f70-4dc9-462c-8fbc-6706b162ef76`). Other checkouts were neither edited nor
copied. Main's casting-cost/X fixes and buffer/overflow contracts remain intact;
the MTG-213 first-64 pilot exception does not authorize truncating new choices.

New shared evidence changes the integration risk, not this Task's ownership.
Jack Heart's MTG-244 CI-repair direction reports matching checked counts/digests
but search total 168.192s → 453.756s, worst reply 4.909s → 15.349s and throughput
12817 → 5654.8 steps/s. Its layer-risk reference fetch also failed before timing;
MTG-244 owns those repairs. These are reported sibling measurements, not an
independently reproduced MTG-241 cost. Combined-head measurements must separate
inherited cost from this branch's overhead and still satisfy the existing budget.
MTG-242's newly required comparison against main reinforces the need for a net
cost baseline: comparisons only between already-modified cuts cannot establish
the cost of the whole change. No baseline reset, accepted cost or speedup claim
is implied.

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

Check (2026-11-04, owned-binding implementation over `f1ce0ce13`):
`CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 --lib hidden_selection --no-default-features`
passes 10 focused debug tests; `git diff --check` and `lf context` pass.
Final-head affected suites and Engine Budget remain with gate.
