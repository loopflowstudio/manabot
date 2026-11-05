# MTG-241 — composable rules execution

## Outcome and boundary

Jack Heart's accepted directive (2026-11-04): one PR through the code Graph,
stopping at interactive review; no merge, auto-merge, training or retained-evidence
rewriting. The Task remains **incomplete**. No human judgment is missing.
Current committed head: `c8ea23d33`, incorporating main `ee85d6fc8` and the
hidden consumer migration at `724d7437f`. The next implementation cut replaces
frame-only private binding ownership; its counterexample and design are below.
The current diff simplifies batch handoff and replacement bookkeeping only.

The selected Wave is `.lf/wave/coverage/`; its GOAL and full MEMORY were read.
Jack Heart's dated directive supplies this execution scope without reopening the
older chapter or authorizing experiments. The approved PRINCIPLES/README guide
came through PR #425 and remains intact. Earlier implementation and validation
history is preserved at `c8ea23d33:scratch/mtg-241-composable-rules-execution.md`.

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

Remaining deletion cut: replace bare remembered-reference transfers that discard
provenance, frame-only discovery as the exclusive binding authority, and the
no-suspended-frame shortcut for retained binding refresh. Preserve public delayed
captures, exact incarnation semantics, existing hidden-selection predicates and
saved-frame aliases. This requires the binding implementation below; do not polish
the predecessor. Distinct-land/named searches, Scry and LookAndSelect retain distinct
visibility and ordering contracts; their overlap still needs an audit.

## Ownership revision — 2026-11-04

**The preceding implement step stopped on a demonstrated architectural
counterexample.** No human judgment is missing; dependent projection work must
wait for the owned-binding correction. This is not review readiness.

A constructed `SelectCards(Hand, min=1, max=1)` creates an end-step delayed
trigger remembering `Selected`. A normal SelectCard Command completes the child
and root; no suspended frame remains. Both direct determinization and count-world
materialization (seed 0) then leave the remembered exact incarnation in Library,
although the capture selected a Hand object and no game action moved it. A later
Remembered effect would execute against that library object. Reproducer:
[completed-private-capture.patch](completed-private-capture.patch), applied to
`c8ea23d33`; the temporary test was removed from source after preserving its patch.
This is a constructed compositional witness, not a claim about a registered card.

Cause: `hidden_selection_requirements` visits suspended frames/decisions only;
`CreateDelayedTrigger` copies bare `Vec<ObjectRef>` and discards selection
provenance. `refresh_hidden_selections` returns immediately without a suspended
decision. Extending observation projection first would encode the wrong sampled
state faithfully. Pinning the original unknown object or merely refreshing the
current offer does not repair the conditional world measure.

Revised implementation design (within the accepted exact-object/sampling scope):
- Replace bare captured-subject storage with one owned binding representation
  carrying exact incarnation and the constraint/knowledge provenance needed by
  sampling. Do not introduce a parallel Game-level capture registry.
- Transfer that representation through `DelayedTrigger`, `PendingTrigger`,
  `TriggerPlacement` (remaining and placing), `TriggeredAbilityOnStack`, and
  resolving `EffectFrame`. Selected/moved captures must carry their actual zone
  and applicable definition/predicate constraints at capture, not infer an old
  selection predicate after movement or later characteristic changes.
- Discover all live private bindings before reassignment and use one joint exact
  assignment across aliases and active selections. Feed the same constraints to
  count-world weights, foretold allocation, direct sampling and materialization.
  Known exact bindings remain exact; unknown bindings preserve constraints without
  exposing their actual identities. Departed incarnations never follow a new one.
- Extend canonical hashing, capture/projection and input-version admission with
  the binding contract in the same cut. Exercise waiting, queued, placing, stack
  and resolving lifetimes, both viewers, known/unknown information and later zone
  movement. Then restore this reproducer as a passing ordinary regression and
  resume the retained-payload and performance work below.

Check: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path
managym/Cargo.toml --lib hidden_selection_completed_capture_stays_in_sampled_hand`
ran one debug test and failed as expected: direct/count-world zones were
`[Library, Library]`, required `[Hand, Hand]` (0.01s test execution). No runtime
fix for this capture defect or full-suite/performance claim follows. The preceding
shared read reported MTG-244's open PR #427 at demo; its comments supplied no
concrete schema reservation. No other checkout was changed.

## Remaining implementation and acceptance

1. Complete the real Leaves from the Vine / Caradora / Doubling Season witness,
   including complete definitions and learning/observation coverage. Current
   `ModifyCounters` drivers use `count * multiplier + additional`; no registered
   card uses that replacement yet. The registered Leaves chapter with constructed
   drivers proves simultaneous placement, not those two complete cards.
2. Finish the hidden-state acceptance audit beyond the migrated consumers.
   Current witnesses cover partial hand/library selections, restricted incompatible
   deals, saved callers/inherited bindings, public precommit entrants, both samplers,
   Commands/replay/stale rejection, divergent forks, frame remembered aliases,
   conditional foretell weights and materialization. Count-only learned sampling
   now rejects unsupported domains before inference. Audit private remembered
   subjects after their selecting frame has ended and unknown legacy
   Scry/LookAndSelect callers, preserving viewer-known inspected prefixes. No claim
   of complete hidden-world coverage follows; new support for the frozen learned
   sampler needs an explicit successor contract, not post-sampling rejection.
3. Project pending batches (including precommit entrants), iteration progress,
   selected/this-way results, completion events and paid/rule callers. Paired states
   with identical current offers must distinguish consequential viewer-safe payloads
   and hide unavailable bindings. Cover native/Python inputs, learning tokens,
   actual model input, LLM, Academy, replay/fork and reconnect. Reject unrepresentable
   older inputs before inference; obligation codes alone are insufficient.
4. Reconcile the successor schema with MTG-244 through ordinary integration. Its
   shared risk directive explicitly names **v8**, but shared comments supplied no
   concrete layout; this Task has not assigned a competing v8. Preserve v1–v7,
   fresh v6, saved omissions, capacities, original action-prefix indices/Commands
   and omitted-overflow-is-error. World/hash, capture boundaries and new conformance
   corpus remain unfinished.
5. Complete Jack Heart's [performance/learning-risk acceptance](mtg-241-risk-validation.md).
   Implementation owns bounded ordinary-step/search probes, equivalent-work versus
   added-branching attribution, targeted mitigations and remeasurement, and paired
   policy/model and hidden-information witnesses alongside the new representation.
   Gate owns final-head same-runner measurements, full debug Rust/Clippy, affected
   Python/headless Academy, current-world admission and Engine Budget. These checks
   cannot substitute for the unfinished implementation or its targeted risk work.
   No speed or learning claim follows from the refactors; no experiment was launched.

Complete the owned-binding correction first, then retained-payload projection and
version admission. The real-card counter witness and bounded risk measurements
remain implementation work. This Task is not ready for review or completion.

## Integration and evidence

MTG-242 owns payment resources/search; MTG-243 naming; MTG-244 layers. Prior
status snapshots remain in the committed plan; publication does not establish
integration. Task comments carry FrameCompletion, the paid-output protocol and
codes 27–29. MTG-244 received the schema coordination request and
`ZoneSelection`/`LibraryEnd`/public-entry interfaces (comment
`dffd3f70-4dc9-462c-8fbc-6706b162ef76`). Other checkouts were neither edited nor
copied. Main's casting-cost/X fixes and buffer/overflow contracts remain intact;
the MTG-213 first-64 pilot exception does not authorize truncating new choices.

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
`c8ea23d33:scratch/mtg-241-composable-rules-execution.md`: nested iteration,
retained batches, hidden consumers and constructor compression; extension rebuilds,
text/parity and sampler rejection. They establish those slices, not full-head
acceptance, complete private-binding support or runtime/learning performance.

Validation (2026-11-04, `c8ea23d33` + `git diff -- managym` SHA256
`abc3706b46ee1826d22887ae52844c782289542d7c6497cc9f384e409c3fe7c6`):
`CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml`
with `--lib resolution_tests` / `-p managym-cards --test cards aang_at_the_crossroads`
passes 41 / 2; root release maturin build and atomic cp312 installation pass;
`uv run --extra dev pytest tests/semantic/test_text_observation.py` selecting
`test_shared_mill_selection_reaches_text_commands_and_both_encoders` and
`test_joo_dee_surveil_and_copy_play_through_the_text_decision` passes 2;
`cargo fmt --check`, `git diff --check`, reproducer `git apply --check`, and
`lf context` pass (both context budgets satisfied). Gate owns full affected suites
and final-head acceptance; implementation owns the binding correction, complete
projection, real-card witness and bounded risk measurements. No performance or
learning-efficacy claim follows from these checks.
