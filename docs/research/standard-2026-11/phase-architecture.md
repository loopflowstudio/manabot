# Phase and managym: engine objects, execution and derived rules

Research date: 2026-11-04. Phase source is pinned to
`381dc910b946881420dc752120c02689d7d53b0f`; managym to `5dd11cec4`.
Jack Heart requested a deep architectural comparison, emphasizing problems
Phase's broader card coverage has exposed. This is source analysis, not a
conformance result or performance comparison. No external code was executed.

**This comparison does not establish that managym's overall architecture is
wrong.** Most differences reflect rules coverage that our smaller card file has
not yet required. Native typed definitions, exact object references, separate
observations, a shared characteristics reader, and cloneable decision state all
remain appropriate foundations. Phase's larger implementation is evidence of
additional requirements, not evidence that its overall shape is preferable.

There is a narrower structural concern: some of our execution boundaries assume
children finish synchronously or share one latest-result slot. Those assumptions
do not compose with nested decisions and replacements. That calls for a local
redesign of continuation/result ownership, not a new engine object model.
Phase itself distributes those responsibilities across many specialized owners;
we should solve the requirement without copying that distribution.

## Coverage gap or architectural problem?

The distinction is whether a missing rule fits behind an existing authority, or
whether the authority's contract prevents correct composition. Amount of work
alone does not decide this: implementing dependencies may be substantial while
still fitting our architecture.

| Finding | Classification | What should change—and what the evidence does not imply |
|---|---|---|
| Additional layers, durations, off-zone abilities and history predicates | **Coverage extension** | Extend existing rule types/readers and their projections. More variants alone are not architectural failure. |
| CR 613 dependency ordering and discovery of newly granted effect generators | **Substantial subsystem extension** | Add ordering/discovery algorithms behind `Game::characteristics`. This may require shared cross-object evaluation and different cache granularity; it does not invalidate the public derived-view design. The exact internal algorithm needs design and witnesses. |
| A new static reads tap state or becomes active in another zone | **Coverage-driven invalidation extension** | Update revision dependencies and all relevant mutation paths. The present dependency set is incomplete for the new mechanic; revision caching itself remains suitable. Repeated omissions would justify stronger dependency metadata. |
| `SourcePower` ignores available LKI; entry conditions use owner instead of entering controller | **Implementation defect / inconsistent authority use** | Consolidate contextual reads and repair callers. Existing identity and LKI types already supply much of what is needed; these bugs do not demonstrate the wrong object model. |
| `ForEachTarget` discards a child's decision in release; nested programs share unscoped result slots | **Local execution-contract redesign** | A parent must retain its cursor, scoped results and child completion through suspension. Adding one card-specific branch cannot provide this property generally. Keep `Effect` as the authored representation and `Decision` as the external boundary. |
| Replacement collection/order is a fixed synchronous pass | **Missing semantics plus a local protocol redesign** | Re-evaluation/order selection are coverage work. Propagating a replacement pause through costs, batches and outer instructions changes the operation contract. A larger replacement enum alone cannot solve it; integrate with the continuation redesign. |
| Broader trigger cases, event keys and off-zone sources | **Primarily coverage extension** | Preserve the index/matcher/placement separation. Logical batches across replacement pauses require explicit retained batch ownership, but there is no evidence to replace the whole trigger architecture. |
| Phase uses persistent collections; we clone compact state | **Performance alternative, not an architectural verdict** | Keep the measured implementation unless representative measurements show a net benefit from a different storage strategy. |
| Phase filters its full state for viewers; we build `Observation` | **Difference worth preserving** | Our explicit projection boundary is appropriate for viewer safety and policy inputs; broader coverage does not require exposing internal state. |

The strongest conclusion is therefore **extend the existing architecture, with
one coherent redesign around nested execution and replacement suspension**.
Dependency-aware layers are a second substantial workstream, but missing that
algorithm should not be described as having chosen the wrong characteristics
architecture. No inspected evidence warrants replacing native definitions,
adopting a parser/IR, merging observations into game state, or replacing full
clones wholesale.

## Structural comparison

| Boundary | Phase at inspected revision | managym today | Useful direction |
|---|---|---|---|
| Physical/rules identity | `ObjectId` storage plus incarnation; context/LKI snapshots | `CardId`/`EntityId`, typed `Incarnation` and `ObjectRef` | Keep exact references; make query time and provenance explicit |
| Mutable objects | One `GameObject` contains base and derived characteristics | Cards, permanents, stack objects and a derived `Characteristics` view | Preserve distinct responsibilities; widen semantics through shared readers |
| Fork storage | Persistent `ObjectMap`, per-object `Arc`, `im` zone/stack collections | Full game clone with shared immutable definitions and derived caches | Measure read/write/fork tradeoffs before replacing storage |
| Execution | `GameAction` × `WaitingFor`; `ResolutionStack` and pending owners | Actions, `Decision`, `EffectFrame` queue and suspended resolution | One composable suspend/resume contract |
| Layers | Reset/rederive mutable characteristics, dependency ordering, guarded incremental flush | Lazy cached views, layer/timestamp subset, revision invalidation | Add dependency semantics without duplicating characteristic readers |
| Trigger discovery | Conservative event-key index, off-zone/LKI paths and logical batch latches | Event-kind index and exact pre-batch source snapshots | Index candidate discovery; keep matching and occurrence identity authoritative |
| Replacement | Re-evaluating pipeline, pending choices, event/batch continuations | One collection/application pass with fixed ordering | A shared resumable proposal pipeline is required |
| Viewer state | Filtered `GameState`, including transactional projection | Separate native `Observation` and shared decision capture | Keep a dedicated viewer-safe contract |

## Objects, state ownership and forks

Phase's `GameObject` keeps its printed/base name, types, abilities and P/T beside
the currently derived values. Copies, face-down profiles, control effects and
grants update these through the layer machinery. Its stable storage ID is not
the complete rules identity: a zone move increments an incarnation. This matches
the principle already encoded more explicitly in managym's `ObjectRef`.
[Phase object](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/game_object.rs),
[managym identity](../../../managym/state/src/game_object.rs).

The breadth of Phase's departure and source-context records is instructive.
They retain characteristics, attachment relationships, trigger provenance and
event-specific facts. Managym should not respond by cloning an entire mutable
object into every frame. It needs named questions: current characteristics of
this exact object, last-known values of that departed incarnation, or facts
latched at a particular event. Those have different lifetimes. Our current
`SourcePower`/`TriggeringPower` discrepancy shows that a good identity type does
not automatically make every reader correct.
[Phase contexts](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/types/game_state.rs),
[current audit](current-engine-audit.md).

Phase's `ObjectMap` is an `im::HashMap<ObjectId, Arc<GameObject>, FxBuildHasher>`.
Clones share trie nodes and objects; mutable access uses `Arc::make_mut` on the
selected object. Sharing is deliberately excluded from game identity. This is
a concrete copy-on-write design, not an undo journal. It reduces copying when
few objects change, but adds map traversal and copy-on-write costs. A layer pass
that rewrites many objects also loses much of that sharing. The net effect is
a workload question; no speed ranking follows from the container choice.
[ObjectMap](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/types/object_map.rs).

Managym's compact full-clone path and cache-carrying clones already have measured
benefits. Preserve them unless profiling a representative Standard workload
shows copying is the limiting cost. Separate immutable program sharing from
mutable state storage: avoiding repeated effect-tree copies may be worthwhile
without replacing every game collection.
[Retained measurements](../engine-speed-2026-11.md).

## Execution and the ownership of a pause

Phase's public `apply` enters `apply_action_boundary`, validates the actor,
dispatches according to `GameAction` and `WaitingFor`, and finalizes the action.
The boundary retains a state snapshot for rejecting failed actions. That is
error rollback, not evidence that arbitrary game decisions should be undoable.
It also distinguishes the authenticated submitter from the semantic player for
turn-control and delegated-decision rules.
[Action boundary](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/engine.rs).

Its `ResolutionStack` stores frames such as ability continuation, repeat,
batch delivery, counter additions, discard, copy-token and post-replacement
drains. A frame's relationship to the currently exposed prompt is validated.
Opaque frame positions replaced positional searches and remembered vector
indices: finding a frame while reading does not confer authority to mutate it.
That is a strong transferable invariant. The large number of specialized frame
and pending kinds is also a warning about letting each mechanic invent its own
continuation protocol.
[Stack](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/types/resolution.rs),
[opaque positions](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/types/resolution/frame_vec.rs).

Managym's `EffectFrame` has a simple ordered queue, target-clause eligibility,
controller, exact source and result slots. Its simplicity is valuable, but its
non-suspending `ForEachTarget` and single latest-result slots cannot safely
express arbitrary nested iteration. This is the concrete local contract problem,
not an objection to queue-based execution itself. Extend it with scoped results
and explicit sequence/iteration ownership. Do not introduce a second authored card language;
runtime execution cursors are a different responsibility from card definitions.
[Current frame](../../../managym/src/flow/decision.rs),
[current resolver](../../../managym/src/flow/resolution.rs).

### Trace 1: a sequence whose zone move opens a replacement choice

| Stage | Actual Phase path | Required invariant for managym |
|---|---|---|
| Execute instruction | `resolve_ability_chain_step` / `resolve_chain_body` / `resolve_effect_step` | Distinguish completed work from a pending child; retain remaining instructions |
| Move a group | `zone_pipeline::move_objects_simultaneously_then` allocates a logical group and calls `deliver_batch` | One owner retains attempted objects, undelivered tail and completion |
| Apply replacements | `replace_event` enters `pipeline_loop`; `NeedsChoice` preserves staged entry/proposal state | A required rules choice is possible even inside an ordinarily automatic verb |
| Resume | Choice handling resumes the exact pending operation; batch completion runs once | Never recover a parent by guessing a frame index or repeating prior work |
| Settle | `run_post_action_pipeline_core` finishes due spell cleanup, collects events and reaches legal priority | A choice pause is not a priority window |

These are source paths, not an executed trace.
[Effect chaining](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/effects/mod.rs),
[batch movement](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/zone_pipeline.rs),
[settlement](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/engine_priority.rs).

Phase's post-action pipeline explicitly prevents ordinary object SBAs from
running while replacement choices leave an entering object incomplete. It also
retains trigger batches across choices and distinguishes spell-finalization
events from earlier resolution events. Our `stabilize_before_priority` is a
clearer central fixed-point owner. Preserve it; broaden its pending-operation
contract so a new replacement or payment choice cannot accidentally expose
priority, run premature SBAs, or lose the outer spell's finalization.
[Managym stabilization](../../../managym/src/flow/stabilization.rs).

### Concrete example: what “suspend” means

Here, suspend means returning a decision while preserving the unfinished rules
operation. It is unrelated to Magic's suspend keyword and does not grant a
priority window.

Suppose Leaves from the Vine's second chapter is resolving on two creatures,
and their controller also controls Doubling Season and Caradora, Heart of
Alacria. Each intended +1/+1 counter can become three counters (double, then
add one) or four (add one, then double). The affected controller chooses the
replacement order, as the [official Caradora ruling](https://magic.wizards.com/en/news/feature/aetherdrift-release-notes)
specifies. The engine must ask that question inside the resolving chapter.
Leaves already exists; the two replacement cards are expansion requirements.
This example is a source-level design witness, not a reproduced current game.

The chapter currently uses `ForEachTarget`. Its actual loop, with only the
subframe construction abbreviated, is:

```rust
for target in frame.active_targets() {
    let mut sub = EffectFrame { /* source, this target, effects, results */ };
    while let Some(effect) = sub.queue.pop_front() {
        let decision = self.execute_frame_effect(&effect, &mut sub);
        debug_assert!(
            decision.is_none(),
            "ForEachTarget sub-effects must not suspend"
        );
    }
}
```

The current API for proposed events returns `bool`; its replacement processing
is:

```rust
let replacements = self.collect_replacements(&event);
for replacement in replacements {
    self.apply_replacement(&mut event, &replacement);
}
self.commit_proposed_event(event)
```

Adding Caradora and Doubling Season as more replacement variants cannot make
that API return a decision. Once replacement processing can return one, the
outer loop must also preserve its progress rather than asserting the decision
away. Debug currently rejects a suspending child; release discards its result.
The stack-local `sub` and loop position do not survive a return.

The existing frame already keeps a queue, targets and source context, so this
does not require a different card representation. It requires retained nested
execution state, conceptually:

```rust
// Illustrative proposed types, not an implemented design.
struct ForEachFrame {
    targets: Vec<ResolutionTarget>,
    next_target: usize,
    active_child: Option<ExecutionFrameId>,
}

struct PendingCounterPlacement {
    event: CounterEvent,
    applied_replacements: Vec<ReplacementId>,
    parent: ExecutionFrameId,
}

enum Progress {
    Complete,
    AwaitingChoice(Decision),
}
```

The runner advances the top frame until completion or a choice; on a choice,
it retains all frames, returns the decision, and resumes the same pending
operation after the answer. Replacement applicability is recalculated after
each chosen transformation. For simultaneous placements, a logical batch must
also retain all members and completion: suspension must not turn one rules
event into separately settled actions, run premature SBAs, or place triggers
mid-instruction. These sketches show missing ownership, not a complete batch
algorithm.

That persisted choice boundary is also an RL requirement: a policy can answer
the choice, and search can fork there to compare both orders without rerunning
earlier effects or hiding the choice in automatic code.

## Continuous effects: broader semantics with a larger invalidation burden

### Trace 2: an entry or copy changes the active continuous effects

Phase's full layer pass resets live characteristics to base values, applies
copy effects, restores face-down characteristics, and refreshes static-source
discovery if copying introduced generators. It then gathers active effects for
later layers. Thus an ability created in an earlier layer can affect the same
pass, rather than waiting for another action. State-dependent durations are
retired after derivation, repeating while stored effects are removed.
[Layer evaluation](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/layers.rs#L3663).

The dependency solver groups modifications by the effect that produced them,
constructs provider/dependent edges, and applies a topological ordering. Strongly
connected components fall back to timestamp order within each dependency loop.
Referenced-provider ability grants use a separate live ordering path because
their dependencies can change as effects apply. This is meaningful machinery
missing from managym's documented layer/timestamp subset. It is not independent
proof that Phase implements every dependency rule correctly.
[Dependency ordering](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/layers.rs#L10050).

Its incremental flush has `Clean`, `Full`, and `EnteredObjects` states. An entry
can update a restricted recipient set only when eligibility checks prove that
other objects and effects remain unaffected; otherwise it escalates to the full
pass. This is the right shape for a safe optimization: explicit eligibility,
one fallback authority, and counters/tests proving the intended route occurred.
But the eligibility logic must grow with every new dependency. Source-zone,
life, tap, turn-step and population conditions can all invalidate earlier
assumptions.
[Flush boundary](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/layers.rs#L5622).

Managym should keep `Game::characteristics` as its public reader and private
layer stages as the internal computation boundary. Its revision counters and
debug recomputation oracle provide a good base. What must change is semantic
coverage and dependency accounting: layer switching, dependency ordering,
newly granted sources, off-battlefield characteristics and conditions reading
fields currently classified as irrelevant. Generalize the computation behind
the view; do not add separate “effective” readers for costs, observation and
combat. A global dependency graph may eventually be necessary for some cases,
but ordinary simple boards should retain a cheap path.
[Managym layers](../../../managym/src/flow/statics.rs),
[tracked writes](../../../managym/src/flow/game.rs).

## Triggers: occurrence, observation and placement are separate

### Trace 3: several permanents leave together

Phase's zone pipeline creates a logical zone-change group before delivery.
Trigger contexts can be latched at the appropriate pre/post boundary; each
member's replacement can pause while the group retains its intended semantics.
Collection produces pending contexts, combines ordinary and delayed triggers,
and orders controller groups APNAP. Controller ordering and trigger-target
selection are subsequent choices before stack placement. Trigger occurrence
therefore cannot be equated with immediately pushing an ability onto the stack.
[Trigger collection and grouping](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/triggers.rs#L15987),
[target placement](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/engine_stack.rs).

Its battlefield `TriggerIndex` is a conservative prefilter: event-key buckets
plus unclassified candidates feed the real matcher. Exhaustive event/trigger
classification prevents a new enum variant silently bypassing the index.
Layer completion rebuilds the authoritative index; incremental zone hooks are
optimizations. This resembles managym's event-kind index and debug checks, with
more discriminating keys and additional off-zone/latched-source paths.
[Phase index](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/trigger_index.rs),
[managym triggers](../../../managym/src/flow/triggers.rs).

The same Phase index documents an unresolved maintenance concern: the battlefield
list and an object's live zone can disagree. Its trigger lookup rejects such
candidates, but that only contains the trigger consequence; other battlefield
consumers need consistent state too. This is evidence for stronger ownership of
zone mutation, not for adding identical defensive filters everywhere. Managym
also maintains several related collections, so the transferable requirement is
one mutation authority plus invariant checks.

Phase's `CoDepartureObserverIndex` builds event/object lookup data once per
collection rather than repeatedly scanning for every departing observer pair.
That optimization directly fits our goals: preserve the same match semantics
and eliminate repeated work at the batch boundary. Extend indexing based on
measured collection costs, not merely the existence of more cards in the catalog.
[Co-departure index](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/triggers.rs#L9438).

## Replacement choices, visibility and simulation

Phase's proposal pipeline re-evaluates replacements and supports resumption.
It also uses `replacement_ordering_is_material`, a typed commutativity analysis,
to suppress ordering prompts it considers immaterial. That is an additional
semantic responsibility, not a free performance optimization. Our complete legal
decision contract must remain authoritative; adopting that suppression would
require a separate argument covering eligibility, side effects, information and
result identity. Even Phase's broader pipeline explicitly documents an omitted
choice for multiple finite prevention shields in a simultaneous damage batch.
Do not equate having a general pipeline with complete replacement semantics.
[Replacement implementation](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/replacement.rs).

Phase's viewer boundary projects a filtered `GameState`. Payment transactions
can expose a staged view only to the player entitled to answer, while other
viewers retain canonical state; internal frames and private fields are stripped.
That supports sophisticated interactions but creates a large redaction surface.
Managym's distinct native `Observation` is a useful architectural advantage:
internal state does not become public merely because it was added to the main
state type. Extend that explicit projection alongside each mechanic, including
turn-control and private-search audiences.
[Phase visibility](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/visibility.rs),
[managym observation](../../../managym/src/agent/observation.rs).

Phase also has invocation-local lifecycle facts for prospective simulation.
They report already-committed delayed-trigger installation and terminal outcomes;
they are not a second rules executor. The useful distinction is between
authoritative state needed to resume and derived evidence needed by a search or
diagnostic consumer. Do not let a search-only cache or receipt supply facts the
ordinary engine cannot establish.
[Lifecycle sidecar](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/src/game/lifecycle.rs).

## What this changes in our plan

1. **Local redesign:** make suspension composable before expanding replacement coverage. A parent
   retains its cursor, scoped results, pending cost and finalization across any
   child choice; completion and cancellation have explicit owners.
2. **Repair plus extension:** preserve one contextual-characteristics authority.
   Repair live/LKI/prospective reads and add dependency-aware layers behind it,
   with debug reference checks. The algorithm may be substantial; the public
   ownership remains sound.
3. **Coverage plus continuation integration:** preserve logical batch identity from proposed event through actual results
   and trigger occurrence. Add indexing at those stable boundaries.
4. **Preserve:** keep rule execution, public projection and search evidence separate. A richer
   engine does not justify a second interpretation for policies or Academy.
5. **Measure before redesign:** retain the existing storage strategy until a measured workload justifies
   changing it. Profile action generation, layer work, trigger collection,
   payment and observation separately before blaming cloning.

The first acceptance matrix should cross these boundaries, rather than adding
one isolated example per new primitive: a simultaneous departure with an
off-zone observer; a replacement choice inside an iteration; a source leaving
before its trigger resolves; and a dependency-changing effect followed by a
fork. Assert exact affected incarnations, retained parent progress, once-only
trigger occurrence and identical derived views before and after the fork.
Exercise both a genuine choice and a forced choice through the same continuation
owner. These are proposed directed fixtures, not results obtained in this review.
They would test the architectural promises at the seams where Phase's larger
implementation has needed explicit ownership and invalidation machinery.

The desired outcome is fewer independent ownership protocols as card breadth
grows. Phase supplies valuable semantic counterexamples and several good local
invariants; its specialized pending paths should be studied as migration risks,
not adopted as the organizing principle for our next few thousand cards.

Validation: traced the named functions at pinned revisions and compared current
local authorities. No builds, games, tests or benchmarks were run. No production
code or runtime contract was changed; implementation recommendations remain
proposals.
