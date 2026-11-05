# Forge, XMage and other engines: execution architecture compared with managym

Research date: 2026-11-04. Status: source analysis and recommendations, not an
approved redesign. Jack Heart requested a deep comparison of engine objects,
state machines, continuous effects and triggers, beyond the earlier authoring
survey. No external code was executed and no speed or conformance comparison
was performed.

Source pins: Forge `a4281f7e58b50542decce8971d81bca4307fd8b4`; XMage
`987fd4e9c73e9f113e235f8d8f98628f2d391667`; Magarena
`efa0aba85e681816a92b4938b28741d540384e35`; managym checkout `5dd11cec4`.
This report inspected the methods cited below, including their callers and
state owners; it does not infer complete behavior from class names.

## The most consequential difference

Forge and XMage commonly execute a rules operation by calling a player/controller
choice synchronously and continuing on the Java call stack. Forge's human
controller uses `showAndWait`; XMage's human player publishes a request and waits
on a response object. Local variables in the calling rule remain live while
the person answers. This makes many nested choices straightforward within one
running game thread. It does **not** give managym's contract of returning from
every decision with all continuation state available for cloning, replay and
policy execution. [Forge controller](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-gui/src/main/java/forge/player/PlayerControllerHuman.java),
[XMage human player](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage.Server.Plugins/Mage.Player.Human/src/mage/player/human/HumanPlayer.java).

Managym's [`Decision` and `EffectFrame`](../../../managym/src/flow/decision.rs)
instead persist the unfinished operation and return an `ActionSpace`. A larger
card library makes that architectural difference more important, not less.
Borrow their separation of semantic responsibilities; translate their local
loops and variables into explicit resumable state. Do not transplant a blocking
controller callback beneath the engine's shared decision authority.

| Boundary | Forge | XMage | Managym implication |
|---|---|---|---|
| Object identity | Card ID plus game timestamp; current-card lookup and LKI copies | UUID plus zone-change counter; separate card/permanent/spell objects | Keep exact `ObjectRef`; make live, LKI and prospective reads explicit |
| Choices | Controller calls within rules execution | Player calls within rules execution | Persist cursors, results and pending event state before returning |
| Characteristics | Remove/reapply static modifications across layers | Reset derived state and apply registered layered effects | Preserve one characteristics owner and revision cache while extending semantics |
| Dependencies | Probe candidate effects and construct a graph | Effect-declared dependency categories and ordered application | Add dependency semantics, not only more timestamp branches |
| Triggers | Active/delayed/waiting registries, later placement | Registered triggers, simultaneous events and APNAP placement | Separate occurrence capture from placement; preserve indexed dispatch |
| Replacements | Choose, apply, re-evaluate; applied-effect tracking | Choose, mutate event, re-evaluate; consumed effect/ability pairs | Replace one-pass ordering with a resumable proposal authority |
| Forks | Explicit object remapping and simulation reconstruction | Copy game/state and replace players for simulation | Retain exact full-clone semantics unless measured evidence favors another design |

## Objects and identity: mature engines still need an incarnation boundary

Forge's `Card` contains mutable characteristics, state/face information, ability
lists and many runtime relationships. `Game.getCardState` locates the current
card corresponding to an older reference; `equalsWithGameTimestamp` checks a
stronger identity. The game's zone-change LKI table is keyed by card ID and
timestamp. Thus a Java object reference or card ID alone is not sufficient for
an exact rules object. [Card](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-game/src/main/java/forge/game/card/Card.java),
[Game](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-game/src/main/java/forge/game/Game.java).

XMage makes this explicit in `MageObjectReference`: UUID plus zone-change
counter, with separate current-card/permanent and LKI accessors. Constructors
that derive identity from an ability are deprecated with a bug reference;
some compatibility methods also permit weaker counter matching. The useful
lesson is to make reference strength visible at the API, not to copy every
fallback. [MageObjectReference](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage/src/main/java/mage/MageObjectReference.java).

Managym already has a cleaner separation between immutable definitions, physical
card storage, battlefield permanents and exact incarnations. Its current
weakness is inconsistent *reading* of those facts: the local audit found
live-source-only power lookup despite available LKI, and prospective entry
conditions reading owner rather than entering controller. The next improvement
is a small contextual query interface, not replacing IDs or adding another
universal game-object hierarchy. Preserve distinct contracts for printed values,
current layered values, captured event facts, departed-object LKI and prospective
entry. Migrate and delete the inconsistent helper paths.

## Trace 1: priority, resolution and triggered abilities

**Forge:** `PhaseHandler.mainLoopStep` runs state-based checks, asks the priority
player for a spell/ability, and tracks which player first needs priority again.
Once everybody passes, an empty stack advances the phase; otherwise
`MagicStack.resolveStack` resolves its top entry. The pre-priority fixed point
repeats state checks while waiting triggers are added to the stack. This is a
recognizable central turn/priority owner, not one state machine per card.
[PhaseHandler](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-game/src/main/java/forge/game/phase/PhaseHandler.java),
[MagicStack](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-game/src/main/java/forge/game/zone/MagicStack.java).

`TriggerHandler` separates active, delayed and waiting triggers. It can retain
already-collected occurrences and their controllers, and stack placement orders
them later. That separation is valuable. A notable operational shortcut is that
the inspected handler stops deferring after 9,999 waiting entries and runs
further triggers directly. Its comment argues side effects are unlikely; this
is not an acceptable substitute for an explicit capacity policy in managym.
The observation is about that source branch, not a reproduced Forge defect.
[TriggerHandler](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-game/src/main/java/forge/game/trigger/TriggerHandler.java).

**XMage:** `GameImpl.playPriority` invokes `checkStateAndTriggered`, updates
effects, then calls `player.priority`. After an action it handles simultaneous
events before clearing short-lived LKI. All-pass resolution calls the stack
object's `resolve`, reapplies continuous effects and resets passing state.
`checkStateAndTriggered` repeats SBAs before trigger placement; `checkTriggered`
walks players in APNAP order and asks each controller which waiting ability to
place. These are concrete ownership boundaries worth retaining in our
stabilization design. [GameImpl](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage/src/main/java/mage/game/GameImpl.java).

XMage snapshots the registered trigger list for each event, filters by event
type, and has optional integrity checks for order, omitted old triggers and
incorrectly processing newly added triggers. Those checks express useful
invariants. The inspected path scans registered triggers; managym's event-kind
index should remain an advantage rather than being replaced with that scan.
[TriggeredAbilities](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage/src/main/java/mage/abilities/TriggeredAbilities.java).

**For us:** preserve [`stabilize_before_priority`](../../../managym/src/flow/stabilization.rs)
and [`TriggerIndex`](../../../managym/src/flow/triggers.rs). Extend their existing
authority with explicit occurrence snapshots, scoped history and complete
placement choices. A trigger match is not stack placement; neither grants
priority during a resolving instruction. Test simultaneous source departures,
ability grants/removals and triggers created during placement. Keep debug
recomputation of the index as its vocabulary grows.

## Trace 2: continuous effects, dependencies and invalidation

**Forge** clears prior static contributions, gathers active statics and processes
layers. It retains affected-object sets across layers, discovers newly granted
statics, and reevaluates ordering. Its dependency routine temporarily applies
another candidate, observes changed existence/affected sets/results, undoes it,
then builds a graph; cycles fall back to timestamp ordering. This demonstrates
that dependency ordering is a semantic computation, not a permanent sort key.
The current result-change helper only supplies a nonempty comparison for the
control layer, so the elaborate machinery is not itself proof of complete
dependency conformance. [GameAction](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-game/src/main/java/forge/game/GameAction.java).

Its modification ledger removes effects by layer and affected card before
reapplication. This keeps one broad layer authority, but correctness depends on
being able to undo every contribution. The large continuous-ability executor
then interprets many independent script parameters. Managym can preserve typed
`Modification` values and a derived view rather than adopting mutable undo
bookkeeping or a string-parameter interpreter.
[StaticEffects](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-game/src/main/java/forge/game/StaticEffects.java),
[StaticAbilityContinuous](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-game/src/main/java/forge/game/staticability/StaticAbilityContinuous.java).

**XMage** resets players, battlefield and combat-derived state before applying
effects. Its evaluator handles copy/face-down effects, repeats control effects
when control changes, refreshes effect lists after several layers, and iterates
ability grants because a granted ability can introduce another effect. This is
broader semantics than managym's deliberately limited layer subset. It also
means repeated rebuild work whose cost cannot be inferred from the class
structure alone. [GameState](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage/src/main/java/mage/game/GameState.java),
[ContinuousEffects](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage/src/main/java/mage/abilities/effects/ContinuousEffects.java).

XMage's dependency relations are chiefly declared through `DependencyType`
sets on effect implementations, with helpers for common dependencies. That can
avoid probing arbitrary state transformations, but authors must describe the
dependencies correctly. Forge's dynamic probing and XMage's declared metadata
are two different strategies, each with an evidence burden.
[ContinuousEffectImpl](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage/src/main/java/mage/abilities/effects/ContinuousEffectImpl.java).

**For us:** extend [`Game::characteristics`](../../../managym/src/flow/statics.rs)
into one dependency-aware layer authority while retaining revision caching and
private layer-stage reads. Add actual required layers, same-layer dependency
reevaluation and grant/removal closure using discriminating cards. Do not cache
one timestamp order and assume new predicates cannot change it. Candidate read
and write metadata can narrow dependency work; a slow independently evaluated
debug path should verify the optimization. This hybrid is a proposal, not a
proven algorithm or a reason to expose Forge-style mutations to callers.

## Replacements are a second state machine, not an optional callback

Forge chooses the affected player/controller, gathers applicable replacements,
asks for an order, marks the selected effect as used, applies it, then recurses
over an updated proposal. XMage similarly loops, retaining consumed
effect/ability pairs and reevaluating applicability. Both distinguish modifying
an event from replacing it completely. That repeated selection is the core
behavior missing from managym's current fixed-order pass.
[Forge replacement authority](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-game/src/main/java/forge/game/replacement/ReplacementHandler.java),
[XMage replacement loop](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage/src/main/java/mage/abilities/effects/ContinuousEffects.java).

Forge also creates a prospective entering-card view when discovering entry
replacements, applying relevant statics before testing eligibility. This is an
important distinction from both current battlefield characteristics and owner
metadata. For managym, the pending event must carry entering controller and
prospective characteristics through the same read authority used by replacement
predicates. Preserve event batching and applied-effect identity across pauses.

Their synchronous loops use temporary maps and mutable effect flags; copying
those into managym would leave necessary data on a vanished call stack. Instead,
a typed pending proposal should own the current event, eligible chooser,
applied replacements, batch cursor and completion. Replace the existing one-pass
loop when consumers migrate, and preserve an allocation-light no-choice path.

Controller conveniences also need scrutiny: both inspected human paths can
collapse replacement options with identical display text. Text equality does
not establish semantic identity under managym's shared decision contract.
Automation must remain identifiable, and a model must receive every meaningful
choice. This is a compatibility warning, not a claim that every such collapse
produces a different result.

## State ownership, history and search copies

XMage stores reusable history observers in `Watchers`, deduplicated by a key;
they receive events and reset at their defined boundary. That is useful evidence
for separating history accumulation from card execution. Managym should use
typed, scoped facts and event indexes rather than adding one field or replaying
the full event log for each card. The inspected watcher dispatcher still visits
every registered watcher; adopt the ownership idea, not necessarily its scan.
[Watchers](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage/src/main/java/mage/watchers/Watchers.java).

Forge's simulation copier constructs a new game and remaps players, cards,
remembered objects, copied states and optional stack contents. It suppresses
some triggers during reconstruction, then refreshes state. The source contains
explicit omissions, including a player-history TODO and skipped zone-less
remembered cards. Therefore it is evidence of the difficulty of copying a rich
graph, not an exact-fork reference implementation for us.
[GameCopier](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-ai/src/main/java/forge/ai/simulation/GameCopier.java).

XMage's game copy includes LKI, cards and `GameState`; the state copy includes
effects, triggers, delayed abilities, watchers, combat and simultaneous events.
Its inspected MCTS path then substitutes simulated players and randomizes
opponent hands/libraries. That simple randomization is not a substitute for
managym's viewer-knowledge-constrained possible worlds. Other AI paths were not
audited. [Game copy](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage/src/main/java/mage/game/GameImpl.java),
[state copy](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage/src/main/java/mage/game/GameState.java),
[MCTS simulation](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage.Server.Plugins/Mage.Player.AIMCTS/src/mage/player/ai/MCTSNode.java).

Retain managym's cloneable rules state, exact continuations, shared immutable
definitions and cache-carrying clones. Measure step cost, search total/tail and
clone/allocation contributions independently. No reviewed source establishes
that a Java graph, persistent collection, undo log or generic interpreter would
be faster for our workload. Representation breadth is not performance evidence.

## Magarena and DeepScry: additional useful contrasts

Magarena is closer to managym's external choice boundary: `MagicGame` advances
through an event queue until an event has a choice, executes a supplied result,
and continues. Its state copy remaps event, stack, trigger and static objects.
However, `MagicEvent` uses generic result arrays and separate artificial-choice
generation; this is not managym's typed, shared legal-action contract.
[MagicGame](https://github.com/magarena/magarena/blob/efa0aba85e681816a92b4938b28741d540384e35/src/magic/model/MagicGame.java),
[MagicEvent](https://github.com/magarena/magarena/blob/efa0aba85e681816a92b4938b28741d540384e35/src/magic/model/event/MagicEvent.java).

Magarena caches derived permanent characteristics and refreshes its trigger
registry during updates. Its layer enumeration includes an extra conditional
ability stage and historical counter/switch subdivisions. This is a reminder
that a layer-shaped enum is not a current-rules conformance proof. Its event
queue is a useful comparison; its exact update and dependency semantics would
need further directed review before adoption.
[MagicPermanent](https://github.com/magarena/magarena/blob/efa0aba85e681816a92b4938b28741d540384e35/src/magic/model/MagicPermanent.java),
[MagicLayer](https://github.com/magarena/magarena/blob/efa0aba85e681816a92b4938b28741d540384e35/src/magic/model/mstatic/MagicLayer.java).

DeepScry remains documentation-only evidence from the [earlier review](external-engines.md):
owner-controlled cache invalidation and snapshots at completed choice boundaries
fit our direction. Its repository was unavailable; this follow-up's requests
for the cache/snapshot/determinism pages also returned access errors. No deeper
claim about its current objects, layers or trigger implementation is justified.

## Recommended architecture work

The comparison supports three cohesive changes, each validated by real cards:

1. **Resumable execution and proposals.** Give sequence/iteration, selections,
   payments and replacements explicit scoped results and completion ownership.
   Migrate their duplicated continuations and delete the old one-off paths.
   Preserve the single native authored representation.
2. **Contextual reads and layer dependencies.** Consolidate live/LKI/prospective
   queries, extend the existing layer authority, and retain precise invalidation
   plus debug recomputation. Retire owner-only/live-only shortcuts and competing
   characteristic readers; do not add a parallel layer interpreter.
3. **Occurrence and history ownership.** Keep event-kind indexing, capture
   exact source facts, distinguish matching from placement, and add reusable
   scoped history. Remove per-card duplicate counters only when all consumers
   have migrated and frozen observations retain their meanings.

Measure these changes against both the retained TLA cohort and bounded new
Standard cases. The desired improvement is fewer repeated interpretations and
less repeated work inside authoritative owners. It is not simply fewer enums,
smaller files, or matching another engine's object model.

The acceptance boundary should follow those owners. For replacement proposals,
use a case where applying one replacement makes another newly applicable, a
case where it makes another inapplicable, and a replacement that pauses for a
choice before resuming the outer instruction. Fork at each pause and verify the
same remaining legal actions and outcome. For layered characteristics, use two
effects whose ordering changes applicability, an ability grant that supplies a
second static ability, and a source leaving while its captured effect remains.
For triggers, use a simultaneous batch in which a trigger source and several
affected objects leave together, then vary the controller's placement order.
These cases distinguish semantic coverage from a large number of happy-path
card examples; they also specify precisely what each migrated owner must retain.

The performance boundaries are equally concrete. A trigger addition should
increase work for its relevant event kinds, not every event. A tapped or damaged
object should not invalidate unrelated characteristic calculations unless an
actual layer predicate reads that fact. A dependency graph should be scoped to
active effects and their potential interactions, not the whole card catalog.
A replacement-free action should not allocate the machinery required for a
multi-choice replacement chain. Search copies must include every pending cursor
and occurrence while sharing immutable definitions. None of these properties
requires a speculative rewrite of all state storage. They give the existing
release benchmarks and debug recomputation meaningful questions to answer as
new Standard mechanics arrive, without importing external engines' workloads or
assuming their card counts establish comparable throughput.

Validation: pinned primary-source inspection and local architecture comparison;
no external builds, tests, games, benchmarks or production changes.
