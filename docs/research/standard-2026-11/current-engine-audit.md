# Current engine audit for Standard expansion

Research date: 2026-11-04. Source revision: `5dd11cec4100aa0e1f4a23c063dfd5778e2ba28f`.
Jack Heart requested identification of existing shortcuts and architectural debt,
including larger refactors that would make a growing card file more composable
and keep both search and ordinary execution fast. This report is a source audit,
not a claim of full rules conformance or a proposal to rewrite the engine.
No production code, evidence, world identity, or runtime admission was changed.
No games, builds, benchmarks, or tests were run for this report.

Acceptance clarification from Jack Heart, 2026-11-04: bugs are expected, and a
generally working engine that supports training is a good outcome. The important
risk is later invalidating conclusions believed learned. Findings below are not
a collective gate on training or card admission. Prioritize material strategic
distortion, training disruption, recurring implementation cost and measured
runtime cost; scope any discovered defect's impact on retained conclusions.

## Conclusion

The strongest next architectural investment is **composable, resumable rules
execution over exact objects**, joined to a real replacement-choice pipeline.
The engine already has the right foundations: native typed definitions, verbs
over selectors, exact incarnation identity, centralized layered characteristics,
and an authoritative decision surface. Those foundations are unevenly applied.
Several recent card-specific composites still require a new effect, decision,
handler, and projection path for each wording variation. A few shared queries
also fall back to early assumptions even though richer state now exists.

Do not measure progress by shrinking the effect enum alone. Distinct Magic
operations deserve distinct semantics. The desired multiplier is that a new
combination of selection, quantity, player, timing, and destination works through
existing owners, including when it suspends, replaces an event, loses its source,
or is cloned for search.

## 1. Repair inconsistent exact-object reads before multiplying their consumers

**Source-confirmed current defect: source power loses LKI.**
[`effect_value`](../../../managym/src/flow/resolution.rs) at lines 40–58 has an
LKI fallback for `TriggeringPower`, but `SourcePower` uses
`source_permanent_id(frame).map_or(0, ...)`. The latter helper at line 2080 only
returns a currently present permanent matching `frame.source_ref`.
[`ObjectLki`](../../../managym/state/src/game_object.rs) already retains power.

This affects existing definitions, not a hypothetical future extension:
Firebending Student and Fire Lord Zuko use source power for their attack-triggered
mana; Combustion Man uses it for its attack trigger's damage choice
([definitions](../../../managym/cards/src/tla.rs), lines 2220, 3420, 3829).
If the source leaves in response, the implementation reads zero. The required
fixture is an attack trigger with a known, modified power, source removal before
resolution, and assertions on the resulting mana or damage. Add a leave/re-enter
case to ensure the later incarnation's power is never substituted. This was not
runtime reproduced here; the contradiction is visible in the current call path.

**Source-confirmed current defect: entry replacements inspect the owner instead
of the entering controller.** In
[`collect_replacements`](../../../managym/src/flow/proposed_event.rs), line 262,
`entering_controller` is unconditionally `self.state.cards[*card].owner`.
`ZoneEntry` already carries a controller override and the eventual commit uses
entry terms. Both conditional entry counters and conditional tapped entry read
the incorrect player. Cruel Administrator's raid entry replacement, for example,
must use the controller under whom it enters. A bounded fixture can give owner
and entering controller opposite attack histories and reanimate the card under
the nonowner. The five TLA lands with `EntersTappedUnlessControls` expose the
same underlying mismatch when entering under another player's control.

These are arguments for one explicit contextual read interface, not a generic
"read anything" framework. It should distinguish current characteristics,
captured exact-object LKI, and prospective entry characteristics/controller.
Delete the duplicated live-or-zero and owner-only assumptions when consumers
move. Preserve the existing `Game::characteristics` authority and layer-internal
read stages; do not route layer computation recursively through the public view.

## 2. Make composition preserve suspension and results

**Verified architectural restriction:** `Effect::ForEachTarget` does not compose
with effects that ask a question. Its handler
([resolution.rs:1824](../../../managym/src/flow/resolution.rs)) manually constructs
a subframe, executes its queue, and only `debug_assert!`s that each returned
decision is absent. In release, a returned decision is discarded. The enum
documents the restriction, and this audit does not assert that an existing
registered card violates it. It is nevertheless a dangerous authoring surface:
adding a superficially ordinary inner `Chosen`, optional payment, or search can
silently change rules behavior in release.

**Verified structural debt:** `EffectFrame` has one mutable slot apiece for
`this_way`, `moved`, `created`, and `excess_damage`, plus a cloned
`VecDeque<Effect>` ([decision.rs:58–119](../../../managym/src/flow/decision.rs)).
These are useful for today's sequential programs, but nested iteration and
branching need explicit result lifetime. A subframe currently initializes several
of these slots from scratch. New "for each … if you do … that many …" cards
should not acquire unrelated bespoke handlers merely to preserve bindings.

Build a small execution-context stack: sequence/branch/iteration cursors,
scoped exact-object results, and one suspend/resume path. Keep authored programs
as native `Effect` values. An execution cursor is runtime state, not a second
card representation, authoring language, compiler, or persisted IR.
Keep semantic keyword wrappers where they emit required events; earthbend must
still record earthbending once per instruction even if its internals share
animation, counters, and delayed-trigger machinery.

The migration should first make iteration safely resumable, then migrate a real
cluster of composites. Candidates include `ReturnMilledThisWay`,
`OptionalDiscardDraw`, `MayDiscard`, and the several search/move combinations.
Use native choices with typed candidate sets, cardinality, visibility, order,
destination, and retained selected-object results. `SearchBasicLand` already
contains several independent axes (`to_hand`, `bottom_source`,
`untap_if_lands`, `count`, `subtype`), while separate variants handle generic
library-to-hand search and opponent splitting. This is concrete evidence that
the same selection machinery is being expanded in different directions.

Delete each migrated dedicated handler and its duplicate decision continuation;
keep directed card behavior tests and shared selection tests. Do not extract
one universal selector before proving private visibility, fail-to-find, reveal,
shuffle, ordered remainders, and simultaneous movement independently. The cost
is substantial: stack/continuation serialization, learner tokens, LLM text,
native/Python observations, Command identity, replay, and forks all participate.
An implementation-only refactor must preserve semantics; a changed decision
contract needs its proper version boundary.

## 3. Replace deterministic replacement shortcuts with actual rules choices

**Verified scope gap:**
[`apply_proposed_event`](../../../managym/src/flow/proposed_event.rs), lines
113–153, collects candidates once, applies them in a fixed order, then commits.
The sorter at approximately line 325 puts opponent damage increases last with a
comment describing that as preferable to the affected player. This does not
offer that player's choice. The limitations are explicit in the
[original replacement contract](../../rules/proposed-events-v1.md), but its
historical instruction to exclude incomplete cards has since been superseded by
the current definition-based playability policy.

The gap is larger than adding another `ReplacementEffect` variant. General
replacement processing needs resumable affected-player choices, applicability
re-evaluation after each transformation, per-event applied-effect tracking,
prospective characteristics for entry, and correct handling of event batches.
Draws, token creation, redirection, replacement-generated actions, and consumable
prevention would otherwise accumulate special entry points.

Retain the existing factual `GameEvent` stream: only committed events become
facts and trigger inputs. Replace the one-shot replacement loop and the
"preferred order" policy with a pending proposal state machine. Keep the
ordinary no-replacement path allocation-light. Add candidate indexing only
after establishing the dependency set; currently damage replacement discovery
walks every permanent slot and inspects its definitions.

Acceptance must include two applicable replacements whose orders differ,
an effect that becomes newly applicable after transformation, source departure,
and clone/resume during a replacement choice. Correctness will sometimes change
the workload and therefore its digest; that is not evidence of a speedup.
Existing cards remain playable while defects are fixed and tracked.

## 4. Treat finite catalog and payment assumptions as explicit scaling work

**Source-confirmed catalog-growth blocker:**
`NameAndExileFromTargetPlayer` builds a `BTreeSet<String>` of every catalog name
and one modal effect vector per name
([resolution.rs:1535–1559](../../../managym/src/flow/resolution.rs)).
The Rise of Sozin's second chapter uses it. This is linear work and allocation over the whole
installed catalog for one in-game decision, and produces one legal action per
name. Fresh native and Python policy inputs allow 512 actions; the Python
encoder explicitly errors beyond capacity
([observation.py:962](../../../manabot/env/observation.py)). A multi-set catalog
will cross this limit even in a small-board game. Native semantic legality and
policy encodability must be considered separately.

Simply raising a constant repeatedly does not solve the architecture. Investigate
candidate-based action scoring or an explicitly shared structured naming
decision, preserving the ability to choose every legal name and retaining
strategically equivalent meaning. Browser-only filtering or model-only
shortlists would violate the shared action authority. Any multi-step decision
design requires shared semantics and separate policy-version consideration.
Cache immutable naming candidates rather than reconstructing owned strings and
effect trees per resolution. Do not claim a chosen design until its decision
and observation costs have been measured.

**Verified payment vocabulary limits:**
[`ManaUse`](../../../managym/state/src/mana.rs), lines 35–53, encodes only
Lesson, Shrine, and Ally in a fixed `u8` mask. Unknown subtype strings disappear
in `filter_map`. This is insufficient for broad spending restrictions, and
silently accepting a new restriction's unknown vocabulary is particularly
unsafe. Replace it with typed restriction predicates/identities and deliberate
unsupported construction errors during migration, preserving frozen projection
meanings. Do not merely keep appending subtype bits.

The mana witness search is a valuable shared owner, but its current source model
uses each source once. A registry test at
[mana.rs:1247](../../../managym/src/flow/mana.rs) requires tapped or once-per-turn
chosen-output abilities and excludes other upfront sacrifice/waterbend costs.
Repeatable mana activations and new cost resources need explicit transitions,
termination reasoning, and resource reservations. Also audit bounded `u8`
amounts/X and two-color-only hybrid parsing before extending those vocabularies.
These are known modeling limits, not proof of a defect in each present card.

## 5. Preserve the good layer/cache architecture while widening its semantics

The header of [`statics.rs`](../../../managym/src/flow/statics.rs) explicitly
excludes layer 7d switching and CR 613.8 dependency ordering. Timestamp ordering
is not sufficient for every possible future interaction. Extend one layer
evaluation authority as actual set requirements arrive; do not create separate
special-case readers for combat, costs, observations, or card predicates.

`TrackedState` and debug recomputation are strong designs worth retaining.
However, the classification of writes is tied to the current rule vocabulary:
the comments at [game.rs:365](../../../managym/src/flow/game.rs) declare tap,
damage, and some other writes irrelevant to cached derivations. A new static
condition reading one of those facts changes that dependency. Each mechanic
must review its reads against invalidation, not just add an enum branch.
Use precise revision ownership, with the slow recomputation oracle preserved.

Two old ignored audit probes remain in
[`cr_608_resolution.rs`](../../../managym/tests/rules/cr_608_resolution.rs):
earthbend following a second zone change and deathtouch surviving an SBA check.
They must not be reported as freshly reproduced. Current source still reads
`deathtouch_damage` in lethal checks and clears it at cleanup rather than after
the relevant SBA check, so that concern remains source-supported. Conversely,
`source_object` now explicitly permits only the next incarnation for a departed
source; the old earthbend test must be re-run before calling its historical
failure current. This is why inherited audit status cannot substitute for
checking the code.

## 6. Make performance evidence cover the architecture being expanded

The retained [engine investigation](../engine-speed-2026-11.md) already found
that ordinary rollout work, rather than outer search-fork setup, dominated its
sample. It also records failed or inconclusive COW/undo and small-allocation
ideas. There is no evidence here supporting a wholesale state-storage rewrite.
Definitions are already shared through `Arc`; the catalog is not cloned into
each game. Preserve compact full-clone search unless new measurements identify
a different bottleneck.

The credible candidates are shared work elimination: payment inventory reuse
across one decision, fewer repeated characteristic traversals, immutable
instruction/candidate sharing across suspended frames, and indexed replacement
sources. These are hypotheses until measured. Payment inventories depend on
tap/use/life/mana/turn state and pending choices; a characteristics revision
alone is not a valid cache key. A future stable instruction cursor could avoid
cloning `Vec<Effect>` trees, but must retain copy/grant semantics and replay
identity.

Keep same-runner candidate/reference measurements for search total/worst reply
and ordinary `Env::step`. Add a small representative Standard cohort covering
newly costly mechanics, while retaining the established TLA cohort. Report
catalog construction/naming separately from hot per-step costs. Keep a
milestone anchor as well as the rolling reference to expose cumulative drift;
compare equivalent traces, and explicitly identify cases where rules fixes
prevent that comparison. No contention generators, unbounded random-game
campaigns, or additional worker pressure are justified by this audit.

## Recommended sequence and evidence

1. Repair the two contextual-read defects with directed native tests and shared
   observable outcomes; validate old ignored probes before relabeling them.
2. Build suspension-safe composition with a real card cluster and retire its
   duplicated composite continuations. Include ordered multi-card selection;
   `Surveil` currently retains existing top order and is only complete for the
   registered count of one (see ability.rs:951 and decision.rs:1948).
3. Complete the proposal/replacement choice authority before multiplying
   replacement-heavy card definitions.
4. Resolve catalog-scale naming and payment restriction/resource modeling before
   presenting all Standard definitions as policy-playable.
5. Extend layers and optimize demonstrated common work alongside each mechanic
   batch, using both correctness oracles and comparable performance evidence.

Review each batch for cards enabled, reused primitives, new rule concepts,
deleted duplicate execution paths, and search/runtime deltas. A new mechanic
can legitimately require new code; the failure signal is repeated code for
already-owned semantics. Module size alone is not that signal. Match setup
still draws seven without mulligans, and development admission is not Standard
tournament validation; those are separate named completeness boundaries rather
than accomplishments implied by completing the card file.
