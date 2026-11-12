# Card implementation retrospective

Research date: 2026-11-04. Inspected checkout: `5dd11cec4100aa0e1f4a23c063dfd5778e2ba28f`.
Requested by Jack Heart for the next card-file milestone: support arbitrary
Standard decks, initially prioritizing the post-rotation pool, while improving
composability and preserving or improving engine speed. This report proposes
delivery practices; it does not select implementation or authorize experiments.

Jack Heart clarified on 2026-11-04 that imperfect correctness is expected: a
generally working, trainable engine is valuable. The consequential failure is
later invalidating conclusions believed learned. This retrospective therefore
distinguishes ordinary defects from training blockers, systematic strategic
distortions, and repeated ownership problems; it does not require exhaustive
conformance before training or admitting cards.

The strongest recurring lesson is that rules, choices, information and execution
must agree sufficiently to support the claims made from a run. The largest
repairs removed multiple owners of those facts. The successful performance work
then removed repeated computation inside the shared authorities, without
changing choices.

## From the two-deck demo to Elemental

The July two-deck plan already had a good foundation: batch cards by shared
capabilities, build resumable decisions, and add observation support with the
mechanic. Its stages grouped triggers, costs/choices, and continuous effects
before registering the full roster. [PR #42](https://github.com/loopflowstudio/manabot/pull/42)
introduced suspended effect frames, cast-time choices, waterbend, ward and
targeting together. The plan survives at
[`87072cb298:wave/rules/01-two-deck-slice.md`](https://github.com/loopflowstudio/manabot/blob/87072cb298/wave/rules/01-two-deck-slice.md).

Two important qualifications appeared later. First, the initial Learn scope
deliberately omitted sideboards. [PR #188](https://github.com/loopflowstudio/manabot/pull/188)
added the missing outside-game choice, copy ownership, public knowledge and
policy focus. That was a real expansion of the chosen slice, not evidence that
all earlier Learn behavior was accidentally broken. Second, the July plan's
waterbend/Cub interaction was underspecified: tapping a creature to pay
waterbend is not tapping it for mana. [PR #332](https://github.com/loopflowstudio/manabot/pull/332)
explicitly corrected that distinction. Related words are not interchangeable
rules events.

The demo's reproducibility work was valuable. [PR #66](https://github.com/loopflowstudio/manabot/pull/66)
pinned the exact 41-card UR/40-card GW setup and asset identity instead of
silently normalizing the lists. Later receipts retained replayable Commands,
viewer checks and source identities. That made behavioral changes diagnosable.
But a stable demo was never evidence for all rules reachable with new decks.

Elemental's batches were sensibly organized: combat/Learn
([#328](https://github.com/loopflowstudio/manabot/pull/328)), sacrifice/landfall
([#331](https://github.com/loopflowstudio/manabot/pull/331)), and restricted mana,
graveyard casting and copying ([#332](https://github.com/loopflowstudio/manabot/pull/332)).
They exercised shared mechanisms with multiple real cards, exact-object
identity and privacy tests. Their costly inheritance was maintaining native
definitions alongside authored JSON, compilation, IR and learning schemas.
Native/compiled parity established agreement between paths sharing primitives;
it did not supply independent evidence that either followed the rules.

## Repeated failure patterns

### Legal offers and execution disagreed

The strongest concrete example is Prophetic Prism. A spell could be offered as
castable, but paying the filter's generic activation cost consumed its only
required green source. [PR #348](https://github.com/loopflowstudio/manabot/pull/348)
retained the original 34-action failure prefix and made probes and execution
share completion-safe payment plans. [PR #353](https://github.com/loopflowstudio/manabot/pull/353)
then preserved recursive payment continuations when Treasure paid Prism's
upfront cost. The failed paid baseline had stopped after four updates without
an updated checkpoint; its exact failure prefix was not recovered, so the
report appropriately did not claim perfect historical reproduction.

Target legality repeated the same ownership problem. [PR #354](https://github.com/loopflowstudio/manabot/pull/354)
unified offers, acceptance and resolution to fix a hexproof bypass. The next
repair, [#393](https://github.com/loopflowstudio/manabot/pull/393), established
the correct *time* for that shared check: resolution-entry eligibility persists
through later instructions and suspended decisions. Rechecking hexproof after
an earlier instruction granted it was also wrong.

**Keep:** executable witnesses, exact failing prefixes, one legality authority,
and retained continuation state. **Improve:** every new cost or targeting family
must test both “offered implies executable” and timing across suspension,
source departure and nested payment. Shared code alone does not establish the
right temporal contract.

### Strategic choices were omitted or indistinguishable

Learn's sideboard choice was followed by broader representation repairs.
[PR #289](https://github.com/loopflowstudio/manabot/pull/289) showed that distinct
targets, modes, generic card choices and abilities could reach the policy with
identical inputs. No training run could recover distinctions erased by the
encoder. The repair tested learning either preference and reproduced the ties
by erasing the repaired fields.

[PR #386](https://github.com/loopflowstudio/manabot/pull/386) replaced fixed
trigger enqueue order with a controller decision. Its Inspector/Bastion witness
made the consequence concrete: one order survives, another loses. [PR #415](https://github.com/loopflowstudio/manabot/pull/415)
restored player-selected bottom ordering for Accumulate Wisdom. Both changes
had to reach native Commands, policies, Academy, replay and reconnect.

**Keep:** native authority for every strategic choice, exhaustive observability
guards, and both-preference fitting. **Improve:** review each sentence for
chooser, timing, cardinality, order, decline and information before implementing
its happy path. A UI demonstration or random game cannot expose every removed
choice. Trivial one-action decisions should still be skipped natively.

### Card-shaped representations multiplied

Optional mana and Waterbending Lesson originally had separate paths.
[PR #344](https://github.com/loopflowstudio/manabot/pull/344) replaced them with
typed payment progress and paid/declined branches. [PR #403](https://github.com/loopflowstudio/manabot/pull/403)
removed obsolete compiler/world machinery and replaced per-card-definition
fields with `StaticAbility` and `AdditionalCost` variants. [PR #412](https://github.com/loopflowstudio/manabot/pull/412)
separated verbs from `ObjectSelector` and fixed/computed amounts, deleting
duplicated object-specific effects. T6's [PR #417](https://github.com/loopflowstudio/manabot/pull/417)
made authored setups decklists over the world's one catalog.

These changes directly support the requested multiplication of reusable
capabilities: a selector, amount, duration or cost can combine with several
verbs without one new handler per combination. Per-card definitions naturally
remain proportional to cards; execution machinery should grow with distinct
rules mechanisms.

**Keep:** typed definitions executed directly, constructors for recurring
compositions, first-printing ownership and one catalog. **Improve:** a batch's
design should identify which existing special cases its shared primitive will
absorb and delete. Do not add a second authoring language or speculative
compiler to avoid inspecting the real commonality. Conversely, do not reduce
distinct rules events to one verb merely because their printed text is similar.

### Derived characteristics had several readers

[PR #399](https://github.com/loopflowstudio/manabot/pull/399) centralized layered
characteristics and duration-bearing effect records. Before that, grants,
ability loss and temporary changes were read from separate fields. Copying
then exposed remaining divergent consumers: [#405](https://github.com/loopflowstudio/manabot/pull/405)
made battlefield observations and the legend rule read the same effective
characteristics, including copy exceptions and legendary status.

[PR #415](https://github.com/loopflowstudio/manabot/pull/415) extended contextual
power to non-battlefield zones so Rallier correctly filters Suki in the library.
That corrected the retained authored game from 142 to 135 Commands. It also
completed printed “any target” damage for planeswalkers and battles, without
claiming those entire card types were implemented.

**Keep:** one characteristics view and explicit printed/effective provenance.
**Improve:** each characteristic feature needs a consumer inventory covering
offering, payment, resolution, search, observations, copying and non-battlefield
queries. A correct field on the permanent does not make every reader correct.

### T6 initially preserved outcomes while changing semantics

T6 review caught approximations that the initial card tests had accepted:
Sozin selected targets before paying X and automatically selected search hits;
face equality stood in for intervening transforms; copied/inherited abilities
could depend on a departed source; and delayed timing used an insufficiently
specific next end step. The repairs retained explicit payment/reflexive-trigger
choices, per-card private search, transform state, captured ability data and
controller-specific timing. The substantive repairs are inspectable in
[`1bc8b33fb`](https://github.com/loopflowstudio/manabot/commit/1bc8b33fbac1dd1fb331833a143ad8fa608529cd)
and [`854d5e282`](https://github.com/loopflowstudio/manabot/commit/854d5e2820f51940206f240bd44e754a0fab4421),
and the accepted learning-evidence standard is recorded in the
[engine principles](../../../../../managym/PRINCIPLES.md).

**Improve:** specify adversarial witnesses before broad authoring: counter the
reflexive trigger, transform away and back, remove the grantor, search zero/fewer
matches, reconnect during search, and reach the other player's end step first.
Matching a simple final board is insufficient when a card exposes different
information or response windows along the way.

### Wider play found capacity and compatibility faults

The early demo raised action capacity from 20 to 32; Learn raised it to 64;
T6 now defaults fresh inputs to 512. TLA self-play exposed repeated-effect
fact pressure and a hard-coded 60-card native allocation. [PR #419](https://github.com/loopflowstudio/manabot/pull/419)
factored complete shared effects before capacity admission; [#420](https://github.com/loopflowstudio/manabot/pull/420)
made scalar/vector encoders honor supplied card capacity. Larger allocations
repair witnessed failures, not every reachable state.

**Keep:** explicit overflow, frozen checkpoint interpretation, exact replay
of retained failures and native/Python parity. **Improve:** batch acceptance
must include generated tokens, outside cards, wide action sets and repeated
effects through the ordinary encoder/rollout path, before paid learning.
Changing current defaults must not silently change saved omitted fields or
synthetic historical experiment fixtures.

## Performance: what worked and what to change

The first characteristics unification materially slowed search: #399 reported
worst Search-64 reply time of 7.0 seconds versus 3.1 on its main comparison.
That historical comparison motivated work; it is not a present benchmark.
[PR #410](https://github.com/loopflowstudio/manabot/pull/410) cached views per
revision; [#411](https://github.com/loopflowstudio/manabot/pull/411) preserved
caches through forks, separated revision domains and indexed effects/triggers.
[PR #414](https://github.com/loopflowstudio/manabot/pull/414) removed repeated
queries, allocations, priority generation and payment-search work.

The retained same-runner result for #414 is strong evidence: **30.43% less
regular step time, 43.93% less total Search-64 time and 49.91% less worst-reply
time**, with identical counts and digests. This is engine evidence, not
end-to-end learner throughput. [The speed investigation](../../../engine-performance/research/engine-speed-2026-11.md)
also records rejected optimizations whose apparent gains did not repeat.

The useful pattern was unify the authority, profile actual work, remove repeated
work, and keep a debug oracle. Full clone remained preferable to the rejected
undo/COW paths on the evidence available. Parallel rollouts spend extra cores;
they do not by themselves reduce engine CPU cost.

For Standard, extend the existing same-runner budget with bounded real
workloads covering new mechanics while retaining the TLA workload. Track a
milestone anchor as well as the rolling reference so successive accepted costs
cannot silently accumulate. Compare unchanged semantics with exact score and
decision digests; when rules change, disclose the changed workload and establish
a new comparable cohort instead of calling fewer decisions a speedup. Report
search total and tail latency separately from regular step throughput. No
artificial contention, stress workloads or cross-runner timing comparisons.

## Current residuals and boundaries

Historical audit tables are not current defect inventories: for example, their
claims that Auras, reflexive triggers and any-target damage are absent have been
superseded. Two explicitly ignored proposal tests remain at this checkout:
[earthbend destination incarnation and deathtouch SBA history](../../../../../managym/tests/rules/cr_608_resolution.rs).
Their source still documents the missing obligations. This pass did not rerun
them; their historical failures use synthetic state changes and do not establish
a currently reachable imported-deck sequence.

The native setup still draws seven without offering mulligans
([setup.rs](../../../../../managym/src/flow/setup.rs)); deck admission remains
development-only and sideboards have no between-game swapping
([decklist import contract](../../../../../docs/engine.md#decks)). Full card-file coverage
must therefore be distinguished from tournament-complete Standard play.
The payment model also retains its documented finite-source restriction;
repeatable mana sources require extending that model, not bypassing it.

## Recommended delivery contract

Acceptance should be proportional to the learning claim and expected workload,
not an exhaustive checklist of all Magic interactions. Prioritize checks that
detect unavailable strategic choices, hidden-information leaks, systematic
misrepresentation, unpayable offers and artifacts that could explain an apparent
learning result. Keep other known defects visible and continue useful training.
When a defect is found, identify the affected versions, scenarios and conclusions;
do not silently rewrite retained evidence or assume all learned weights are lost.

For each proposed mechanic batch, require a short rules contract and concrete
cards; named shared authorities and deletion targets; adversarial timing,
identity and privacy cases; native-to-policy/text/Academy coverage; and measured
search/runtime cost where the hot path changes. A new general primitive should
demonstrate its intended distinct compositions, rather than claim reuse from
one card and several aliases. Record how many cards need only definitions and
how many need genuinely new machinery, without making a quota that rewards
incorrect generalization.

Use directed independently specified outcomes to establish correctness, then
Command/replay/fork and cross-language parity to establish integration. Keep
bounded full-game cohorts as complementary evidence. Preserve negative results
and immutable receipts; retire obsolete executable paths instead of forcing
every new mechanic to maintain old worlds. Known bugs remain visible work and
must not revoke playability of already admitted cards.

Research validation: read git history, current source, retained evidence and
the linked PR descriptions; no tests, benchmarks, games, training or external
mutations were run. PR descriptions report their own dated checks, not checks
performed by this retrospective. Only this research report was authored.
