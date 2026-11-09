# Engine development principles

managym exists to support useful, reproducible learning. Bugs are expected;
perfect Magic conformance is not a prerequisite for training. The consequential
failure is drawing conclusions that later have to be invalidated because the
engine omitted a choice, exposed private information, or rewarded a rules bug.
Prioritize defects by their effect on gameplay, training and experimental claims.
A discovered bug does not automatically invalidate every run or checkpoint.
Identify affected versions and conclusions, preserve the original evidence, and
repeat the comparisons whose interpretation depends on the fix.

The Allies–Lessons, Elemental and TLA implementations taught a recurring lesson:
a plausible final board can hide a different game. The most useful refactors
gave each rule one owner, then removed repeated work inside that owner. These
principles guide new mechanics and reviews; they are not an exhaustive
conformance gate or an additional card-admission policy.

## 1. A working demo establishes a useful slice

Build with real decks and group cards by shared capabilities. The two-deck demo
established triggers, costs and resumable choices that later cards reused. Its
success did not establish behavior for every new combination. Distinguish an
intentional scope expansion, such as Learn gaining outside-game choices, from
a defect within the supported behavior.

State what a scenario demonstrates and what a learning result supports. Choose
small, consequential counterexamples for new mechanisms before relying on long
games to discover their assumptions. Preserve exact failure prefixes so fixes
can be checked against the original problem.

## 2. Legal offers, payment and execution share one authority

An offered action needs an executable completion under its announced choices.
Prophetic Prism exposed affordability checks that consumed a resource needed
to finish casting; Treasure paying Prism's upfront cost then exposed missing
nested payment state. Shared payment reasoning and retained progress repaired
both families more effectively than card-specific exceptions.

Factor rules into reusable operations, selectors, values, costs and durations.
A new combination should reuse their owners, including across choices and
source departure. When generalizing a special case, migrate its consumers and
delete the redundant handler. Keep distinct rules events distinct: tapping to
pay waterbend is not tapping for mana.

## 3. Timing, identity and information are behavior

For a mechanic, identify who chooses, when they choose, what they can inspect,
what can respond, and which exact object or historical fact later instructions
read. Current characteristics, resolution-entry eligibility, last-known
information and prospective entry state answer different questions.

Target legality needed both a shared reader and the right evaluation time.
TLA exposed targets chosen before payment, automatic search selections, current
faces substituted for transform history, and delayed effects using the wrong
player's end step. Final-board assertions alone missed these distinctions.

Use witnesses that distinguish the intended behavior: remove the source,
transform away and back, decline or select fewer objects, or pause and fork
during the relevant choice. Retain unfinished work explicitly; a player decision
inside resolution does not grant priority.

## 4. Choices reach the learner intact

The native decision authority owns meaningful alternatives for policies,
search and Academy. The implementation previously exposed distinct actions
through identical policy inputs and automatically selected trigger and library
bottom orders. A model cannot learn a distinction its input erases.

Carry mechanics through executable definitions, learning tokens, viewer-safe
observations and legal actions. Use the observability guards and native/Python
parity, then check consequential distinctions rather than only matching shapes.
One trigger order surviving while another loses is a better witness than two
differently labeled buttons. Where useful, verify that either preference can be
learned. Continue skipping sole legal actions natively; never replace a wide
choice with an unreported shortlist or fallback.

## 5. Consolidate ownership, then measure and remove repeated work

Centralizing characteristics initially slowed search. Revision caches, shared
indexes, cache-preserving forks and fewer repeated queries subsequently improved
both regular execution and search. The retained same-runner comparison for
PR #414 measured 30.43% less step time and 43.93% less total search time with
unchanged counts and digests. These are historical engine measurements, not a
promise for future refactors or an end-to-end training result.

Consolidation does not waive the performance budget. Measure ordinary steps and
search separately on equivalent work; retain a debug recomputation oracle for
caches and indexes. A changed rules trace needs an explained comparison, not a
speedup claim based on fewer decisions. Track cumulative milestone cost as well
as the rolling reference. Prefer eliminating demonstrated repeated work over
speculative storage rewrites; more workers do not establish lower CPU cost.

## Architectural boundaries to preserve

The broader-engine reviews exposed requirements to incorporate as coverage
grows. These are design boundaries, not claims that all their cases already
work in managym:

- **Own unfinished work.** A parent operation retains its position, scoped
  results and finalization while a child asks a question. Resume the exact
  child and complete the parent once. Use a common ownership/completion
  protocol with operation-specific data; avoid a new continuation convention
  for every mechanic. A logical simultaneous batch survives a pause without
  exposing priority or running premature state-based actions.
- **Separate proposals from facts.** Replacement processing transforms a
  proposed event, rechecks applicability and records which replacements have
  applied. Only committed events enter trigger matching. Candidate indexes
  accelerate discovery; they do not own eligibility, occurrence identity or
  placement order. Retain the relevant pre-event sources and last-known facts.
- **Keep one derived-characteristics authority.** Extend layer ordering,
  dependencies and discovery of granted abilities behind that view. New reads
  of tap state, zones, history or other conditions require corresponding cache
  invalidation. Preserve debug recomputation. Full dependency support belongs
  with the mechanic batch that needs it; ordinary definitions need not wait
  for an unrelated layer extension.
- **Keep rules state and viewer input distinct.** Preserve exact object
  incarnations and distinguish current, last-known and prospective reads.
  Project permitted information explicitly. Search forks retain pending work,
  history and RNG, and hidden-world sampling respects the viewer's knowledge.
  A reconstructed approximation of the game is not an exact fork.
- **Share semantics before changing storage.** Native definitions remain the
  executable card representation. Broader engines are sources of counterexamples
  and algorithms, not a mandate for a parser, second IR or storage rewrite.
  Choose persistent collections, instruction sharing or other optimizations
  only from measured costs and preserved execution semantics.

## Evidence and implementation contracts

Historical examples include the [two-deck foundation](https://github.com/loopflowstudio/manabot/pull/42),
[payment witnesses](https://github.com/loopflowstudio/manabot/pull/348),
[nested payments](https://github.com/loopflowstudio/manabot/pull/353),
[policy choice distinctions](https://github.com/loopflowstudio/manabot/pull/289)
and [composable effects](https://github.com/loopflowstudio/manabot/pull/412).
These are dated evidence, not a current defect inventory.
The [speed investigation](../.lf/wave/engine-performance/research/engine-speed-2026-11.md) retains
measurements and rejected optimizations. Use the [engine README](README.md),
[repository conventions](../AGENTS.md) and
[validation guide](../docs/development.md#validation-by-change-area) for current
ownership, compatibility and required checks.
