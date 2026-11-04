# MTG-241 — composable rules execution

## Outcome and status

Accepted directive: Jack Heart, 2026-11-04. One PR using the code Graph, stopping
at interactive review. Original base `4463e57f5`; checkpoint `37928095d` contains
nested iteration and both contextual-read fixes. Checkpoint `e6b268956` adds
common frame completion and shared library ordering; `58535f9f2` integrates
main `8e8d702f8`. Checkpoint `cf9f5a24b` adds retained inspection, keyword
completion and public-result selection after that sync. The current implementation extends that checkpoint with retained event batches
and affected-player replacement choices (2026-11-04). The full Task is **incomplete**, with
hidden-zone selection and successor-input projection still unfinished. No human decision is
missing. No PR publication, PR merge, training or retained-evidence rewriting occurred.
The Task initially had no Wave; `lf context` now reports the `coverage` binding.
Local `.lf/wave/coverage/GOAL.md` and `MEMORY.md` are absent; the supplied Task
directive remains the available scope. Wave memory has not been edited.

Effect execution retains nested operations, exact objects, scoped results and
root finalization across native choices. Replacement proposals must retain
transformed events, affected chooser, applied identities and logical batches,
re-evaluate applicability, and emit facts only at commit. Selection must preserve
cardinality, visibility, order, shuffle, reveal and result binding.

## Implemented ownership

`EffectFrame`/`run_frame` own execution; `Decision` owns the current question.
Native `Effect` remains the sole authored program. `FrameCompletion` now owns
one return destination: `Priority`, `Spell`, `Frame(Box<EffectFrame>)`, or
`Decision(Box<SuspendedResolution>)`, or a completion `Event` with its caller.
It replaces the independent `parent` and
`finalize` fields. Iteration keeps its remaining targets and creates a fresh
child scope; branches share their scope. Parent results and stack cleanup wait
until every child completes. `EffectFrame::new` initializes fresh result state.

`advance_mana_upfront` commits the automatic mana prefix, tap and life costs,
then retains the output decision and original payment frame. A sacrifice runs
through `run_before_decision` as an ordinary effect frame. Completing it restores
the saved output without re-entering payment, paying costs again or granting
priority. Nested continuation projection includes the saved caller's frame and
obligation chain before the active child.

Mana activation transfers the caller frame and decision by ownership; it does
not clone the decision chain or allocate a placeholder frame to overwrite.

At checkpoint `e6b268956` this established the completion owner and its payment
caller, not yet resumable zone/replacement execution. The cost-completion witness uses a constructed child
program (sacrifice then Surveil) to exercise a real pause through Commands. It is
not evidence of the still-unimplemented linked-entry replacement scenario.

`OrderLibrary` owns both top and bottom ordering. It retains cards, a committed
prefix cursor and `LibraryEnd`. Bottom ordering preserves its existing immediate
placement and private remaining-card contract (obligation code 23). Surveil's
keep action enters top ordering when multiple cards remain; all kept cards stay
privately visible/pinned, and choices commit a first-to-draw-first prefix. Top
ordering appends obligation code 27; its `mode` is the prefix length. Native
context, labels, text and Academy name that choice. The shared encoders already
encode the obligation code/cursor and visible library cards in their native order.
Existing widths and code meanings are unchanged. Complete iteration/results and
event payload projection still needs its explicit policy version boundary.

Top placement has one helper for forced and chosen ordering. Selecting a card
rotates it into the committed prefix, preserving the remaining offer order
without removing and reinserting it. Nested iteration, payment and ordering
witnesses share the same bound Command/replay/stale-rejection check.

`SuspendedResolution::continuation` owns the root-first walk through frames and
saved decisions. Observation projection, inspection permissions and search read
that borrowed chain; they no longer reconstruct separate frame/mana ancestry.
Projection maps each retained part and links it to its predecessor. This changes
neither serialized runtime state nor continuation codes.

## Delete — do not maintain

Removed in checkpoint `37928095d`:
- Local ForEachTarget subframe loop and no-decision assertion; retained scopes
  preserve eligibility/order, isolated results and root finalization.
- Owner-based entry conditions and live-only SourcePower. Surviving reads use
  prospective entry control and exact-source current/LKI power.
- Repeated fresh-frame literals and the one-use `with_parent` helper.

Removed in checkpoint `e6b268956`:
- Independent `EffectFrame.parent` and `FrameFinalize` ownership; completion
  names the interrupted scope or caller in one place.
- Mana sacrifice's post-call `ManaUpfront` mutation; its output is retained
  before child work starts.
- Mana activation's cloned resume chain and throwaway frame; transfer the
  retained caller directly.
- `OrderLibraryBottom` and `library_bottom_decision`; all current consumers use
  shared top/bottom ordering. Surveil no longer resumes immediately on keeping
  multiple cards.

Implemented in `cf9f5a24b` (2026-11-04): retained inspection walks all caller/iteration
frames and saved mana decisions, shared by observation and determinization.
`FrameCompletion::Event` retains a keyword completion fact until child choices
finish. Surveil retains its graveyard partition until selection finishes,
then moves it synchronously under one pre-movement trigger-source snapshot,
then orders kept cards; Scry and Surveil announce completion after their choices.
Planetarium now matches both keyword events. Completion immediately matches
trigger sources before the caller can remove them, without placing triggers or
granting priority; a directed departure witness verifies that boundary. These
changes do not yet make replacement or zone entry suspendable.
The Surveil witness proves no movement during partition selection, one move
per chosen card and one keyword event after kept ordering. The movement loop
still calls the synchronous zone authority for each card; it does not prove
a retained simultaneous batch across replacement choices.

Removed in the current cut: `MillAndSelect`, `ReturnMilledThisWay`, `MilledBonus`
and `Decision::SelectMilled`. Native `SelectCards` chooses exact selector results
with min/max cardinality, then binds `Selected` in a child scope. Mill/discard
results now retain exact objects in `this_way`, independently of later `Move`.
`CardsThisWayMatch` replaces the discard-specific condition. Avatar Destiny
preserves its source-return between milling and selection; Ostrich-Horse keeps
its decline counter and Sparring Dummy its Lesson condition. Mill movement uses
one pre-batch trigger snapshot. Shared selection appends obligation code 28;
legacy code 5 remains retired. Text and Academy name the new stage. The native
Command witness covers partial selection, two forks, exact incarnation rejection,
a child pause, preserved caller mill results and one finalization.
Sparring Dummy now reads that retained mill result after selection, removing the
duplicated selection in its Lesson/non-Lesson branches. Selection completion
uses its already-clamped maximum rather than checking candidate count again.

The current selector migration covers public-zone results. Extending it to hand
search/discard needs explicit hidden-world handling: a retained exact hand
candidate can be reassigned to the library by determinization. Rebuilding only
the active action list cannot repair a saved caller or its partially selected
subset. The migration must preserve viewer-known exact memberships and refresh
unknown commitments coherently through every retained frame, using the same
source/visibility contract as possible-world materialization. Do not extend
snapshot candidates into hidden-zone consumers before that cut is implemented.

Still slated for removal in the same PR (updated after the event-batch cut):
- OptionalDiscardDraw, MayDiscard,
  SearchBasicLand and overlapping search/move handlers and dedicated decisions,
  after moving real-card consumers to shared selection. Preserve distinct
  keyword events, private inspection, decline and ordered remainders.
- Remaining synchronous hidden-search movement/finalization postambles; move
  their consumers to the batch owner as part of deleting those composites.

## Current event-batch cut — 2026-11-04

`EffectFrame.batch` owns `EventBatch`: proposals, stable original ordinals,
applied identities (exact source, definition index and static/replacement
namespace), cursor, pre-departure LKI and trigger sources, committed results and
consequent child batches. Applicability is recomputed after each replacement.
`Decision::OrderReplacement` (code 29) gives the affected player the choice;
labels and action focus name its source. No proposal commits until every member
is ready. Linked returns retain the root's results instead of replacing them.
Already-paid tap/life events are matched before the next batch can remove their
sources; their triggers stay queued during replacement choices. Lifelink and
counter loss from damage commit within the same damage batch, without a nested
trigger pass. Counter placement emits the transformed count; entry counters, including Saga
lore, pass through replacement before entry. Departures match against pre-batch
sources, entries against the finished battlefield. LKI is captured before any
member leaves. Snapshots are limited to batches with departures; no cache was
introduced and no performance improvement is claimed.

Frame completion now retains spell entry/finalization, paid announcements and
activation objects. Cost source/target metadata is captured before sacrifices;
hand-ability source identity binds after its discard cost. Combat damage uses a
batch, then emits its successful-hit group once. First-strike's SBA interval
retains the normal-damage return destination. SBA progress is explicitly
Stable/Changed/Pending, and token/attachment cleanup waits for child work.
`turn_tick` exposes a pending turn action before stabilization. Token creation,
Earthbend, library placement, chosen exile/sacrifice groups, counter distribution
and LookAndSelect battlefield movement now retain their postambles. Surveil's
partition is one batch before the saved kept-order decision becomes available.

Removed in this cut: one-pass replacement sorting; synchronous recursive linked
returns and Saga counter insertion; `finalize_frame`; synchronous sacrifice,
destruction, moved-object and explicit-source event-processing wrappers; damage
wrappers measuring mutable post-call state; synchronous combat hit loops.
`apply_immediate_event` is the same batch executor with an explicit no-question
precondition, for fixture mutations and the remaining immediate callers. It is
not a pending-as-success adapter. The surviving SearchBasicLand path remains a
planned deletion; it is not converted to snapshot hidden candidates here.

Five Command/fork/replay/stale-rejection witnesses cover two-object counter
batches producing 3 or 4, newly applicable entry replacements and once-only
identity, actual paid mana activation → sacrifice → linked-entry choice → output
→ original payment, simultaneous combat damage, and SBA linked returns/LKI.
The counter drivers are explicitly constructed `ModifyCounters` programs, **not**
registered Caradora/Doubling Season cards or the real Leaves from the Vine
acceptance witness with those cards. A further directed test uses the registered
Leaves from the Vine chapter and constructed replacement sources through Commands,
fork/replay and stale rejection. Its former `ForEachTarget` counter loop was a
newly found predecessor: it committed each target separately. Removed that loop;
`ObjectSelector::Targets` supplies every eligible slot to one verb/batch, while
`ForEachTarget` retains its distinct sequential-program meaning. The learning
token guard covers the selector. Both placements remain uncommitted through the
second object's question, then each receives three or four counters once.
The complete Task remains **incomplete**.

## Remaining implementation and acceptance

1. Complete the real Leaves from the Vine / Caradora / Doubling Season witness,
   including complete definitions, learning/observation coverage and its logical
   simultaneous counter event. Current fixtures prove the replacement operation,
   not those cards. `ModifyCounters` expresses positive placements (including
   entry) as `count * multiplier + additional`; no current card is registered
   using it yet.
2. Migrate hidden-zone selection/materialization and delete OptionalDiscardDraw,
   MayDiscard, SearchBasicLand and their exclusive handlers/decisions. Preserve
   decline, inspected membership, reveal, shuffle, ordered remainders and all
   result bindings through every retained caller. Public-result SelectCards
   still snapshots exact objects; its hidden-world warning above still applies.
   Complete the audit of immediate callers during this deletion cut; search's
   source-bottoming postamble must also retain linked returns.
   Materialization audit: both `Game::determinize` and
   `HiddenHandConstraints::materialize_validated` reassign hidden membership,
   repin inspected library prefixes, then refresh only the active offer. The
   cut needs retained source/visibility provenance and traversal of selections
   in saved decisions and child scopes. Unknown selected subsets must be
   resampled consistently with the sampled zone and predicate; known exact
   membership stays fixed. A partially selected restricted search also constrains
   which deals are compatible (enough eligible objects must remain); merely
   discarding missing selected IDs would erase an earlier commitment. This is
   required behavior for the shared authority, not permission to preserve stale
   snapshots or silently narrow legal candidates.
3. Project pending batches (including precommit entrants), iteration progress,
   selected/this-way results, completion events and paid/rule callers. Current
   obligation 29, labels and source focus are only the legal-choice surface;
   they do not complete policy, LLM, Academy or reconnect observability.
   Paired states with identical current offers must distinguish consequential
   viewer-safe payloads and hide unavailable bindings. Preserve frozen versions
   and reject unrepresentable older inputs before inference.
4. Reconcile successor schema with MTG-244 through ordinary integration. Its
   latest shared risk directive explicitly names **v8**; no concrete layout has
   been supplied in Task comments. This Task has not assigned a competing v8.
   MTG-243 received code 29 and completion-interface coordination. Preserve
   v1-v7 meanings, fresh v6, capacities and overflow contracts. World/hash,
   capture version boundaries and a new conformance corpus remain unfinished.
5. Jack Heart's added performance/learning-risk acceptance is in
   [risk validation](mtg-241-risk-validation.md). Gate owns bounded same-runner
   step/search total/tail measurements, profiling and remeasurement of targeted
   mitigations, paired policy inputs/model coverage, full debug Rust/Clippy,
   affected Python/headless Academy, world admission and Engine Budget. No
   benchmark, new training experiment or speed claim has been made in this cut.
   No publication or merge is authorized before interactive review.

## Integrated upstream contracts — 2026-11-04

Main `8e8d702f8` derives native scalar/vector capacities from supplied arrays,
validates companion shapes, and adds explicit `action_overflow=first-legal`.
Continuation projection must retain those capacities and original prefix
indices/Commands; omitted overflow remains an error. Academy and native legality
still expose all actions. The MTG-213 pilot exception is not permission to cap
new replacement or selection choices implicitly. Main also shares the selected
casting cost between X announcement and payment and preserves X under any-type
mana permissions; payment integration must retain those fixes.

The updated validation guide adds observation, cutoff and buffer-capacity tests
for encoding/action changes, and active-training tests for input/model changes.
These are gate obligations alongside the existing world-admission checks.
The focused checks recorded in `cf9f5a24b` ran after this merge. Thus the supplied
iteration feedback's “post-sync runtime validation remains unclaimed” applies
to its earlier checkpoint, not this one. Broader gate validation is still
unclaimed; focused passes do not close the remaining acceptance cases.

## Completion-boundary evidence

The original `advance_mana_upfront` counterexample is now executable through
real Commands: an activated mana source pays mana/tap/life and sacrifices,
ending a linked exile duration; the returned object's entry replacement asks a
question while the already-paid output and original payment remain retained.
After choosing, output is offered once and the original payment completes once.
This replaces the former constructed sacrifice-then-Surveil proof for that
specific integration claim. Full observation/learning acceptance remains open.

## Coordination and evidence

MTG-242 owns payment resources/search; MTG-243 naming; MTG-244 layers. Shared
Task reads on 2026-11-04 during this reconciliation show MTG-242 idle after its
second loop-or-next pass, MTG-243 published for review as
[PR #424](https://github.com/loopflowstudio/manabot/pull/424) at demo, and
MTG-244 implementing. Publication does not establish integration here.
Task comments now carry FrameCompletion and the paid-output child protocol;
MTG-243 also has the code-27 reservation. MTG-244's dated comment records Jack
Heart's successor-input approval and rejection before inference for older inputs
that cannot represent its payload. No shared comment reserves a concrete input
or world version. Their checkouts were not modified or copied. Ordinary Task sync
remains integration; the layers algorithm is untouched.

Wizards' [Guilds of Ravnica release notes](https://magic.wizards.com/en/news/feature/guilds-ravnica-release-notes-2018-09-20)
confirm that surveil triggers after its operation completes, including an empty
library. The Planetarium/Joo Dee witness covers the real keyword interaction.

Wizards' Aetherdrift release notes (Caradora) give counter replacement order to
the affected permanent's controller, including entry. Foundations release notes
state Doubling Season also changes entry counters. Thus (1 + 1) × 2 = 4 and
(1 × 2) + 1 = 3 are distinct legal outcomes:
- https://magic.wizards.com/en/news/feature/aetherdrift-release-notes
- https://magic.wizards.com/en/news/feature/foundations-release-notes

Prior checkpoint evidence is at
`37928095d:scratch/mtg-241-composable-rules-execution.md`: 43 focused debug tests,
two real-card contextual tests and one native/Python parity test; extension
rebuilt/atomically installed. The nested Command witness covers both seats,
two nesting levels, divergent forks, replay, stale rejection and one finalization;
the timing witness defers SBAs. Combustion Man uses departed seven-power LKI
instead of the returned four-power incarnation; raid uses entering control.

Approved `managym/PRINCIPLES.md` and its README link remain. Its final evidence
paragraph follows Jack Heart's steer: public PRs #42, #348, #353, #289 and #412
plus the speed investigation, without relying on the uncommitted retrospective.

Previous check details remain at `e6b268956:scratch/mtg-241-composable-rules-execution.md`.
Post-sync evidence and exact commands remain at
`cf9f5a24b:scratch/mtg-241-composable-rules-execution.md`: 38 focused debug
Rust tests, inherited real-card/trigger checks and four Python text/parity
cases passed; the release extension was rebuilt and atomically installed.
These results are reused, not rerun for this documentation-only reconciliation.
Gate owns affected suites, successor-input admission, reconnect/browser coverage
and Engine Budget after the remaining implementation cuts.

Check (2026-11-04, current cut): focused debug Rust filters passed: `--lib retained_batch` (5), `--lib surveil` (2), `--lib every_decision_code_is_within_the_validated_bound` (1), and `--test rules_tests` filters `cr_614_replacement` (5), `cr_510_combat_damage` (3), `cr_704_sba` (10), `cr_602_activated_abilities` (8), `strike` (4), `lifelink` (2), with `-p managym --no-default-features`; the five new witnesses were rerun after the paid-tap timing fix. After adding the Targets selector, `-p managym-cards --test cards` filters `leaves_from_the_vine` (3) and `every_object_and_player_selector_reaches_the_learning_tokens` (1) passed in debug. The final cp312 release extension was rebuilt and installed atomically after the selector change; `uv run --extra dev pytest tests/semantic/test_text_observation.py -q -k joo_dee_surveil` passed (1). `git diff --check` and `lf context` pass. Gate owns broader checks and all performance/model-risk measurements; earlier checkpoint evidence above remains historical.
