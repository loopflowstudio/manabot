# MTG-241 — composable rules execution

## Outcome and status

Accepted directive: Jack Heart, 2026-11-04. One PR using the code Graph, stopping
at interactive review. Base `4463e57f5`; checkpoint `37928095d` contains nested
iteration and both contextual-read fixes. The working tree adds common frame
completion and shared library ordering. The full Task is **incomplete**, with
replacement/event completion the largest remaining cut. No human decision is
missing. No publication, merge, training or retained-evidence rewriting occurred.
This Task has no Wave.

Effect execution retains nested operations, exact objects, scoped results and
root finalization across native choices. Replacement proposals must retain
transformed events, affected chooser, applied identities and logical batches,
re-evaluate applicability, and emit facts only at commit. Selection must preserve
cardinality, visibility, order, shuffle, reveal and result binding.

## Implemented ownership

`EffectFrame`/`run_frame` own execution; `Decision` owns the current question.
Native `Effect` remains the sole authored program. `FrameCompletion` now owns
one return destination: `Priority`, `Spell`, `Frame(Box<EffectFrame>)`, or
`Decision(Box<SuspendedResolution>)`. It replaces the independent `parent` and
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

This establishes the completion owner and its payment caller, **not** resumable
zone/replacement execution. The cost-completion witness uses a constructed child
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

## Delete — do not maintain

Removed in checkpoint `37928095d`:
- Local ForEachTarget subframe loop and no-decision assertion; retained scopes
  preserve eligibility/order, isolated results and root finalization.
- Owner-based entry conditions and live-only SourcePower. Surviving reads use
  prospective entry control and exact-source current/LKI power.
- Repeated fresh-frame literals and the one-use `with_parent` helper.

Removed in the working tree:
- Independent `EffectFrame.parent` and `FrameFinalize` ownership; completion
  names the interrupted scope or caller in one place.
- Mana sacrifice's post-call `ManaUpfront` mutation; its output is retained
  before child work starts.
- Mana activation's cloned resume chain and throwaway frame; transfer the
  retained caller directly.
- `OrderLibraryBottom` and `library_bottom_decision`; all current consumers use
  shared top/bottom ordering. Surveil no longer resumes immediately on keeping
  multiple cards.

Still slated for removal in the same PR:
- MillAndSelect, ReturnMilledThisWay, OptionalDiscardDraw, MayDiscard,
  SearchBasicLand and overlapping search/move handlers and dedicated decisions,
  after moving real-card consumers to shared selection. Preserve distinct
  keyword events, private inspection, decline and ordered remainders.
- Fixed-order/one-pass replacement processing. Preserve exact applicability,
  cheap no-replacement execution, pre-batch sources and single commit.
- Synchronous zone/event postambles whose callers can suspend. Replace them
  with retained completion/results, not a pending-as-success bool/void adapter.

## Remaining implementation and acceptance

1. Extend common completion across event batches and zone entry, combat and
   SBA callers. Event results must carry committed/moved objects and damage
   results to the interrupted instruction, before its suffix or caller resumes.
   The current payment frame is ready to retain child work, but sacrifice,
   zone moves and `finalize_frame` still execute synchronously underneath it.
2. Retain replacement batches across affected-player ordering choices. Recheck
   applicability after every replacement, retain applied identities and all
   proposals, and commit the logical batch once using pre-batch sources.
   Leaves from the Vine with Caradora and Doubling Season must yield three or
   four counters depending on order. These replacement types/cards are not
   implemented in this checkout. A separate witness must make a previously
   inapplicable replacement applicable and prevent identity reapplication.
3. Actual payment integration witness: sacrifice a mana source with a linked
   return; suspend the return's entry replacement; show committed costs remain
   single, output unavailable until completion, and original payment resumes.
   Neither the constructed completion test nor existing Prism/Treasure tests
   proves this integration yet.
4. Complete shared candidate/move/result selection migration and delete the
   listed composite handlers. Surveil kept ordering is implemented; its
   graveyard selections still move one at a time, so simultaneous selection
   movement and the complete keyword-event contract remain open.
5. Complete versioned native/Python inputs, learning, LLM, Academy and reconnect
   coverage for nested iteration/results and future batches. Retained caller
   inspection across nested decisions also needs auditing: current observation
   and determinization inspected-card readers primarily use the active Decision.
   Current root/caller frame projection alone does not establish full observability.
   World/hash and capture version boundaries and a new conformance corpus remain
   part of the completed change; frozen receipts must not be rewritten.
6. Gate owns full debug Rust/Clippy, affected Python and headless Academy checks,
   world admission and same-runner Engine Budget under `docs/development.md`.
   No speed claim follows from focused checks. A rules-driven digest change
   requires its own explanation and accepted baseline handling.

## Completion-boundary evidence and next interface

The original counterexample (2026-11-04) was `advance_mana_upfront` sacrificing
then assuming `ManaUpfront` survived. `sacrifice_permanent` → `move_card` →
`commit_zone_move` → `handle_leaves_battlefield` returns linked exiled cards.
With competing entry replacements, that return must ask a question. Installing
a Decision in today's synchronous helpers either clobbers its caller or permits
premature completion. Saving the payment caller now removes its post-sacrifice
assumption, but does not make those helpers return a child operation.

`finalize_frame` likewise moves a permanent spell then emits SpellResolved and
restarts priority. Combat accumulates hits in a local vector. Saga entry puts
counters recursively during zone commit. These postambles need retained owners;
per-target iteration alone cannot preserve a simultaneous event.

Technical continuation of the accepted design: an event batch owns transformed
proposals, applied identities, logical progress, pre-batch trigger sources and
results. Its completion returns results to the caller frame/instruction before
continuing. Zone commit owns its linked returns and Saga follow-up; terminal
root completion distinguishes resolution, turn action and SBA stabilization.
No-replacement work completes inline. A second replacement-only scheduler,
replaying already committed costs, or silently accepting Pending is excluded.
The concrete batch/result payload is still to be implemented; the frame/caller
completion interface above is working code, not a claim that this wider cut is done.

## Coordination and evidence

MTG-242 owns payment resources/search; MTG-243 naming; MTG-244 layers. On
2026-11-04 their shared Task status showed ongoing implementation (242/243).
Task comments now carry FrameCompletion and the paid-output child protocol;
MTG-243 also has the code-27 reservation. No response has reserved an input or
world version. Their checkouts were not modified or copied. Ordinary Task sync
remains integration; the layers algorithm is untouched.

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

Validation (2026-11-04): `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 -p managym --lib -- resolution_tests payment_tests` — 27 passed; the cards target with filters `audit_prism_treasure_empty_pool_activation`, `prism_payment_offers_execute_and_restore_without_mutating_sibling_forks`, `accumulate_wisdom_look_three_pick_one_rest_to_bottom`, `joo_dee_` — 7 passed. Required release wheel rebuilt and extension atomically installed; `uv run --extra dev pytest tests/semantic/test_text_observation.py::test_accumulate_wisdom_ordering_reaches_text_commands_and_both_encoders -q` — 1 passed. Earlier decision-code bound and text-label checks passed and remain applicable. Formatting, diff whitespace and context-budget checks passed. Gate owns broader checks after implementation; top-order native/Python parity and full reconnect proofs remain unclaimed.
