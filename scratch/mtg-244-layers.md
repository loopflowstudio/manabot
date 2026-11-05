# Composable layers — MTG-244

Jack Heart accepted one composable-layers Task on 2026-11-04: Beorn/Kellan as
real cards, historical interactions as rules fixtures. The code Graph publishes
one PR for review; no merge, world freeze or training launch. Jack Heart approved
a successor policy input the same day: "new input version is totally fine".
Explicit v8 is implemented; fresh v6 and frozen saved versions remain unchanged.

Status, 2026-11-04: published head `68ef4a63d` includes reusable source
plans, conservative dependency pruning and lazy pre-mutation SBA snapshots.
Its long integration passes in CI `41956687926`; the budget, Rust/main
integration and Layer Risk `41956688122` are still running; the
preceding `a7b0ddaff` checkpoint failed the unchanged step budget by 10.54%.
[Risk validation](mtg-244-risk-validation.md) owns measurements/limits and
[CI repair](mtg-244-ci-repair.md) retains earlier failures. No product decision
is missing and publication does not establish acceptance.

Accepted outcome: one `Game::characteristics` authority composes types, colors,
abilities and power/toughness for rules and policy consumers, preserving a cheap
common path, immutable revision caches and debug recomputation. This Task owns
layers and their cache dependencies. Composable execution owns pending frames,
replacement proposals and contextual source/entry callers; mana consumes the
layered abilities, and naming owns large choices. Those adjacent refactors are
not prerequisites for unrelated simple card definitions.

## Approved successor input; implementation design

Structured grants, effective colors and keyword-counter timestamps reach
native/Python/text projections and model inference. Explicit v8 is the chosen
successor; fresh v6 and all frozen v1–v7 widths, omissions, parameter shapes
and timestamp normalization remain unchanged. No default migration was requested.

V8 retains v7 object/relation/header/member widths. Its effect details are
4096 × 3: kinds 11–19 carry the shared vocabulary's token kinds 1–9, with exact
signed i32 values split into 16-bit halves. Local facts default to 512 rows
(configurable); kinds 11/12/13/14 carry copy programs, effective colors,
keyword-counter timestamp ranks and continuation programs. Kind 14 identifies
the frame (index + 1), token kind/value and ordinal. Ordered model reducers
consume programs and numeric values; v1–v7 parameter shapes are unchanged.
Native scalar/vector buffers select v8 from the distinct detail capacity.

Shared `validate_policy_input` runs at both encoder boundaries:
exactly representable states continue to work; reject unrepresentable new layer
payloads before inference with a diagnostic naming the selected version and
required successor. Silent omission, implicit upgrading or
an embedding-index failure is not an admission policy. Card/deck support remains
definition-based; input compatibility cannot become a bug-based card exclusion
list. Saved architecture loading and current-world admission are separate proofs.
No historical checkpoint, receipt or conformance corpus is rewritten.

Required evidence includes native/Python/text/token agreement and a complete
policy forward pass on the successor. Both existing fact encoders and the v7
shared-effect encoder need valid retained-input regression checks plus explicit
unsupported-payload rejection, not tests that force new payloads through frozen
architectures. Retained model configurations load through `from_checkpoint`,
preserving omissions and parameter shapes; this alone does not admit historical
trained checkpoints to the changed world.

## Evaluator

`Game::characteristics` remains the sole derived battlefield authority, including
effective colors and abilities. Copy/control retain their event owners. A private
context starts from copiable values and evaluates affected objects together.
Static indexes discover candidates; final ability loss must not filter candidates
before layers. Explicit context reads prevent public-characteristics recursion.
Cache entries remain immutable and clone-safe with debug recomputation.
Named layers and a separate dependency selector keep discovery/application apart
from ordering. Resolved P/T amounts are mapped once; the common independent-P/T
path avoids dependency matrices in release while debug checks the full graph.
Source-relative predicate alternatives recurse through one shared matcher, so
layer and final rules questions retain the same source name.

Within a layer, compare effect existence, applicability and instructions before
and after applying another pending effect (CR 613.8a), not resulting values:
ordinary competing setters are independent. Reevaluate after each application.
Ignore only edges within dependency loops, preserve external dependencies and
use timestamps inside loops. A continuing multi-layer effect retains its affected
set (613.6). Newly granted abilities carry their grant's timestamp (613.7a).
CDAs precede ordinary setters; switches follow modifiers and counters in 7d.
Keyword counters need their own timestamp, renewed on addition (613.7c).
Tap predicates/source conditions require cache invalidation; `tap` and `untap`
advance a separate tapped revision. Trigger candidates consume
layered grants and suppression, with exact departed-object ability information.
Mana writes retain the raw board unless any discovered layer node reads unspent
mana, including nested conditions and granted statics. Tap-state reads have the
same explicit revision treatment. Fixed activations retain finalized views;
conditional outputs and cost reductions refresh. Trigger event kinds follow raw
layer dependencies, while matching reads current LKI. Debug recomputation checks
all cached views and indexes.

## Captured programs

The copy/grant/loss witness found an activation offered and paid from the
layered list but silently discarded by printed-definition lookup at resolution.
The narrow handoff now retains the chosen effects on the stack; resolution and
retargeting consume that snapshot. Catalog provenance remains an input encoding
optimization only. Dynamically granted programs remain captured through stack,
trigger placement and suspended resolution; v8 adds ordered continuation-program
facts (kind 14, frame index plus one, token kind/value and ordinal). Existing
catalog programs keep their retained input representation. This is the necessary
layer-consumer interface, not a redesign of the adjacent execution Task.

## Implemented witnesses and evidence limits

Directed witnesses include timestamp orders, dependency loops, removal/re-entry,
copy/grant/loss, real-decision forks and event-driven invalidation. Rules consumers
and native/Python/text/policy inputs use the same derived view. Beorn/Kellan are
real definitions with payment/combat/trigger and Scryfall checks; historical
fixtures do not register a pool. The Saga chapter-II fixture uses Ballistic
Boulder as an existing Construct marker; it makes no Saga-token sizing claim.

The supplied brief's expected outcomes are independently asserted. The preserved primary rules and card responses anchor their tests:

| Witness | Required distinction |
| --- | --- |
| Beorn/Kellan | Bear addition versus Rogue replacement in both orders; Bear anthem, persistent trample counters and retained granted abilities; complete real-card behavior. |
| Moon/Urborg | Layer-4 dependency removes Urborg's generating ability first in either timestamp order; no Swamp grant. |
| Sun/Urborg | Layer-6 loss occurs after Urborg grants Swamp in layer 4. |
| Moon/Saga | After chapters I/II, retain separately granted mana/Construct abilities; without chapters, no precombat lore counter or chapter-completion sacrifice. |
| Dress Down/Magus | Magus loses abilities after its layer-4 Mountain effect applies. |
| Dress Down/Tarmogoyf | Loss before the CDA's layer leaves 0/1 absent other modifiers. |
| Dress Down/Opalescence | With Opalescence present first, Dress Down enters as an abilityless 2/2, with neither draw nor end-step sacrifice trigger. |
| Humility/Opalescence | One Opalescence does not animate itself; Humility is abilityless, 1/1 if its timestamp is later, 4/4 if Opalescence's is later. |
| Ygra/Imprisoned | Enchanted own Ygra becomes a land with granted mana ability, losing ward/death trigger; other creatures remain Foods with the granted Food ability. |
| Lattice/March | Dependency makes ordinary noncreature lands 0/0 artifact land creatures, dying without bonuses; Lattice/March become 6/6 and 4/4. |
| Painter/Dress Down | Painter's previously chosen color addition survives later layer-6 ability loss. |

The optional Trickster/Dryad witness distinguishes removed additional-land
permission from retained basic land types. It does not replace any required pair.

The focused Python module covers v8 forward/backward, checkpoint reload,
scalar/vector and native/Python parity, retained v5–v7 architectures and Kellan's
trigger after source death. `semantic-kernel-w17-v2` records 14 cases, 3,473
commands and 15 Phase rows; v1 remains unchanged. These checks establish neither
full-head acceptance nor the still-required isolated performance outcome.

## Hot-path design — 2026-11-04

Selected implementation: retain immutable copiable source objects and static
plans per permanent incarnation across revision rebuilds and search forks.
Copy-on-write intermediate objects let dependency probes copy only modified
objects. Conservative read/write summaries rule out impossible dependency pairs;
plausible pairs retain exact instruction/existence comparison and reevaluation.
Debug evaluates from fresh source plans and checks skipped pairs independently.

Alternatives: merely widening independent batches leaves repeated definition
construction and whole-board cloning. Per-object invalidation could avoid more
work but needs complete dynamic scope/value dependency tracking; defer that
higher-risk change until this cut's measured residual cost justifies it. Existing
attribution (368,743 boards, 178,899 context clones, no ordinary dependency edges)
and payment allocation traffic support sharing plans and avoiding impossible
probes first. Local timing under contention is not acceptance.

Delete — do not maintain (implemented cuts; no remaining deletion target):
- `ScopeView`, `LayerBasis`, `StaticIndex` and their staged characteristic
  readers: the explicit layer context is the surviving authority.
- `stacked_activated_ability`: resolve the effects captured at activation;
  catalog lookup is only observation provenance.
- Per-revision source construction, deep whole-board dependency clones and
  blanket release probes: immutable seeds, copy-on-write objects and conservative
  read/write summaries replace them. Preserve the fresh debug oracle.
- Eager trigger snapshots in unchanged `perform_state_based_batch` calls:
  retain one pre-batch snapshot immediately before its first relevant mutation.
- Repeated `SourcePlan::reads_predicate`/`reads_tapped` walks: one `ReadSet`
  traversal gathers characteristic, mana and tap reads, including alternatives.
- Duplicate counter/record node assembly: `Node::resolved` owns their fixed
  membership. Seeds already discover their statics, so the initial redundant
  `Context::discover` is gone; grants still rediscover. Singleton pending sets
  need no write-summary allocation or dependency scan.
- Temporary `static_sources` vectors: stream the existing immutable board
  iterator in the same order, retaining one debug validation per scan.

No public rules type, saved input, choice or behavior test is removed. Seeds
validate definition identity/incarnation/timestamp; copy/transform detaches.
Whole-board invalidation stays conservative. The reductions remove repeated
work, not cache invalidation, dynamic dependency checks or debug comparisons.

## Caller contract

The accepted concept-review amendment is implemented in
`docs/training.md#policy-inputs` and `manabot/README.md`: select v8 before
collector/model construction. Older inputs reject before buffer writes/action
prefix selection, potentially after earlier actions succeeded. Preserve failure
evidence; construct a compatible model for new work. No rollback, checkpoint
reshaping, live swap or automatic restart is recovery. The recipe is exercised
by `test_v8_recipe_collector_and_evaluation_select_saved_input`.
`scratch/concept-review.html` illustrates these separate current-view,
captured-program and saved-architecture lifetimes; it is not a browser proof.

## Remaining work

1. Inspect final-head Layer Risk `41956688122` and CI `41956687926` for
   `68ef4a63d`. The redesign is implemented and measured locally; matched isolated
   timing and the unchanged budget remain unaccepted. Investigate any remaining
   regression using retained exact counts/digests; do not claim review readiness
   from measurement checkpoint publication. Further per-component invalidation
   or instruction borrowing needs residual-cost evidence to justify complexity.
2. Retain and compare the new isolated receipts with the completed predecessor
   run `41952171271`. That run validates the producer and all seven policy cases,
   but its engine still regresses. Local source-plan measurements preserve every
   full score, action, simulation, payment/resource and step digest. The exact
   tiny-comparison passes in optimized debug locally (53.24 s); hosted long
   integration passes at preceding `a7b0ddaff`, while its budget still fails.
   Source-plan/final evidence bundles are `49f516b3b42188a9e170e59a870907632d17fbcd`
   and `6baa683f7ec901ea243538ec53e91108429f66ed`.
3. Gate owns affected suites and current-world admission: full debug Rust,
   clippy, model/checkpoint/active-training checks, and current Academy receipt/
   fixture refreshes required by the catalog/state hash. Preserve historical
   evidence. The unchanged 10% same-runner Engine Budget remains separate.
   The baseline remains unchanged: current checked counts/digests match, and
   no performance cost has been accepted. Fewer choices are not speedup evidence.
4. Preserve integration boundaries. Main's `action_overflow=first-legal` pilot
   exception follows input admission; default overflow stays error and fresh
   input stays v6. MTG-241's future retained frame/result/replacement-batch
   payloads are not claimed by current v8. Coordinate schemas through ordinary
   sync before combined publication without adding a second execution authority.
   Its later 2026-11-04 comment `cc08a059` reports integration of `1930146b0`
   at `d77bc8676` and a private-capture correction at `91517292c`: remembered
   bindings become `ObjectBinding` with exact object, captured `ZoneSelection`
   and per-seat knowledge. Retained frames/batches remain its authority.
   Causal counter provenance and logical token proposals still precede its
   complete successor projection. This is reported adjacent work, absent from
   this checkout; it supplies no combined-source or complete-v8-payload proof.

## Validation and retained sources

Check: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml` with `--test rules_tests cr_613_continuous_effects`, `--lib flow::statics::cache_tests`, and `--features benchmark --example bench_layers` passes 23 + 2 + 1 tests; `uv run --extra dev pytest tests/model/test_layer_input.py -q -k native_python_text_forward_backward_and_reload` passes 3. Release wheel rebuilt/installed atomically; rustfmt and diff checks pass. Gate owns full suites and isolated performance acceptance.

The reduction preserves all dependency step/search digests and counts, plus all
16 complete payment traces/resources. Against `68ef4a63d`, allocations fall
563,724→552,779 for dependency steps and 49,115,896→48,159,001 for dependency
search; pool/land payment allocations remain 436/509. No timing claim. Exact
receipts, source patch and binary hashes are retained in gzip tar git blob
`928f19c27f4729601867502f72b3da7180a255e6` (`git show <blob> > evidence.tar.gz`).

The complete pre-reduction plan, including uncommitted incoming notes and exact
prior test/evidence references, is git blob
`2df8f0bfe2cdec1b230766932598f536e54d0894` (`git show <blob>`). It retains the
archived implementation draft and previous concept-review/reconciliation history.
The risk and CI-repair notes keep contrary measurements and remaining obligations
current. Main sync through `51b4bd274` changed configuration only, not runtime.

Primary rules fetched 2026-11-04 from
https://media.wizards.com/2026/downloads/MagicCompRules%2020260925.txt
are preserved at blob `4dc4fead596288be1ca6aa9c913ea1739c680a29`.
CR 613.6–8, 305.7 and 714.3c/714.4 confirm: a Saga without chapters neither gains
a precombat lore counter nor sacrifices for completing chapters; setting a basic
land type removes printed/copiable abilities, not external grants. `tick.rs`/`sba.rs` now read the effective final chapter.

Scryfall API printing responses, fetched 2026-11-04:
- https://api.scryfall.com/cards/fdn/91 — Kellan, Planar Trailblazer,
  ID `f46a9329-7b91-441d-8653-50c1152c9120`, blob
  `536eb670e576265a2373af07576f4875dc7cbe53`.
- https://api.scryfall.com/cards/hob/119 — Beorn the Fierce,
  ID `367d5f8b-77ee-47f7-bc71-972d62c280a9`, blob
  `2aa675392995f0b13e724cddfa77c6d3894689ce`.
Beorn is a 6/6 Bear Shapeshifter Warrior with trample, a Bear anthem and combat
trigger, not an Adventure card. Kellan's upgrades replace creature types; they
retain previously granted abilities. These distinguish composition/timestamps
from genuine same-layer dependency.
