# Composable layers — MTG-244

Jack Heart accepted one composable-layers Task on 2026-11-04: Beorn/Kellan as
real cards, historical interactions as rules fixtures. The code Graph publishes
one PR for review; no merge, world freeze or training launch. Jack Heart approved
a successor policy input the same day: "new input version is totally fine".
Explicit v8 is implemented; fresh v6 and frozen saved versions remain unchanged.

Status: PR #427 published; first CI found a large engine regression and stale
current-world Academy fixtures. Repair is in progress; isolated risk receipts
and combined-head Gate checks remain outstanding. No human decision is
missing. [Risk validation](mtg-244-risk-validation.md) owns measured results,
configurations, exact sources and limitations; remaining acceptance is listed below.

Accepted outcome: one `Game::characteristics` authority composes types, colors,
abilities and power/toughness for rules and policy consumers, preserving a cheap
common path, immutable revision caches and debug recomputation. This Task owns
layers and their cache dependencies. Composable execution owns pending frames,
replacement proposals and contextual source/entry callers; mana consumes the
layered abilities, and naming owns large choices. Those adjacent refactors are
not prerequisites for unrelated simple card definitions.

The earlier draft could not add program kinds to frozen v5–v7 embeddings
without changing saved parameter shapes and meanings. Explicit v8 resolves that
counterexample; tensor parity alone never established model admission. The full
prior plan is git blob `f8ff5bc693386baa3e247936684437da3ca0443d`.

## Approved successor input; implementation design

Implemented versioned policy representation:
structured granted abilities, effective colors and keyword-counter timestamps
must reach model inference as well as native/Python/text projections. Preserve
v1–v7's widths, omitted fields, parameter shapes and timestamp normalization.
Use an explicit successor input (v8), with fresh construction
still v6. This is the operator's implementation choice following approval of
a new version, preserving the standing default; no default migration was
requested. Layer-capable callers select the successor explicitly. Account for
ordinary default callers encountering unrepresentable state through the shared
input rejection contract below, and exercise the successor end to end.

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
all cached views and indexes. The earlier isolated pool-write mitigation reduced
471 allocations to 10; the CI repair also removes independent tap rebuilds.
These component changes do not establish whole-game throughput acceptance.

## Implementation correction — 2026-11-04

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

Directed witnesses cover all eleven supplied pairs: Beorn/Kellan; Moon/Urborg;
Sun/Urborg; Moon/Saga after chapters I/II; Dress Down/Magus, Tarmogoyf and
Opalescence; Humility/Opalescence; Ygra/Imprisoned; Lattice/March;
Painter/Dress Down. Include relevant timestamp orders, dependency loops,
removal/re-entry, copy/grant/loss, real decision cloning, event-driven cache
invalidation, combat, targeting, payment and native/Python/text/policy agreement.
Beorn and Kellan have verified definitions, real payment/combat/trigger tests
and catalog/Scryfall snapshot integration. Historical fixtures do not register a
card pool. The Saga chapter-II fixture uses the existing Ballistic Boulder
Construct token as a marker for its retained creation program; Saga token sizing
is not a layer assertion. Generic grant/copy and sacrifice/suspension fixtures
also execute the captured activation programs.

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

Risk evidence required by Jack Heart remains open before review readiness; see
[mtg-244-risk-validation.md](mtg-244-risk-validation.md). The native wheel was rebuilt from the root
and installed atomically; the focused Python module covers v8 rows/typed forward/backward,
checkpoint reload, native/Python parity, scalar/vector schema selection, retained
v5–v7 architectures and the live Kellan trigger after its source dies. The new
`semantic-kernel-w17-v2` corpus records 14 cases / 3,473 commands / 15 Phase rows.
The old corpus remains unchanged because w17 is open, not frozen. The common release P/T path skips
pairwise dependency clones when no instruction reads power; debug compares it
against the dependency computation.

## Delete — do not maintain

Completed cuts and current CI repair:

- Replace repeated per-effect discovery scans and independent-effect suffix
  comparisons with initial/grant-triggered discovery and one ordered independent
  batch. Preserve dependency reevaluation and the complete debug pairwise oracle.
- Replace unconditional tap-driven board eviction with a tap revision and
  discovered reads; preserve dynamic activation refresh, layer-dependent trigger
  kinds, stun counters, nested predicates and cloned-game isolation.
- Replace BTreeMap-backed derived lookups with permanent-slot vectors, matching
  the existing native identity convention. Keep immutable Arc views and holes.
- Box the large expression/trigger variants of private layer operations so every
  resolved per-target instruction need not reserve space for an 840-byte buff.
  Rules payloads and serialized inputs are unchanged.
- Replace per-permanent whole-board recomputation in trigger, mana, SBA and
  observation scans with immutable batch reads. Keep scan order, the raw-board
  oracle on every cache read and independent finalized-view debug comparisons;
  singular reads retain their existing cache path.
- Replace repeated board reads in CDA counts, shared-creature-type counts,
  fixed mana totals and static play permissions/costs with the same batch API.
  Entry replacements borrow layered abilities and retain their original indexes;
  remove the cloned ability lists and reuse the play-permission helper for
  flashback. These readers need no activation finalization. No exclusive tests
  or fixtures are removed, and no deletion target remains from this cut.

Earlier completed cuts:

- `LayerBasis`, `ScopeView`, recursive stage readers and final-loss-filtered
  discovery are replaced by `layers::Context`. Unused `CardDefinition::has_static`
  and `CardPredicate::reads_types` are removed; no exclusive fixtures remain.
- Combined TLA/added-board benchmark totals are replaced by explicit `--layers`
  runs. Shared search selection/digest helpers retain the original tie order and
  full-score wire identity; both benchmarks use them.
- Unconditional mana eviction is replaced by discovered condition dependencies,
  using the existing `Condition::reads_mana_pools`. Finalization reuses the raw
  immutable view when it has no activations; only activation pricing needs a copy.
- Whole-buffer v8 program recurrence and raw saturating integer projection are
  replaced by packed independent programs and signed log magnitudes. Shared
  membership routing has one implementation; v7 retains its original reduction.
- Glob-discovered policy cases and count-only receipt completeness are removed;
  the producer/comparator share exact case identities. No exclusive fixtures remain.
- The unqualified v7 complete-effect claim and stale 67/55 object widths in
  usage docs are replaced by explicit v8 selection/recovery and selected-space
  layouts; retained inputs and their compatibility boundaries remain documented.
- `SHARED_EFFECT_SHAPES` and its copy/override path are removed; the selected
  version's `shared_effect_shapes` owns the layout directly. Local program
  unpacking uses Torch's `unpack_sequence`; independent program ordering,
  parameter shapes and gradients retain their existing regression witnesses.
- Duplicate Kellan walkers in tests and the policy benchmark are replaced by
  `verify.representation::{kellan_upgrade_root,kellan_trigger_root}`. Commands,
  bounded prefixes, retained inputs, program ownership and gradients are preserved.

## Concept review — 2026-11-04

Accepted amendment: explicit input selection and failure recovery belong in the
caller-facing usage contract. Implemented in `docs/training.md#policy-inputs`
and `manabot/README.md`; reconciliation at `64e6bca33` found no open product
choice. This follows Jack Heart's successor-input approval, not a default
migration or checkpoint conversion.

New layer-capable work selects v8 before collector/model construction. Older
inputs reject before buffer writes/action prefix selection; a later observation
can fail after earlier actions succeed. Preserve failed-run evidence, construct
a compatible model for new work, and retain saved inputs and world admission.
No rollback, weight reshaping, live swap or automatic restart is recovery.
The recipe matches `test_v8_recipe_collector_and_evaluation_select_saved_input`.

The current view, captured programs and saved architecture have distinct
lifetimes. Caches remain internal. `scratch/concept-review.html` illustrates
this contract; it is not a browser capture or performance proof. Exact earlier
review/check details remain in blob `f8ff5bc693386baa3e247936684437da3ca0443d`.

## Remaining work

Jack Heart's subsequent [speed direction](mtg-244-speed-direction.md) supersedes
budget-only repair: profile and compare concrete hot-path designs, implement the
strongest supported design, and aim to beat the pre-layer engine on matched
ordinary workloads. Realign/loop-or-next must return to implementation as needed;
this compress pass and a measurement checkpoint are not completion.

1. Repair the failures retained in [CI repair](mtg-244-ci-repair.md). PR #427
   exists; its first isolated workflow failed to acquire the reference commit.
   Keep the original performance baseline and collect a corrected run. The configured producer runs original TLA baseline/before/after,
   dependency before/after, complete payments and seven full-policy cases per
   revision. Require full score digests/counts, search samples, bounded 128-step
   prefixes, native/Python encoding, inference/backward throughput and process
   RSS. Mitigate demonstrated regressions and remeasure before review readiness.
   Standard Engine Budget does not execute the dependency/model harnesses.
   Retained contended timings remain exploratory; no further
   product decision is needed. [Risk evidence](mtg-244-risk-validation.md) owns
   measured results, exact sources, configurations and limitations.
2. Gate owns affected suites and current-world admission: full debug Rust,
   clippy, model/checkpoint/active-training checks, and current Academy receipt/
   fixture refreshes required by the catalog/state hash. Preserve historical
   evidence. The unchanged 10% same-runner Engine Budget remains separate.
   A necessary digest/reference refresh uses the candidate's own CI artifact,
   a documented reason and the seeded-search assertion in
   `tests/infra/test_engine_budget.py`; it cannot hide a regression. Fewer
   choices are not speedup evidence.
3. Preserve integration boundaries. Main's `action_overflow=first-legal` pilot
   exception follows input admission; default overflow stays error and fresh
   input stays v6. MTG-241's future retained frame/result/replacement-batch
   payloads are not claimed by current v8. Coordinate schemas through ordinary
   sync before combined publication without adding a second execution authority.
   Its 2026-11-04 Task comment `dffd3f70` additionally identifies retained
   `ZoneSelection`, state-owned `LibraryEnd`, optional `Move.library_end` and
   public precommit entrants as pending integration interfaces. That report
   supplies no combined-source proof or independently reserved input version.

Earlier validation commands and outcomes remain at
`64e6bca33731010843531fff4cdabd177d08bb5e:scratch/mtg-244-layers.md` and
payment bundle `88d6abd34d8b56a9606c3ec58696063955d08226`.
Current repair results and remaining isolated acceptance are in
[mtg-244-ci-repair.md](mtg-244-ci-repair.md); earlier passes do not supersede
its first-publication CI failures.

Compress validation, debug `CARGO_PROFILE_DEV_OPT_LEVEL=1`, root commands:
- `cargo test --manifest-path managym/Cargo.toml -j 2 --test rules_tests cr_613_continuous_effects`: 22 passed.
- `cargo test --manifest-path managym/Cargo.toml -j 2 -p managym-cards --test cards -- iroh hakoda zhao lieutenant suki toph_enters team_avatar beorn kellan`: 25 passed after replacing Zhao's obsolete layer-6-removes-layer-4 assertion with retained type and fork-departure checks.
- `cargo test --manifest-path managym/Cargo.toml -j 2 --lib -- cache_tests flow::mana::tests`: 3 passed. Format/diff checks pass; Gate owns full suites and isolated timings.
- Release wheel rebuilt via the documented root command and installed atomically; `uv run --extra dev pytest tests/model/test_layer_input.py -q -k native_python_text_forward_backward_and_reload`: 3 passed.

## Preserved evidence

The complete plan before this reduction, including uncommitted concept-review
and documentation evidence, is git blob `6486ea3e1dde73d3977e8ce160ada16962ddfe6a`.
Implementation started at `4463e57f5a92ebd3b60e2fb50c046b84ac2c0372`;
reconciliation uses `64e6bca33731010843531fff4cdabd177d08bb5e`.
Runtime mitigation: `b2fb06cbadb8978d1f2373c9b2490551c358c907`; measurement/payment
checkpoint: `fb264f7c78dcd1a864cbe35a9e7d4cd985f9faf7`. Main's intervening
`115f88165` / `510b5ec35` Wave changes introduced no runtime diff. Coverage memory
and goal govern this Task; its Wave has no immediate child memory directories.

The complete pre-compression plan, including detailed validation scope and source
history, is blob `7b17e207b67ed3d9785e9915a69439a1e6eb44a1` (`git show <blob>`).


Pre-compression plan and exact earlier test filters:
`929d8324becbb243bdb88809d9670dada8ef11f3`; decision note:
`3059c4d7a78ecd3f358a027d25c4208c376c52a2`.

The archived pre-implementation draft patch, including new files, is git blob
`f5de99621d9b59ad3a7464c2feaed8b2d0ebe519` (153,208 bytes).
It can be inspected with `git show <blob>` and recovered as a patch against the
base. It is an incomplete draft, not an approved migration or deliverable.

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
