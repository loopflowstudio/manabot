# Composable layers — MTG-244

Jack Heart accepted one composable-layers Task on 2026-11-04: Beorn/Kellan as
real cards, historical interactions as rules fixtures. The code Graph publishes
one PR for review; no merge or experiment launch. On 2026-11-04 Jack Heart
approved a new input version: "new input version is totally fine". The prior
input-version blocker is resolved. The implementation started from
`4463e57f5a92ebd3b60e2fb50c046b84ac2c0372`; reconciliation on 2026-11-04
uses `fb3555ffbd36bb797908110fc549999fe8e1a381`, which integrates main
`1546aae3fa7b207b2c5e8ac92a6571af036ddf83`.

Implementation 2026-11-04: recovered the archived draft and replaced the old
staged evaluator with a complete board context. The branch now contains the
engine/consumer cutover, both real cards, all eleven interaction fixtures and
explicit v8 encoders/model support. No merge, world freeze
or experiment launch is selected.

Status: implemented with focused correctness evidence; not yet ready for review.
Jack Heart's 2026-11-04 risk-measurement direction remains accepted work before
demo/PR, detailed in [mtg-244-risk-validation.md](mtg-244-risk-validation.md).
Local-fact isolation, numeric saturation and unnecessary mana-board rebuilding
are repaired. Shared-program and real configuration-selection checks pass;
allocation, operation and buffer costs are measured. Clean comparative engine/
full-policy timings remain unavailable on the contended local host, and combined-
head Gate checks remain outstanding. No further product decision is needed.

Accepted outcome: one `Game::characteristics` authority composes types, colors,
abilities and power/toughness for rules and policy consumers, preserving a cheap
common path, immutable revision caches and debug recomputation. This Task owns
layers and their cache dependencies. Composable execution owns pending frames,
replacement proposals and contextual source/entry callers; mana consumes the
layered abilities, and naming owns large choices. Those adjacent refactors are
not prerequisites for unrelated simple card definitions.

## Counterexample to the draft design

The pre-v8 design could not extend frozen model architectures in place:

- `manabot/env/observation.py` fixes `FACT_KINDS = 10`.
  The retained-input paths in `manabot/model/facts.py::FactEncoder` and
  `manabot/model/state_records.py::SharedFactEncoder` allocate eleven kind
  embeddings and reject larger IDs.
- The retained v7 `manabot/model/shared_effects.py::SharedEffectEncoder` allocates four
  detail-kind embeddings and admits only kinds 1–3 (keyword, subtype, color).
  The draft's complete granted-ability
  program needed structured tokens and numeric values beyond those categories.
- Native/Python tensor parity alone would miss this: both encoders could agree
  on arrays that policy inference rejects. Enlarging the tables in place changes
  existing checkpoint parameter shapes; repurposing old kinds changes their
  meaning. Hashing a whole granted program into one token also loses the shared
  compositional vocabulary required by the Task.

This counterexample stopped the earlier draft; explicit v8 now resolves it.
Earlier focused engine passes did not establish end-to-end acceptance.

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
Tap predicates/source conditions require cache invalidation; `tap` now advances
the characteristic revision, as does `untap`. Trigger candidates consume
layered grants and suppression, with exact departed-object ability information.
Mana writes retain the raw board unless any discovered layer node reads unspent
mana, including nested conditions and granted statics. Finalized activation
outputs and trigger indexes still refresh; debug recomputation checks both views.
The isolated pool-write probe falls from 471 allocations to 10; tap invalidation
is unchanged. This does not establish whole-game throughput.

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

## Completed simplifications

The explicit board context replaces `LayerBasis`, `ScopeView`, recursive stage
readers and final-loss-filtered candidate discovery. Evaluated characteristics
replace printed battlefield colors, redundant `CardDefinition` queries and the
separate lost-keyword/attached-ability paths. Keyword counters retain timestamps.
Stacked activations resolve their captured programs rather than looking them up
again on the printed definition. Event snapshots directly supply granted triggers.
The redundant node flag and `ContinuousEffect::follows` are removed.

V8 local-fact recurrence now batches independent live programs, validates ordinal
and provenance boundaries, preserves duplicate occurrences and skips recurrence
when no program exists. This fixes cross-program contamination and reduces the
measured component cost; it does not establish full-policy performance or learning
quality. Required scopes, attachments, activation pricing and source-independent
resolution remain on the surviving authorities.

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

Completed cuts; no remaining deletion targets identified:

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
- Duplicate Kellan walkers in tests and the policy benchmark are replaced by
  `verify.representation::{kellan_upgrade_root,kellan_trigger_root}`. Commands,
  bounded prefixes, retained inputs, program ownership and gradients are preserved.

## Remaining work after risk implementation — 2026-11-04

- Clean performance evidence: the default TLA benchmark is separate again;
  `bench_layers` supplies a true Moon/Urborg dependency with complete per-decision
  score digests, counts and cache/dependency instrumentation. One original step
  run retained 101,650 steps and the reference digest. Original search was stopped
  after severe external host contention was discovered (load average >500).
  Completed local latencies are exploratory, not comparative acceptance. Clean
  parent/change step/search total/tail, full-policy inference/backward/encoding
  latency and peak-memory comparisons remain unavailable here. Gate's isolated
  runner owns the 10% budget; no reference refresh is authorized to hide a cost.
- Implemented risk evidence: pool-write allocations 471→10 and requested bytes
  133,918→2,054; shared GRU dispatches 16→1 at batch 16. Buffer/model bytes and
  shared/local occupancy are measured for explicit v6/v7/v8. The numeric collision
  (4096 versus 8192) is repaired. Shared program order, timestamps, source/target
  joins, duplicates, independent record order, remapped objects, padding and
  gradients pass, alongside all three v8 model paths. A production collector
  obtains one v8 transition; saved-checkpoint evaluation admission selects v8.
  No optimizer or experiment launch. Raw receipts, commands, hardware, code
  provenance and limitations live in the risk note's evidence blob.
- Integration: main's explicit `action_overflow=first-legal` pilot exception is
  separate from input admission; both encoder boundaries validate the selected
  input before prefix selection, while default overflow stays error and fresh
  input stays v6. Existing pilot protocols/evidence do not authorize a launch here.
  MTG-241's Task comments identify future retained frame/result/replacement-batch
  payloads. Current v8 carries layer and captured-program payloads only; it does
  not claim to encode those unpublished execution structures. Ordinary sync and
  schema coordination remain necessary before combined publication, preserving
  already frozen versions. No second execution authority is part of this Task.

Gate owns affected suites and current-world admission, including current Academy
receipt/fixture digest refreshes required by the new catalog and state hash.
Preserve historical evidence; current-world corpora/consumers must agree before
publication. Gate also owns the full debug Rust suite, clippy, model/checkpoint
and active-training checks in `docs/development.md`. Engine Budget retains its
10% limit. Changed digests or workload definitions require the normal reference
procedure using the candidate's own CI artifact and a documented reason,
including the baseline's seeded-search assertion in `tests/infra/test_engine_budget.py`.
A refresh cannot erase an unaccepted measured regression; old/new timings remain
visible even when corrected semantics prevent digest equality. Captured stack
programs and added roots can change digests; fewer choices are not speedup evidence.
Learning efficacy remains unmeasured without a separately authorized experiment.

The post-sync overflow fixture correction uses real observations and checks
input admission before action-prefix selection on both encoder boundaries.
Earlier focused results and their exact scope are preserved in the pre-compression
plan, blob `813f319d781339abaee7dfe47cdfc74db6a99b8d`; the risk note retains raw
measurement evidence. They do not certify subsequent edits or combined-head Gate.

Check: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml`
with `--lib mana_revision_reuses_independent_board_but_reprices_dynamic_outputs`,
`--test rules_tests cr_613_continuous_effects` and `--example bench_search` — 27 passed;
`uv run --no-sync --extra dev --extra artifacts pytest tests/model/test_layer_input.py tests/model/test_shared_effects.py -k 'not ordinary_learner' -q`
— 22 passed, two optimizer smoke cases deselected. The optional dependency benchmark
compiles, the root wheel was rebuilt/installed atomically, and format/Ruff/diff
checks pass. Gate retains full suites and clean comparative performance acceptance.

The consolidated fixture's `bench_layer_input.py --version 8 --state program --inventory`
receipt (blob `81fa8070f9ff772ff01b2eab6c210e2ac72e362b`) matches the retained
observation SHA256, configurations, 208,368 input bytes and 21/11 live detail/fact
rows. No new timing claim follows from immutable-view reuse or these reductions.

## Preserved evidence

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
