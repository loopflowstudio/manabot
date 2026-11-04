# Composable layers — MTG-244

Jack Heart accepted one composable-layers Task on 2026-11-04: Beorn/Kellan as
real cards, historical interactions as rules fixtures. The code Graph publishes
one PR for review; no merge or experiment launch. On 2026-11-04 Jack Heart
approved a new input version: "new input version is totally fine". The prior
input-version blocker is resolved; continue implementation under the rollout
below. Checkout base is `4463e57f5a92ebd3b60e2fb50c046b84ac2c0372`.

Implementation 2026-11-04: recovered the archived draft and replaced the old
staged evaluator with a complete board context. The branch now contains the
engine/consumer cutover, both real cards, all eleven interaction fixtures and
explicit v8 encoders/model support. No merge, world freeze
or experiment launch is selected.

Accepted outcome: one `Game::characteristics` authority composes types, colors,
abilities and power/toughness for rules and policy consumers, preserving a cheap
common path, immutable revision caches and debug recomputation. This Task owns
layers and their cache dependencies. Composable execution owns pending frames,
replacement proposals and contextual source/entry callers; mana consumes the
layered abilities, and naming owns large choices. Those adjacent refactors are
not prerequisites for unrelated simple card definitions.

## Counterexample to the draft design

New fact/detail kinds cannot extend frozen model architectures in place:

- `manabot/env/observation.py` fixes `FACT_KINDS = 10`.
  `manabot/model/facts.py::FactEncoder` and
  `manabot/model/state_records.py::SharedFactEncoder` allocate eleven kind
  embeddings and reject larger IDs.
- `manabot/model/shared_effects.py::SharedEffectEncoder` allocates four
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
Tap predicates/source conditions require cache invalidation; current `tap` does
not advance the characteristic revision. Trigger candidates must account for
layered grants and suppression, with exact departed-object ability information.

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

## Delete — do not maintain

Completed replacements in this indivisible cut. Unprefixed flow paths are under `managym/src/`.
Required scopes, attachments, inherited abilities, activation pricing and
source-independent resolution remain on the surviving authority.

- `flow/resolution.rs::stacked_activated_ability`: delete printed-definition resolution lookup; preserve independent source lifetime and retargeting using captured effects.
- `LayerBasis`, `ScopeView`, and recursive stage readers in `flow/statics.rs`:
  replace with explicit context reads and final-view readers. Preserve current
  scopes, conditions, attachments, inherited abilities, activation pricing and
  dynamic mana-output behavior.
- `build_static_index` filtering `StaticIndex.sources` by final ability loss,
  and `land_type_overwrite`/`DerivedCache.land_type_setters` in `flow/statics.rs`:
  replace with candidate discovery and evaluated abilities. Preserve CR 305.7
  intrinsic mana and retained external grants.
- `flow/statics.rs::colors_among` reading printed battlefield colors: read
  effective colors.
- Counter keywords treated as printed abilities: preserve counters separately
  with timestamps so later counters survive earlier ability loss.
- `flow/effects.rs::lost_keywords`'s unconditional union and final-loss-based
  attached-ability filtering: the common ordered layer application owns these
  semantics.
- `state/src/card.rs::CardDefinition::{final_chapter, library_top_casting,
  attacks_despite_defender, pt_buffs, keyword_grants, triggered_grants,
  additional_trigger_predicates, restrictions, type_grants}` and
  `state/src/effect.rs::ContinuousEffect::follows`: obsolete printed/staged
  queries with no remaining callers or exclusive tests. Their consumers now
  read evaluated characteristics. No compatibility shim is needed.
- `flow/triggers.rs::granted_triggered_abilities`: its sole remaining operation
  cloned the event snapshot's grants. Iterate that snapshot directly.
- `flow/layers.rs::Node.static_ability`: redundant with captured targets versus
  a still-live static source; remove the extra state bit.
- `manabot/model/facts.py`'s recurrence across the whole fact buffer: replace
  with independent live programs, preserving duplicate occurrences and gradients.

## Required witnesses and remaining work

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
and installed atomically; six Python checks cover v8 rows/typed forward/backward,
checkpoint reload, native/Python parity, scalar/vector schema selection, retained
v5–v7 architectures and the live Kellan trigger after its source dies. The new
`semantic-kernel-w17-v2` corpus records 14 cases / 3,473 commands / 15 Phase rows.
The old corpus remains unchanged because w17 is open, not frozen. The common release P/T path skips
pairwise dependency clones when no instruction reads power; debug compares it
against the dependency computation.

Gate owns affected suites and current-world admission, including current Academy
receipt/fixture digest refreshes required by the new catalog and state hash.
Preserve historical evidence; current-world corpora/consumers must agree before
publication. Gate also owns the full debug Rust suite, clippy, model/checkpoint
and active-training checks in `docs/development.md`. Same-runner Engine Budget
must adopt the PR's own run after review (including the baseline's seeded-search
assertion in `tests/infra/test_engine_budget.py`): original TLA workloads remain and a
bounded dense-board root is added (16 step games, 8 search decisions). Digests
change with captured stack programs and added roots; fewer choices are not
speedup evidence. No exhaustive conformance, historical-checkpoint admission or
performance claim is made here.

Check: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 --test rules_tests cr_613 -- --test-threads=2` — 21 pass; card filters `fdn::` / `hob::` — 3 pass; `uv run --no-sync python -m pytest tests/model/test_layer_input.py -q` — 13 pass against the rebuilt, atomically installed release extension; formatting, Ruff and `git diff --check` pass. Gate owns full affected suites; remaining risk measurements are required before demo/PR.

## Preserved evidence

Pre-compression plan and exact earlier test filters:
`929d8324becbb243bdb88809d9670dada8ef11f3`; decision note:
`3059c4d7a78ecd3f358a027d25c4208c376c52a2`.

Complete uncommitted implementation patch, including new files, is git blob
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
