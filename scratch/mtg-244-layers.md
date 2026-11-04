# Composable layers — MTG-244

Jack Heart accepted one composable-layers Task on 2026-11-04: Beorn/Kellan as
real cards, historical interactions as rules fixtures. The code Graph publishes
one PR for review; no merge or experiment launch. Design **needs review**;
runtime, cards and tests are restored to base
`4463e57f5a92ebd3b60e2fb50c046b84ac2c0372`.

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

This counterexample stopped implementation. The archived draft was not built;
earlier focused engine passes do not establish end-to-end acceptance.

## Revised proposal, not accepted

Resolve the versioned policy representation before migrating engine consumers:
structured granted abilities, effective colors and keyword-counter timestamps
must reach model inference as well as native/Python/text projections. Preserve
v1–v7's widths, omitted fields, parameter shapes and timestamp normalization.
The fresh-default decision must be explicit: AGENTS currently requires v6.
Define admission/rejection for earlier inputs without silently dropping state.
No historical checkpoint, receipt or conformance corpus is rewritten.

Require native/Python parity, real forward passes through both fact encoders and
the shared v7/successor effect encoder, and retained model-configuration loading.

## Evaluator approach retained for review

`Game::characteristics` remains the sole derived battlefield authority, including
effective colors and abilities. Copy/control retain their event owners. A private
context starts from copiable values and evaluates affected objects together.
Static indexes discover candidates; final ability loss must not filter candidates
before layers. Explicit context reads prevent public-characteristics recursion.
Cache entries remain immutable and clone-safe with debug recomputation.

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

## Delete — do not maintain

These replacements remain pending the input design and complete consumer cutover;
none is a safe standalone deletion. Paths below are under `managym/src/`.

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

## Required witnesses and remaining work

All eleven supplied pairs remain acceptance work: Beorn/Kellan; Moon/Urborg;
Sun/Urborg; Moon/Saga after chapters I/II; Dress Down/Magus, Tarmogoyf and
Opalescence; Humility/Opalescence; Ygra/Imprisoned; Lattice/March;
Painter/Dress Down. Include relevant timestamp orders, dependency loops,
removal/re-entry, copy/grant/loss, real decision cloning, event-driven cache
invalidation, combat, targeting, payment and native/Python/text/policy agreement.
Beorn and Kellan still need verified complete definitions, directed card tests
and catalog/snapshot integration. Historical fixtures do not register a card pool.

Remaining: resolve input design; finish the indivisible implementation and
witnesses; verify focused debug behavior; rebuild the native extension. Gate owns
affected suites, current-world admission and the same-runner Engine Budget,
including a bounded layer-rich workload while retaining old workloads. No
exhaustive conformance, checkpoint compatibility or performance claim.

Check: `git diff --check` — passed for notes only; implementation owns focused
debug tests and extension rebuild, gate owns the affected suites above.

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
land type removes printed/copiable abilities, not external grants. Current
`tick.rs`/`sba.rs` instead inspect the printed Saga/chapter definitions.

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
