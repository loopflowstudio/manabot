# PR D: decklist import and reproducible training collections

## Accepted concept correction: checkpoints bind worlds, not training regimes

Jack Heart clarified on 2026-10-28: checkpoints should bind to worlds, whose
core contract is (1) fixed rules-engine behavior and (2) the card file. He further
stated that models/checkpoints should not really know about their training regime.
This supersedes the review's 3a/3b framing and its proposal to retain training
collection provenance inside the checkpoint. There is no remaining product
choice between a training-roster allowlist and a world-bound policy.

### Ownership and experienced behavior

- **World:** the versioned rules-engine contract and card file, including the
  authoritative semantics needed to interpret observations and legal actions.
  A matching name or tensor shape is insufficient; validate actual identities.
- **Portable model/checkpoint:** weights, architecture/inference configuration
  and a world binding. Observation/action encodings and output semantics needed
  to interpret the model are inference contracts, not training-regime knowledge.
  No training deck collection, source files, sampling policy or regime is needed
  to save, load, inspect or execute the portable model.
- **Training run:** the regime, admitted deck collection and source provenance,
  sampling policy/curriculum, seeds, metrics and producer history. It references
  the checkpoint artifact by verified identity; the checkpoint need not contain
  or load the run. This preserves provenance without making the model own it.
- **Recovery state:** optimizer, scheduler, collector/RNG and exact-population
  state owned by the run. Exact-resume checks happen here and in run admission,
  not in portable checkpoint compatibility.

A model may play any deck admitted by its bound world, regardless of which decks
were used in training. Deck construction/content admission remains shared native
domain behavior. Unsupported cards, wrong rules/content and incompatible model
I/O still reject; unseen-but-admitted combinations do not. The distinction is
compatibility, not a claim about strength. Uniform distinct-deck sampling remains
the initial run policy; curriculum is deferred and never becomes model metadata.

### Accepted usage copy (target behavior; implementation still differs)

Prepare an admitted deck collection and select it in a training run. The run
records its exact collection, sampling policy and history. Export or load the
model with its world; the model does not require the training files or regime.
Use any decklists admitted to that same world for play or evaluation, recording
actual evaluation inputs in the evaluation record. Resume a training run using
its recovery state and original run contract. A changed training population is
an explicitly recorded change of experiment/transfer, never a silent exact
resume. Cross-corpus training transfer and adaptive curriculum are still deferred.

### Implementation consequence and proof

At c76c814ae, `checkpoint_world` embeds both `setups` from the training collection
and `deck_collection`, and `validate_agent_setup` uses those setups as an allowlist.
Checkpoint/catalog code imports `DeckCollection` and re-admits its source records.
That is the wrong ownership for the accepted model. Merely weakening the roster
check while leaving regime/corpus dependencies in the model is insufficient.

Remove this coupling from the current portable model path. Derive world binding
from a world/content descriptor, not a representative training match. Move corpus
provenance and producer/regime metadata to the owning run records, retaining
artifact links and verification. Review current exporter `extra`/`bc` metadata,
producer admission and catalog summaries for the same boundary; the run should
supply training facts where a consumer needs them. Keep model architecture and
observation/action/output compatibility strict. Do not add a parallel policy
runtime, duplicate rules registry or speculative curriculum framework.

Frozen checkpoints, experiments and old schema semantics remain unchanged.
Version new portable artifacts explicitly where needed; keep historical readers
separate from current authoring rather than silently broadening old bindings.
Existing world version reservations/content ownership stay with their Tasks.

Proof: export and load a current model without a training collection, recipe or
source files available; inspect it without re-admitting training data; execute an
unseen but native-admitted same-world deck; reject unsupported definitions and
changed rules/card-file/input identities; preserve exact run recovery rejection
for changed corpus/sampler/state; verify the run still links the checkpoint and
its training provenance. No weight updates, paid compute or data download needed.

The review updates the design and screen only. Implementation remains pending;
this correction is not a claim that the existing checkpoint path already has
this separation. Working screen: `scratch/concept-review.html`.

## Accepted direction — Jack Heart, 2026-10-28

Implement, test, refine and publish one cohesive PR for review; stop before merge.
Training matchup generation is primary and Academy is a secondary consumer.
Jack Heart selected uniform sampling over distinct normalized decklists (2a),
with curriculum learning deferred. Initial real-world input will be 17Lands after
full TLA support. Preserve the completed importer while extending training.

No training campaign, weight updates for this proof, paid compute, bulk dataset
download, follow-up launches, frozen-evidence rewrites or independent TLA rules
work. Real 17Lands acquisition/schema mapping and held-out corpus policy remain
later design work. Do not change MTG-175's reserved curriculum experiments.

Prepare complete decks against an explicit qualified pool. Reject the entire
collection on invalid input, retaining source/line diagnostics; never silently
filter cards or decks. Collapse identical normalized main/sideboard lists while
retaining every source record. Use one fixed whole-pool policy vocabulary across
all sampled pairs. Record game starts separately from learner transitions:
uniform starts do not imply uniform transition exposure.

`uniform-distinct-v1` draws two different normalized deck IDs with unbiased
ordered seats using the native reset seed. Excluding mirrors is an implementation
assumption following the different-decks request, not a separate explicit user
choice (scratch/questions.md). Require at least two lists; no pair table,
adaptive weights or speculative curriculum framework.

## Contract and ownership

- Content Task 892c47 owns canonical card authoring, qualification, compiled
  content and learning identity. Consume native `content_pool(pack_key)` and
  `pool_content(pack_key)`; registration or authored availability alone does not
  make a card eligible for novel combinations. No second rules registry or
  generic native fallback for imported files.
- This Task owns counted-lines-v1 parsing, named normalized records, shared
  native construction admission, collection manifests and consumer entry points.
  Preserve source bytes including BOM/CRLF, filenames and line diagnostics.
  Resolve front/full DFC names; reject reverse-only, ambiguous, unknown and
  unsupported entries without substitution.
- `development-v1` reflects established construction behavior: nonempty main,
  positive counts, eligible non-token definitions in both sections. It claims no
  tournament legality or size/copy/color restrictions. Operational ceilings are
  64 KiB per source and 1,000 total cards per player. Sideboards are fixed
  outside-game copies; no between-game sideboarding, draft/sealed UI, Commander
  or multiplayer is introduced.
- Existing authored presets, exact checkpoint world/input binding, native
  Commands, stale rejection, pending decisions on reconnect, replay/fork identity
  and viewer safety remain required. Imported provenance stays private.
- Reuse native `VectorEnv` reset/sampling authority. PR383/MTG-175 owns adaptive
  reassignment and curriculum experiments; PR389/19d240 owns shared simulation
  and evaluation APIs. MTG-177 owns target-resolution changes.

Integrated content dependency: 73865361f via merge 2806d03d5, then qualified
checkpoint 1e0d3c998 via merge 4e206ce3b. This supplies tla-pool-v1, w16/schema 8,
31 qualified cards and corrected modal dual-land mana; Rocky Rebuke remains
unsupported. These are dependency checkpoints, not final upstream qualification
or PR readiness. Review importer changes against that dependency, or revalidate
after it lands; generated content is not importer-owned work.

Training overlap reviewed at PR383 d0f48ca96 and PR389 c85470a59; scoped coordination
is retained in Task 1d7884 comment 2c523295-b690-4e45-a989-1dd37b91cabb. Adaptive
reassignment, experimental distributions and shared simulation APIs are untouched.

## Implemented path and remaining proof

The importer connects parser/resolution to native admission, sim file flags and
Academy's editable two-file validation/start flow. Retained imports bind exact
pool/content identity and reject source/setup tampering. Simulation supports
random/default players; Academy supports built-in opponents. Imported setup
checkpoint opponents remain rejected until consumer compatibility admission
exists. Current collection-trained exports bind all member decks; this is superseded by
the accepted world-only checkpoint design above and needs correction.

Training configuration embeds a loaded, versioned collection rather than keeping
a mutable file pointer. Set `deck_collection: path/to/manifest.json` and
`agent.semantic_pack: tla-pool-v1`; an omitted representative match is derived.
Stages require current-self behavior, one producer, ordinary roots and the
fixed curriculum field. See docs/decklist-import.md for preparation and usage.

Collection digest binds normalized membership, all retained source records,
content/world, schema and sampling policy. Native roots choose actual decks;
collector counters, snapshots, exact/current-game recovery, checkpoint/catalog
admission and failure records currently retain those identities. Keep them in run
and recovery records; remove the collection dependency from portable model
checkpoint/catalog admission as directed above. Failure capture journals
isolated steps and synchronizes replacement roots through CapturedVector.

Acceptance: three or more eligible non-preset decks; changing pairs over resets;
identical pair/seat/seed and policy RNG after recovery; reject invalid/unsupported
input before collection; reject changed corpus/content/schema on resume; retain
actual sampled IDs in failure/capture records; one compatible policy vocabulary.
Use bounded fixtures and the actual collector reset path, without weight updates.

Gate/demo/delivery own full affected suites, acceptance review, final upstream
content revalidation and publication. The world-only checkpoint correction above
is now required implementation work. Existing focused evidence: native admission/catalog tests, 50 Python
import/startup/identity/reconnect/fork/stale/source checks, frontend check/build
and a real browser import/reconnect proof passed. The documented fixed-seed CLI
proof stopped after four decisions, deliberately unfinished, with setup identity
0471df58fab1310ced94add4190c76509abb878f8df191c601c12bcafc8fe3fa.
Training proof at d8fc9833c passed nine collection tests and one existing Elemental
isolated-failure regression, plus the focused debug native collection-root test;
the release extension was rebuilt and atomically installed. These are focused
results, not full gate certification.

## Delete — do not maintain

Completed: remove explicit-pack authored-only selection from semantic/content.rs;
retain implicit presets alongside qualified novel admission. Remove unused
`env.match.parse_deck` and its exclusive TestDeckParsing/deck_formats tests.
Move deck hashing from rollout accounting to shared decklist ownership, preserving
its existing serialized identity. Generalize the existing root sampler's admission
without changing Elemental's exact roster, mirrors or RNG sequence.

Collection preparation groups admitted records once by deck ID, so membership
and provenance cannot drift between parallel maps. Both file consumers share the
bounded UTF-8 reader, preserving their existing filename conventions. Setup
hashing stays local: importing infra.identity here reenters infra.config and
breaks Academy's fresh-process import boundary.
No remaining deletion targets in this training pass; no card-authoring deletion
belongs here. The Academy redesign below is explicitly deferred.

## Secondary Academy direction (1a, accepted; not implemented)

Jack Heart selected one visible match draft with two per-seat deck sources
(preset or imported) and an explicit “Use the same list” opponent operation.
Current UI still has competing preset selectors and a whole-match import override;
its successful deck check says Ready before opponent compatibility is checked.
Importing one person's deck currently requires a second file. Training priority
means this review does not authorize launching that redesign now.

When taken up, replace the override with one draft, retain editable source on
errors, and check the selected opponent before “Ready to play.” Deck-only checks
should say “Decks valid.” Keep incompatible opponent choices visible with an
explanation; never silently switch them or replace a live game on failed setup.
Mixed presets/imports must satisfy pool eligibility, regardless of prior curated
permission. Mirror copies a list explicitly; it adds no game rule. No saved deck
library or deckbuilder. Proposed proof covers one-file mirror, distinct imports,
line correction, incompatible presets/opponents and no live-game replacement,
plus existing Command/reconnect/replay/fork boundaries. The original alternative
1b and detailed usage sketch remain at d8fc9833c:scratch/f871a8-import-decklist-files.md;
working screen is scratch/concept-review.html.

## Retained context and handoff

Original accepted eight-PR TLA brief and detailed history:
d8fc9833c:scratch/f871a8-import-decklist-files.md. Source exploration is Task 0d9ba3;
base c5d406e3d2a701525c15c8500a30f7da7896cc40. Historical audit/handoff is preserved
at .lf/tmp/decklist-evidence/decklist-research-20261028.tar.gz. Final research copy
is .lf/tmp/decklist-evidence/full-tla-final/ with plan digest
7cdb6308f8750ec6d1e30ca485e1676c53af5456401fffa3cb88f10c496c07ef.

Preserve local copied scratch/reference, scratch/tla-evidence and
scratch/tla-card-ledger.jsonl; local Git exclusions now keep these out of staging.
Their bytes remain in this checkout. Prior staging failure was
recovered with scoped checkpointing and `lf sync --manual`. Before a command-step
handoff, stage explicit owned paths and use `lf commit --no-add`. Publication
belongs to a later Graph step, with no merge or follow-up launch.

The manifest references `.lf/wave/engine/GOAL.md` and `MEMORY.md`, but neither
exists in this checkout. The supplied Task, accepted design and dated Steers
provide the implementation scope; Wave context was not reconstructed.

Check — `uv run --extra dev --extra artifacts pytest tests/academy/test_advice.py::test_default_play_import_does_not_require_training_dependencies tests/training/test_deck_collection.py tests/semantic/test_decklist_import.py tests/academy/test_decklist_setup.py -q`: 28 passed; existing Elemental recovery/refill checks: 5 passed; Ruff and diff checks passed. Gate owns full affected suites (debug Rust, training recovery/learning-state/Elemental and combined CI collection), per docs/development/validation.md.
