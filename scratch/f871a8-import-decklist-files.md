# PR D: decklist import and reproducible training on decklists

## Accepted correction: one deck concept, validated against the world

Jack Heart clarified that training receives a verified world with its card set,
then supplies decks during training. A fixed list may be validated upfront, but
pre-registering every possible training deck must not define compatibility.
Decks may eventually be computed or modified as training progresses. He also
rejected the confusing distinction between `CollectionDeck`, deck, and
`DeckCollection`: a deck is a deck, regardless of where it came from or how it
was selected. These statements supersede any fixed-population requirement below.

### Accepted experience and ownership

- **World:** fixed rules behavior and card definitions; shared native validation
  decides whether a submitted deck can execute in that world.
- **Deck:** main-deck and sideboard card counts. Identity is derived from that
  content. File text, filenames and parse diagnostics describe an import/source;
  they do not create a special kind of deck. Computed decks need no invented file.
- **Training:** chooses or produces decks and submits them at game creation.
  Initially this can sample uniformly from a supplied list. A fixed list is
  checked upfront so bad input is reported before training. Future computed or
  modified decks pass the same world validation before use. All submitted decks
  must be valid; a cached validation result for an unchanged deck/world can avoid
  repeating work without turning registration into permission.
- **Checkpoint:** binds the world and inference contract, never the list of
  training decks or the method of choosing them.
- **Run and recovery:** retain the actual decks used, selection/generation state,
  seeds and changes needed to explain and reproduce continuation. An evolving
  deck sequence can be part of one run. Exact resume means restoring that
  process faithfully, not requiring an eternally fixed population. An unrecorded
  substitution or unrecoverable source state must not silently pass exact resume.

### Usage draft and implementation direction (implemented 2026-10-28)

Choose world W. Supply decks A, B and C; validate them against W and sample
matches. Later, a deck-producing procedure could modify B into D and submit D.
If D passes W's validation, it can start a subsequent game with the same model
and world. An unsupported card rejects D before game creation with a diagnostic;
it does not alter a game already underway or silently replace the deck. Correct
the deck and submit it again. Exact failure/retry policy for a future adaptive
producer remains future work.

Consolidate around one shared deck representation; reuse or reshape existing
domain types rather than adding a new nominal wrapper. Separate parsing/source
records, world validation, training selection and recovery by responsibility.
Remove `CollectionDeck` as a separate domain concept. A list of decks may be
stored as data, but `DeckCollection` must not be a mandatory admission lifecycle
or own the sampler/model-input schema. Review native `RootRoster`/collection
setup and collector admission as well as checkpoint code: removing only the
checkpoint coupling would still leave pre-registration as a training constraint.
Preserve the existing Elemental sequence and historical evidence.

The immediate path can remain fixed-list uniform sampling. Do not implement a
curriculum or deck-mutation framework merely to reserve this future capability.
The boundary must accept a valid deck without requiring membership in a frozen
training roster. Proof should cover a new same-world deck at a later game root,
an invalid generated deck rejected before execution, fixed-list upfront errors,
and faithful recovery of the actual deck-selection process without weight updates.

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
- **Training run:** the regime, deck selection/production and source provenance,
  sampling policy/curriculum, seeds, metrics and producer history. It references
  the checkpoint artifact by verified identity; the checkpoint need not contain
  or load the run. This preserves provenance without making the model own it.
- **Recovery state:** optimizer, scheduler, collector/RNG and deck-selection
  state owned by the run. Exact-resume checks happen here and in run admission,
  not in portable checkpoint compatibility.

A model may play any deck admitted by its bound world, regardless of which decks
were used in training. Deck construction/content admission remains shared native
domain behavior. Unsupported cards, wrong rules/content and incompatible model
I/O still reject; unseen-but-admitted combinations do not. The distinction is
compatibility, not a claim about strength. Uniform distinct-deck sampling remains
the initial run policy; curriculum is deferred and never becomes model metadata.

### Usage sketch (implementation details are not separate user approvals)

Supply decklists for a training run to sample from. The run
records the decks used, selection policy/state and history. Export or load the
model with its world; the model does not require the training files or regime.
Use any decklists admitted to that same world for play or evaluation, recording
actual evaluation inputs in the evaluation record. Resume a training run using
its recovery state and run contract. Deck changes produced by that process may
belong to the same run; unrecorded input/state substitutions must not silently
pass exact resume. Adaptive curriculum and deck generation remain future work.

### Implementation consequence and proof

Before this pass (c76c814ae), `checkpoint_world` embedded training `setups` and
`deck_collection`, and `validate_agent_setup` used those setups as an allowlist;
checkpoint/catalog code re-admitted `DeckCollection` source records. That
coupling was removed (1b71de345); merely weakening the roster check would not
have been sufficient.

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

Implemented 2026-10-28 per Jack Heart's approval (1b71de345).

### Concept review of the implementation (2026-10-28, headless Graph step)

Deck-supplied runs now match the accepted model: one `Deck`, `TrainingDecks`
is run data (supply plus provenance, checked upfront, re-admitted natively at
every root), checkpoints export `pool_world` from the native pool descriptor,
and execution admits any deck that pool admits. No `CollectionDeck`,
`DeckCollection` or `deck_collection` remains outside frozen evidence.

One remaining path contradicts the accepted decision, which settles it:
**authored-pack checkpoints still bind their training pair.** `checkpoint_world`
binds the complete authored roster only for the two Elemental packs; every other
compiled authored pack binds the actual training match. `mtg122-complexity-v1`
authors six decks over one card file, and native reset admits any pair of them
(checked: `gw_allies` v `l2_heavy`). A checkpoint trained on `l1_swift` v
`l1_heavy` is nevertheless rejected for Allies v Lessons by
`validate_agent_setup` ("checkpoint setup differs from the execution match") —
the training-roster allowlist Jack Heart rejected, and the MTG-122 L1-then-
Allies-v-Lessons case. Two-deck packs are unaffected in practice (pair = pack).

Resolution (implied by the accepted decision, not a new product choice):
default (uncompiled) content has no native deck admission and keeps its exact
pair binding; native admission, not a Python mirror rule, decides mirrors.

Implemented 2026-10-28: `checkpoint_world` binds the complete authored roster
for any compiled authored pack when the actual setups are authored, marked
`setup_binding: authored-pack-v1` (Elemental stays unmarked; its digests are
frozen). `expected_world` reads unmarked non-Elemental compiled bindings with
the historical `_pair_world` reader. `validate_agent_setup` admits any roster
pair; the pack must be explicit unless it is the pack's own distinct pair that
native reset infers. Scenario replay compares via `expected_world`.

Recorded deferral, not reopened here: checkpoint `bc` extra still carries
opaque run/regime producer links (questions.md). They do not affect world
compatibility, and moving them is a training-state change across all producers.

### Concept review at 821824996 (2026-10-28, headless Graph step)

The authored-pack binding now matches the decision. One consumer path still
contradicts it, and the decision settles how: **imported decks still refuse
every checkpoint player.** `SimulationConfig.validate_imported_players`
(`manabot/infra/config.py:240`) and Academy's start path
(`academy/server.py:734`) reject any checkpoint player when the match is imported,
saying "checkpoint setup admission is not available". That admission now exists.
Sim rollouts already call `validate_agent_setup` (`manabot/sim/rollout.py:174`),
and `CheckpointVillain` calls it when given player configs. The run this PR
produces, a pool-bound deck-trained checkpoint, therefore cannot be simulated
on imported decks from its own pool. The decision says to "use any decklists
admitted to that same world for play or evaluation". The earlier deferral called
this consumer scope, but sim is manabot evaluation and the special case is a
second, older rule standing in front of the shared one.

Next implement direction: delete `validate_imported_players` so sim relies on
the shared check. In Academy, replace the blanket rejection with that check only
if every `CheckpointVillain` built for an imported match (start, fork, reconnect,
rematch) receives the imported player configs; otherwise keep Academy's refusal
and reword it to name the Academy limitation, not a missing admission. Neither
case is the deferred 1a redesign. Update `docs/decklist-import.md`
("Imported setups currently reject checkpoint opponents") and questions.md. Proof
uses bounded fixtures and no weight updates. A fixture checkpoint exported with
`save_checkpoint(content_pool="tla-pool-v1")` (untrained weights) plays a sim
game on two imported files. An authored/Elemental checkpoint on the same files
is rejected by `validate_agent_setup` with its world diagnostic. Random/default
imported sim and existing Academy imported/built-in tests stay green.
Working screen: `scratch/concept-review.html`.

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

### Implementation choices, distinct from explicitly accepted direction

In the PR review, Jack Heart asked whether “Files become one admitted training
collection” was agreed. The explicit agreement is random training matches from
decklists, uniform sampling over distinct lists initially, future curriculum,
and world-bound models independent of training regime. A separate preparation
command, immutable manifest, and collection type bundling source records, pool,
sampling policy and learning schema are implementation choices; the heading
must not present them as separately approved product concepts. The question does
not itself approve or reject that lifecycle. The subsequent clarification at
the top of this document now establishes world validation and a single deck
concept, without requiring that lifecycle; the preparation command, manifest
and collection type were removed in the 2026-10-28 implementation.

Current implementation: the regime checks a fixed list upfront against an explicit qualified pool. Reject the entire
list on invalid input, retaining source/line diagnostics; never silently
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
  native construction admission, run-owned training decks and consumer entry points.
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
checkpoint opponents remain rejected in those consumers (scratch/questions.md).

Training (world-validated decks, implemented): `decklist.Deck` is the one deck
concept (counts; identity from content). `AdmittedDeck` is an import record
exposing `.deck`. `training/decks.TrainingDecks` is run-owned data: pool key,
distinct decks, every source record; `from_documents`/`from_files` reject the
whole fixed list upfront; `check()` re-admits decks natively at regime
validation. Regime field `decks: {pool, files}` embeds at load. Native
`VectorEnv.set_training_decks` admits every deck before replacing the supply for
later roots (no pre-reset restriction; rejection keeps prior decks; running
games unchanged); `current_player_configs` reports actual setups, used by the
collector and failure capture. `current_matchup_indices` is Elemental-only.

Checkpoints: deck runs export `save_checkpoint(content_pool=...)` →
`model.world.pool_world`: native `content_pool` descriptor, rules, input schema
and learning-schema hash; no setups, decks, sources or sampler. Execution admits
any deck the pool admits (`validate_agent_setup`), rejects unsupported/other
pools; load/catalog reject changed descriptor/rules/input. Exact setup bindings
bind a compiled authored pack's complete roster (versioned
`authored-pack-v1`; Elemental unmarked and frozen); historical unmarked pairs and
default content keep exact pairs. Run recovery keeps the decks digest in collector
snapshots and `GameAttempt.decks_digest`; learning-state resume compares
`regime.decks`; policy admission no longer compares training decks.

Acceptance (training): three or more eligible decks; changing pairs over resets;
identical pair/seat/seed/policy RNG after recovery; invalid/unsupported fixed
input rejected upfront; changed supply rejected on resume; actual setups in
failure/capture records; unseen same-world deck at a later root; invalid
generated deck rejected before play; world-only export/load/inspect. Bounded
fixtures, actual collector path, no weight updates.

Remaining: gate owns full affected suites (debug Rust, training recovery,
learning-state, Elemental and combined CI collection) per
docs/development/validation.md. Run→checkpoint linkage is existing TrainingRun
records; not re-proved end-to-end by executing a regime here.

## Delete — do not maintain

Completed earlier: explicit-pack authored-only selection; `env.match.parse_deck`
and its tests; deck hashing moved to decklist (identity preserved). This pass
removed `manabot/deck_collection.py` (`CollectionDeck`, `DeckCollection`,
prepare CLI and manifest), checkpoint/catalog `deck_collection` binding and
roster allowlist, collector checkpoint-digest check, policy-admission deck
equality, native `set_deck_collection` pre-reset restriction, and
tests/training/test_deck_collection.py (coverage moved to test_training_decks.py).
Setup hashing stays local (infra.identity import breaks Academy's boundary).
The Academy redesign below is explicitly deferred.

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

Check — 2026-10-28 at 63c9f6bf5: `uv run --extra dev pytest tests/model/test_world.py tests/model/test_catalog.py tests/training/test_training_decks.py tests/training/test_matchups.py tests/eval/test_mirror_admission.py -q`: 39 passed (covers the four approval proofs and the authored-pack proofs). Earlier wider run: Elemental program count 8 vs 7 (test_elemental_world.py:151) and belief runtime `KeyError: 'relations'` (test_runtime.py) fail identically at c42bf4934 without this work; gate owns them with full suites and debug Rust.
