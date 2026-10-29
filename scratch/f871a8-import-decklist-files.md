# PR D: decklist import and reproducible training on decklists

Full wording of the accepted corrections and usage sketches below:
e3b4f0583:scratch/f871a8-import-decklist-files.md.

## Accepted corrections (Jack Heart, 2026-10-28; implemented 1b71de345 ff.)

**One deck concept.** A deck is main/sideboard counts with identity from content,
whatever its source or selection. File text, filenames and diagnostics describe
an import, not a kind of deck. Training receives a verified world and submits
decks at game creation; a fixed list is checked upfront, and future computed or
modified decks pass the same native world validation before use. Pre-registering
every training deck must not define compatibility. `DeckCollection` is never a
mandatory admission lifecycle or the owner of the sampler/model-input schema.
Proof: an unseen same-world deck at a later root, an invalid generated deck
rejected before execution, upfront fixed-list errors, and faithful recovery of
the actual selection process without weight updates. No curriculum or
deck-mutation framework; the retry policy for adaptive producers is future work.

**Checkpoints bind worlds, not training regimes.** A world is fixed rules-engine
behavior plus the card file (actual identities, not names or shapes). A portable
checkpoint holds weights, architecture/inference configuration, observation/
action/output contracts and the world binding; it needs no decks, sources,
sampler or regime to save, load, inspect or execute. This supersedes the 3a/3b
review. The training run owns the regime, deck provenance, sampling policy,
seeds, metrics and recovery state, and references the checkpoint by verified
identity. Exact-resume checks live in the run, not in checkpoint compatibility.
A model may play any deck its world admits; unsupported cards, wrong rules/
content and incompatible I/O still reject. Frozen checkpoints, experiments and
old schema semantics stay unchanged; new portable artifacts are versioned.

**Scope.** Ship one cohesive PR for review and stop before merge. Training
matchup generation is primary; Academy is a secondary consumer. Uniform sampling
over distinct normalized decklists (2a); curriculum deferred. 17Lands becomes
the real-world source after full TLA support; its acquisition, schema mapping
and held-out policy are later design work. No training campaign, paid compute,
bulk download, frozen-evidence rewrite, follow-up launch, independent TLA rules
work, or change to MTG-175's reserved curriculum experiments.

Implementation choices beyond these (not separately approved): reject the whole
fixed list on invalid input with source/line diagnostics, never silently
filtering; collapse identical normalized lists while retaining every source
record; one whole-pool policy vocabulary; game starts recorded separately from
learner transitions. `uniform-distinct-v1` draws two different deck IDs with
unbiased seats from the native reset seed. Excluding mirrors is an assumption
(scratch/questions.md); at least two lists are required. Settled follow-up
reviews (detail at 12a2b9fc4) are reflected in "Implemented path". Deferred:
Academy 1a redesign; checkpoint `bc` producer links.

## Contract and ownership

- Content Task 892c47 owns canonical card authoring, support, compiled
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

Content base: PR #394 (Task 892c47) at 86ebe9ccf, merged here on 2026-10-28
after the earlier be2b7586b merges (541365b28, 2f7f7c09c), replacing the
1e0d3c998 copy. 2f7f7c09c carried only be2b7586b; the first-printing card keys
and qualification-gate removal arrived with 86ebe9ccf. #394 supplies typed Rust
cards, one world (w16), first-printing keys (`lea.forest`) and support defined by
a set file's definition; defined-but-incomplete TLA cards are listed in
`tla::INCOMPLETE` and are unsupported, and undefined cards are unknown, not
unsupported pool rows. `content_pool` keeps its shape. Importer code relies on no
hard-coded keys or qualification data; this branch's unsupported-card fixtures
use Accumulate Wisdom. Generated content and world-bound evidence are #394's.

Training overlap reviewed at PR383 d0f48ca96 and PR389 c85470a59; scoped coordination
is retained in Task 1d7884 comment 2c523295-b690-4e45-a989-1dd37b91cabb. Adaptive
reassignment, experimental distributions and shared simulation APIs are untouched.

## Implemented path and remaining proof

The importer connects parser/resolution to native admission, sim file flags and
Academy's editable two-file validation/start flow. Retained imports bind exact
pool/content identity and reject source/setup tampering. Checkpoint players on
imported setups go through the shared `validate_agent_setup`: sim admits every
model player in `simulate_models` before games; Academy's only CheckpointVillain
construction receives `config.to_rust()` (imported configs). Pool-bound
checkpoints play imported decks; authored/Elemental ones refuse them.

Training (world-validated decks, implemented): `decklist.Deck` is the one deck
concept (counts; identity from content). `DeckImport` is an import record:
source `document` plus its admitted `deck`. `training/decks.TrainingDecks` is run-owned data: pool key,
distinct decks, every source record; `from_documents`/`from_files` reject the
whole fixed list upfront; `check()` re-admits decks natively at regime
validation. Regime field `decks: {pool, files}` embeds at load and is the
regime's only setup record (an authored `match` beside it is rejected). Native
`VectorEnv.set_training_decks` admits every deck before replacing the supply for
later roots (no pre-reset restriction; rejection keeps prior decks; running
games unchanged); `current_player_configs` reports actual setups, used by the
collector and failure capture. `current_matchup_indices` is Elemental-only.

Checkpoints: deck runs export `save_checkpoint(content_pool=...)` →
`model.world.pool_world`: native `content_pool` descriptor, rules, input schema
and learning-schema hash; no setups, decks, sources or sampler. Execution admits
any deck the pool admits (`validate_agent_decks`, shared by
`validate_agent_setup` and producer admission), rejects unsupported/other
pools; load/catalog reject changed descriptor/rules/input. Exact setup bindings
bind a compiled authored pack's complete roster (versioned
`authored-pack-v1`; Elemental unmarked and frozen); historical unmarked pairs and
default content keep exact pairs. Run recovery keeps the decks digest in collector
snapshots and `GameAttempt.decks_digest`; learning-state resume compares
`regime.decks`; producer admission requires a bound checkpoint and applies
`validate_agent_decks` to every deck, never a representative pair.

Acceptance (training): three or more eligible decks; changing pairs over resets;
identical pair/seat/seed/policy RNG after recovery; invalid/unsupported fixed
input rejected upfront; changed supply rejected on resume; actual setups in
failure/capture records; unseen same-world deck at a later root; invalid
generated deck rejected before play; world-only export/load/inspect. Bounded
fixtures, actual collector path, no weight updates.

State (2026-10-28): PR #395 head 917037b6e is pushed and stacked on #394
(base `jack-heart/892c47-author-tla-cards`, 86ebe9ccf an ancestor). The three
Avatar Aang face tests that failed after the 86ebe9ccf merge were replaced by the
two-face pool fixture (e3b4f0583). 917037b6e makes failure capture read actual
native setups whenever roots are sampled and hashes the collector's deck supply
once.

Remaining: world-bound evidence belongs to #394 and is unregenerated at
86ebe9ccf. Check at afdb82447+ (debug `cargo test --no-fail-fast`, the CLAUDE.md
world set + tests/semantic + training-deck + Academy decklist): only
`semantic_conformance_tests` (w15 manifest "different world", ur-vs-gw-5eed
digest drift; identical in #394's Rust CI) and four learning-schema-version
asserts (8 vs 3/4/7: test_elemental, test_elemental_world, test_optional_payments,
test_sacrifice) fail. This branch's diff touches neither. Not regenerated here.
Check at 917037b6e: decklist/training-deck/Academy decklist suites 27 passed; recovery/game-error/capture suites 83 passed.
#394 removed every double-faced card, so face resolution is tested on a pool
fixture with one row renamed to two faces. Run→checkpoint linkage is the existing
TrainingRun records, not re-proved by executing a regime here.

Concept review (2026-10-28, at 29f8823c8): aligned; nothing to change. One
`Deck`; `TrainingDecks` is run data checked upfront and re-admitted natively;
pool checkpoints carry only the content-pool descriptor, rules and input; producer
admission checks every deck. `root_match` on a deck run only opens the collector
and fingerprints (native reset samples from the submitted supply; frozen
opponents are excluded for deck runs). The `bc` producer links stay deferred.

## Delete — do not maintain

Completed earlier: explicit-pack authored-only selection; `env.match.parse_deck`
and its tests; deck hashing moved to decklist (identity preserved). Later passes
removed `SimulationConfig.validate_imported_players` and Academy's imported
checkpoint refusal, plus `manabot/deck_collection.py` (`CollectionDeck`, `DeckCollection`,
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

Compress (2026-10-28): native `deck_admission::development_rejection` is the one
development-v1 player-deck rejection, shared by game creation and
`set_training_decks`; `validate_regime` owns the pool-vocabulary check.

Scratch remains ~2.2k tokens over its 12k target: #394's tracked notes and
`scratch/reference` (~10k tokens) arrive with its branch, and deleting them here
would appear as deletions in #395. They are left to their owner and clear when
#394 lands; this Task's note is ~4k tokens.
