# MTG-243 — catalog naming (2026-11-04)

Jack Heart authorized one reviewable PR, no merge or training experiments.
The Task directive is the design seed; the inherited T7 note is historical.

## Design

Replace per-resolution `BTreeSet<String>` and `Vec<Vec<Effect>>` construction
with the catalog's immutable, sorted name candidates and one `Decision::NameCard`.
Only the selected name constructs `ExileNamed`; private search and its permission
lifetime stay at that existing continuation authority. Keep the published modal
Command ordering and labels, so humans, LLMs, replay and search choose the same
complete name in one step. Candidates are vocabulary entries, never fake objects.

Use explicit policy input v8 with a public name row and optional candidate-definition join per action and
catalog-derived action capacity. This is linear scoring of the complete support;
it avoids sequential rejection or extra game decisions. Candidate identity and
its existing learning definition both reach scoring. Encode candidate programs
only on naming steps. Existing v1–v7 widths and omitted checkpoint defaults stay
frozen; fresh default remains v6 pending explicit adoption. No truncation or
random fallback. The action axis is the larger of configured capacity and vocabulary size; scoring
trims unused rows so ordinary decisions do not run the larger action network.

Current naming universe is exactly the existing non-token real catalog,
including back faces. Names outside supported definitions remain an unresolved
rules coverage question: this work does not claim all legal Magic names or
expand deck admission. `NameVocabulary` can hold validated names without executable definitions; its
digest enters state identity, and v8 checkpoints bind its ordered names and
optional definition joins. Only the test fixture extends it today.

## Delete — do not maintain

- `NameAndExileFromTargetPlayer`'s per-resolution name set and owned modal
  effect branches; preserve sorted labels, every name and search behavior.
- The corresponding `Effect::ExileNamed` modal-label special case; name labels
  belong to the naming decision. Keep `ExileNamed` as the selected continuation.
- The extra naming-only program encoding in `ActorCritic`: name and object
  references use one table, carried by `EmbeddedObservation`. Slice action
  tensors directly; copying a prepared observation loses its inference scope.
- The temporary deduplication map in `ContentPack::name_candidates`:
  registration already rejects duplicate names; `NameVocabulary` sorts once.
- The second action traversal for name joins in both encoders: action features,
  candidate joins and focus now share one loop. Native capacity validation owns
  rejection; the subsequent action loop needs no truncation.
- The vector naming test's duplicate navigation: replay one recorded Command
  trace through scalar and vector step APIs. Both must retain per-step events;
  a cloned observation snapshot has already drained them and is not equivalent.

These cuts are complete. No exclusive fixtures or deletion targets remain.

## Implementation and remaining gate work

Implemented shared candidates/decision; v8 native scalar/vector/Python parity;
complete name identity and learning-definition scoring; typed immutable
`NameCandidate` exports; catalog-derived capacity;
LLM/Academy stage presentation and reconnect; explicit checkpoint vocabulary
binding. The controlled fixture combines the installed universe with 600
Scryfall-identified names (928 distinct choices), without adding definitions.
Native proofs cover rows 511/512, back faces, legality, exact forks, replay,
stale rejection and fail-to-find. The policy proof gives every candidate positive
support, distinguishes identities and improves the selected log probability in
one bounded supervised gradient step. This is not a strength experiment.

Semantic Observation is schema 8, capture 5, state hash 16: public input joins
and a shared vocabulary digest replace owned effect branches. Current w17
conformance, authority, both parity receipts and Academy fixtures were refreshed;
the authority's 135 Commands are byte-for-byte unchanged. Frozen evidence was
untouched; the RUL-9 dependency was read from its artifact cache. Old semantic
world bindings fail explicitly; historical tensor shapes/omissions remain fixed.

Check: `uv run --extra dev pytest tests/model/test_catalog_naming.py tests/env/test_observation.py -q` — 60 passed; `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 --lib <filter>` with `naming_tests` and `agent::observation_encoder::tests` — 8 passed. Release rebuild/atomic extension install, Ruff, rustfmt and diff checks passed. Prior model/client results remain applicable (`1e7dc47d4:scratch/mtg-243-naming.md`). Gate owns full affected suites, current-world admission/monitoring, checkpoint compatibility, final parity source fingerprints and same-runner Engine Budget, once.

Pre-reduction debug allocation attribution (not a speed threshold): catalog setup
5,232 allocations / 1,863,512 bytes; naming decision 0 / 0; 928-action vector
1 / 37,120; observation projection 1,880 / 154,583; encode into existing buffers
5 / 792; naming fork 15 / 47,012; ordinary fork 15 / 10,404; ordinary fork+step
393 / 41,980. One current-catalog CPU semantic forward (344 candidates, hidden
width 8) took 19.5 ms before sharing its program table. Setup, projection, encoding, scoring and ordinary work are
reported separately; no cross-runner performance claim. Reproduce native
attribution with the focused `catalog_naming_cost_attribution` test and the
`benchmark` feature, one test thread. Hash-contract changes require an explained
new CI digest, not a claim that ordinary rules traces changed.

Publication belongs to the Graph's delivery step; interactive review remains the
boundary. No merge, auto-merge or training experiment was performed.
