# Semantic content

Current card authoring lives in `current/definitions.json`: typed characteristics,
programs and presentation text shared by native play and learning. `current/setups.json`
selects deck and sideboard entries; it does not redefine cards. The compiler resolves
definition references and token closure, validates a closed instruction schema, and
emits canonical IR plus coverage reports:

```bash
uv run python -m scripts.compile_current_content
uv run python -m scripts.compile_current_content --check
```

The [TLA authoring and qualification contract](../../docs/rules/tla/README.md)
describes the inventory, independent witnesses, unsupported cards and ownership
boundaries. A definition's availability in an authored setup does not qualify it
for arbitrary deck entry. Qualifications bind reviewed witnesses to the complete
transitive definition closure; editing a clause invalidates that qualification.

## Native runtime and identity

`SemanticPack::compile_content_pack` lowers checked IR into the ordinary native
`CardDefinition` runtime. It supplies characteristics, costs, targets, triggers,
static abilities and effects. Presentation text comes from the same definition;
Oracle text never supplies executable rules. Historical IR without embedded text
uses `current/presentation.json`, a frozen text snapshot.

Each generated `semantic_index` is local to its IR document, distinct from native
`CardDefId`. The compiled pack records its pack key, IR hash and source hash in
`ContentPackManifest.compiled_semantics`. `managym.content_pool(pack_key)` exposes
that identity, native IDs, semantic keys and eligibility without starting a game.
Policy binding validates the native receipt before those IDs reach an encoder.

Exact authored setups select their compiled runtime through the shared native
catalog. Explicit selection also supports repeated-deck matches. The generic
native registry lowers its TLA cards from the same current definitions and retains
non-TLA registrations. Independent directed rules tests witness behavior;
comparisons between two projections of the same authoring source are not rules
evidence.

Current setup projections preserve historical counts, including the 41-card UR
Lessons deck and 40-card GW Allies deck. The older milestone's 40-card description
does not authorize deleting a card. Product deck changes remain the curated deck
owner's decision.

## Learning projection

Current execution uses world **w16** and `current/learning_schema.json` (schema **8**).
`BoundSemanticPack` binds a native manifest to the exact IR and schema, resolving
registry names once and discarding them from model inputs. The shared catalog
stores typed definition/program token spans and definition-reference edges.
Viewer observations carry ragged bindings into that immutable catalog.

The `semantic_only` mode masks opaque `CardDefId` inputs; `identity` exposes them
for controlled ablations. Neither mode tokenizes card names. The projection reads
only viewer-safe `managym.Observation`, never `Game` or `MatchState`; mutable
counters, damage, control and other match facts remain in observation tensors.

Datasets and checkpoints persist `BoundSemanticPack.artifact_header()`. Loaders
reject missing or mismatched schema, IR, content, layout and identity-mode
provenance. Current manifest binding uses explicit entries in `CURRENT_IR_PATHS`.
Historical artifact inspection requires its original schema and IR paths; frozen
receipts and checkpoints are never silently upgraded.

```bash
uv run --extra dev pytest -q \
  tests/semantic/test_current_content.py \
  tests/semantic/test_learning_projection.py \
  tests/semantic/test_learning_projection_env.py \
  tests/model/test_policy_catalog.py
```

## Historical sources and coverage evidence

The `v1/` and `w7/` through `w11/` artifacts retain earlier sources, schemas,
fixtures and receipts. They are not inputs to current generation. Historical
compiler checks remain available through `scripts/compile_semantic_content.py`;
the explicit source/output invocations in `.github/workflows/ci.yml` validate
those frozen artifacts independently of the current generator.

`v1/coverage.evidence.json` records rule-family ownership, executable witnesses,
support/deviation annotations and admission history. Its coverage-gap artifact
retains the original rolling policy: more than 20% kernel-changing cards among
the latest 20 distinct nonbasic, nontoken admissions requires an acknowledged
response through the newest history sequence. It records the initial two-deck
expansion as a `0 content-only / 20 total` breach with the semantic-kernel redesign
as its response; it is not the current pool's eligibility authority.

```bash
uv run scripts/generate_coverage_gaps.py --check
```

Historical Elemental operations and fixtures remain useful references for modal
choices, target clauses, private selection and dynamic quantities. A spell's
`target_binding: "inferred"` leaves target selection to shared effect traversal;
omitting it retains the explicit binding contract. The
[sacrifice and landfall rule contracts](../../docs/rules/elemental-sacrifice.md)
retain their behavioral coverage and release boundaries. Earlier documentation
of each pack's original rollout is available at
`c5d406e3d2a701525c15c8500a30f7da7896cc40:content/semantic/README.md`.
