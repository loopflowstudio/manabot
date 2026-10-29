# Semantic content

Every current card is a typed Rust function in its printed set's file under
[`managym/src/cardsets/`](../../managym/src/cardsets/) (`tla.rs`, `stx.rs`,
`lea.rs`, …), defined once and keyed by set (`tla.forest`, `stx.pop_quiz`;
tokens made outside TLA are `token.*` in `tokens.rs`).
[`cards/authoring.rs`](../../managym/src/cards/authoring.rs) defines symbolic
programs, costs, targets, predicates and characteristics. Card text is
presentation only.

The engine compiles those definitions in process
([`semantic/compile.rs`](../../managym/src/semantic/compile.rs)): it resolves
definition references and token closure, checks target roles, assigns opcodes and
program indexes, and hashes canonical IR. No card catalog or IR is generated on
disk. Python reads the same documents through `managym.semantic_ir_json(pack_key)`.

Edit a card's Rust function, add it to its set's `cards()` list, and rebuild.
TLA membership and fixed tokens derive from that set list. Do not add another
native registration. `cargo test --lib semantic::compile` compiles every current
preset and checks qualifications. `current/setups.json` remains ordinary
deck/sideboard data, and `qualifications.json` remains independent review evidence.
Changes to rules or token dependencies require renewed qualification, even if the
Rust source compiles.

The [TLA authoring and qualification contract](../../docs/rules/tla/README.md)
describes the inventory, independent witnesses, unsupported cards and ownership
boundaries. A definition's availability in an authored setup does not qualify it
for arbitrary deck entry. Qualifications bind reviewed witnesses to the complete
transitive definition closure; editing a clause invalidates that qualification.

## Native runtime and identity

`SemanticPack::compile_content_pack` lowers compiled IR into the ordinary native
`CardDefinition` runtime. It supplies characteristics, costs, targets, triggers,
static abilities and effects. Presentation text comes from the same definition;
Oracle text never supplies executable rules. Historical IR without embedded text
runs from the git revision that produced it.

Each compiled `semantic_index` is local to its IR document, distinct from native
`CardDefId`. The compiled pack records its pack key, IR hash and source hash in
`ContentPackManifest.compiled_semantics`. `managym.content_pool(pack_key)` exposes
that identity, native IDs, semantic keys and eligibility without starting a game.
Policy binding validates the native receipt before those IDs reach an encoder.

Exact authored setups select their compiled runtime through the shared native
catalog. Explicit selection also supports repeated-deck matches. The default
content pack registers every typed definition through the same lowering, plus
the few native cards that still lack one (`managym/src/cards/native.rs`). Independent directed rules tests witness behavior;
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
provenance. Current manifest binding uses explicit pack keys in `CURRENT_PACK_KEYS`.
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
fixtures and receipts. They are not inputs to current content. Historical
compiler checks remain available through `scripts/compile_semantic_content.py`
(the retained Python compiler); the explicit source/output invocations in
`.github/workflows/ci.yml` validate those frozen artifacts.

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
