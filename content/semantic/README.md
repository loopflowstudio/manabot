# Semantic content

Every current card is one Rust function in the file of the set that first
printed it, under [`managym/src/cardsets/`](../../managym/src/cardsets/)
(`tla.rs`, `stx.rs`, `lea.rs`, …), returning the engine's own
[`CardDefinition`](../../managym/src/state/card.rs) written with its rules types
(`Ability`, `Effect`, `TargetSpec`, `TriggerCondition`, …). That value is the
card: the engine runs it, the pool describes it and the policy reads it. There
is no second card vocabulary, compiler or IR for current content. Each
definition carries its key (`<set>.<card>`, tokens `token.*`) and the Scryfall
id of that printing; Oracle text, rulings and images are reachable from the id,
while the printed `text_box` stays in the definition so play works offline.

New behavior is a new rules-type variant (for example an `Effect`) with a
directed card test in `managym/tests/cards/<set>.rs`, never card-specific logic.
Edit a card's function, list it in its set's `cards()`, and rebuild the
extension. `current/setups.json` is ordinary deck/sideboard data naming cards by
key.

**Support is definition.** The world supports a card for deck entry when its set
file defines it, unless that set file marks it incomplete (`tla::INCOMPLETE`, one
reason per card). Incomplete cards still run in the fixed setups that name them.
Tokens are never deck entries. Rules tests are ordinary tests; they never gate
support. See the [TLA notes](../../docs/rules/tla/README.md) for the open TLA work.

## Packs and identity

The default content pack registers every set-file definition plus the rules
fixtures in `managym/src/cards/fixtures.rs` (Crossroads of Destiny), keeping the
legacy `CardDefId` prefix. Each authored setup is a pack of exactly its decks'
cards and the tokens they create, in key order
([`cards/setups.rs`](../../managym/src/cards/setups.rs)); exact authored
decklists, or an explicit `content_pack`, select it. A pack's identity is its
`content_digest`; its manifest also names its `pack_key` (absent for the default
pack). `managym.content_pool(pack_key)` exposes native IDs, keys and eligibility
without starting a game; `managym.set_cards(["tla", "lea"])` narrows the world
pool (`world`, every defined card) to chosen sets.

## Learning projection

Current execution uses world **w18** and `current/learning_schema.json` (schema
**9**). `managym.learning_catalog(pack_key)` walks every definition of a pack
with a `serde` serializer ([`cards/tokens.rs`](../../managym/src/cards/tokens.rs)):
struct and variant boundaries, field names, `Effect::…` variants, integers,
booleans and printed categories become tokens, so a new rules variant reaches
the policy without a vocabulary edit. Each definition has a characteristics span
and one program span per ability or spell; a created token is a reference to its
definition row. `key`, `scryfall_id`, `name` and `text_box` are never tokens.

`BoundSemanticPack` binds that catalog to a live manifest: the digests must match
and manifest IDs join rows by registry name, so rows never depend on `CardDefId`
allocation. The `semantic_only` mode masks opaque `CardDefId` inputs;
`identity` exposes them for controlled ablations. The projection reads only
viewer-safe `managym.Observation`; mutable counters, damage, control and other
match facts remain in observation tensors.

Datasets and checkpoints persist `BoundSemanticPack.artifact_header()`. Loaders
reject mismatched schema, content, layout and identity-mode provenance. Older
worlds' artifacts run from the git revision that produced them.

```bash
uv run --extra dev pytest -q \
  tests/semantic/test_current_content.py \
  tests/semantic/test_learning_projection.py \
  tests/semantic/test_learning_projection_env.py \
  tests/model/test_policy_catalog.py
```

## Historical sources and coverage evidence

The `v1/` and `w7/` through `w11/` artifacts retain earlier sources, schemas,
fixtures, IR and receipts. They are not inputs to current content and are never
rewritten. The retired Python compiler (`manabot/semantic/compiler.py`,
`scripts/compile_semantic_content.py`) only reproduces those frozen artifacts;
the explicit source/output invocations in `.github/workflows/ci.yml` validate
them.

`v1/coverage.evidence.json` records rule-family ownership, executable witnesses,
support/deviation annotations and admission history. Its coverage-gap artifact
retains the original rolling policy: more than 20% kernel-changing cards among
the latest 20 distinct nonbasic, nontoken admissions requires an acknowledged
response through the newest history sequence. It records the initial two-deck
expansion as a `0 content-only / 20 total` breach with the semantic-kernel redesign
as its response; it has no bearing on current support.

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
