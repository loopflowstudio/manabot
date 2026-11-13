# Semantic content

Every current card is one Rust function in the file of the set that first
printed it, under [`managym/cards/src/`](../../managym/cards/src/)
(`tla.rs`, `stx.rs`, `lea.rs`, …), returning the engine's own
[`CardDefinition`](../../managym/state/src/card.rs) written with its rules types
(`Ability`, `Effect`, `TargetSpec`, `TriggerCondition`, …). That value is the
card: the engine runs it, the pool describes it and the policy reads it. There
is no second card vocabulary, compiler or IR for current content. Each
definition carries its key (`<set>.<card>`) and the Scryfall id of that
printing; Oracle text, rulings and images are reachable from the id, while the
printed `text_box` stays in the definition so play works offline.

Cards and tokens share one `PrintedCharacteristics` (name, cost, types,
colors, keywords, abilities, text, power/toughness); a `CardDefinition` adds a
card's key and required Scryfall id. Tokens are not cards. Each is a
`TokenDefinition` in
[`managym/cards/src/tokens.rs`](../../managym/cards/src/tokens.rs): a `token.*`
key, an optional Scryfall id and its `PrintedCharacteristics`. Effects create a token by key
(`Effect::create_tokens("token.spirit", 1)`), so two tokens may share a printed
name. A token's Scryfall id is its first token printing's, and is absent only
for a token that was never printed.

New behavior is a new rules-type variant (for example an `Effect`) with a
directed card test in `managym/cards/tests/cards/<set>.rs`, never card-specific logic.
Edit a card's function, list it in its set's `cards()`, and rebuild the
extension. `current/setups.json` is ordinary deck/sideboard data naming cards by
key.

**Support is definition.** The world supports a card for deck entry when its set
file defines it as a non-token card. Tokens, rules fixtures and unknown cards
are rejected. Known rules bugs are tracked as Tasks without revoking
playability; playable does not mean bug-free. Rules tests are ordinary tests;
they never gate support. See the [TLA notes](https://github.com/loopflowstudio/manabot/blob/cac6fa27d/docs/rules/tla/README.md) for
the open TLA work.

## The catalog and identity

A world has one card list. The catalog registers every set-file definition,
every token definition and the rules fixtures in `managym/src/cards/fixtures.rs`
(Crossroads of Destiny), in name order, each double-faced card's back face right after its front; every
match runs on it. Its identity is its `content_digest`, which is the world
identity's card digest. Authored setups are decklists over it
([`cards/setups.rs`](../../managym/src/cards/setups.rs)), named by preset or
by the `pack_key` frozen evidence records; they are not packs.
Each row is an `ObjectDefinition`: identity, printed characteristics and token
status. Token rows have native IDs and learning rows so created tokens can be
observed, but are never eligible for deck entry.
`managym.content_pool("world")` exposes the world pool's native IDs, keys and
eligibility without starting a game; `managym.set_cards(["tla", "lea"])` narrows
it to chosen sets.

## Learning projection

Current execution uses world **w17** and `current/learning_schema.json` (schema
**10**). `managym.catalog_tokens()` walks every catalog definition
with a `serde` serializer ([`cards/tokens.rs`](../../managym/src/cards/tokens.rs)):
struct and variant boundaries, field names, `Effect::…` variants, integers,
booleans and printed categories become tokens, so a new rules variant reaches
the policy without a vocabulary edit. Each definition has a characteristics span
and one program span per ability or spell; a created token is a reference to its
definition row. `key`, `scryfall_id`, `name` and `text_box` are never tokens.

`BoundSemanticPack` binds that catalog to a live manifest: the digests must match
and each manifest ID joins the catalog row of the same definition key, so rows
never depend on `CardDefId` allocation. A back face or rules fixture has no key
and joins by its printed name. Names alone do not identify a definition: two
tokens may print one name, and a token may print a card's.

```python
raw = managym.catalog_tokens()
raw["keys"][row]            # "token.spirit", "tla.aang_s_iceberg", or None
raw["registry_names"][row]  # "Spirit": the printed name, possibly shared
```

The `semantic_only` mode masks opaque `CardDefId` inputs;
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

## Earlier worlds

Earlier worlds' sources, IR, learning schemas, coverage evidence and the Python
compiler that produced them (`v1/`, `w7/`–`w11/`) are not in the tree; they run
and read from the git revision that recorded them (see
[WORLDS.md](../../WORLDS.md#running-an-older-world)). Earlier documentation of
each pack's rollout is at
`c5d406e3d2a701525c15c8500a30f7da7896cc40:content/semantic/README.md`.
