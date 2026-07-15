# Curated semantic content

`v1/two_deck.source.json` is the reviewed authoring input for the exact
UR Lessons versus GW Allies product matchup. It describes card characteristics,
typed ability programs, exact deck admission, and token closure without parsing
Oracle text at runtime.

The compiler validates every object and instruction against a closed schema,
resolves semantic definition references, lowers named operations to stable
numeric opcodes, and writes canonical JSON plus a coverage report:

```bash
uv run scripts/compile_semantic_content.py
uv run scripts/compile_semantic_content.py --check
```

The generated `semantic_index` is local to this IR document. It is deliberately
not `CardDefId`. At pack load, `content_pack_binding` resolves once through the
current `ContentPack` adapter and every executable program thereafter carries
typed indices and opcodes rather than card names. The Rust conformance test
checks that every binding resolves to a real `CardDefId`; W2-179 remains the
owner of `ContentPack`, definition storage, and match-state layout.

`v1/two_deck.fixtures.json` contains deterministic lowering expectations for
representative branch, linked-exile, multi-target, and trigger programs. These
are compiler fixtures, not claims of end-to-end rules conformance. Runtime
interpretation will require differential traces against the existing effect
path before it can replace it.

One source fact is intentionally conspicuous: the checked-in UR product deck
currently contains 41 cards while the milestone prose calls it a 40-card deck.
The compiler records the actual product manifest rather than silently deleting
a card. GW Allies contains 40 cards. Resolving that content decision belongs to
the curated deck owner, and changing it will make the product-manifest parity
test fail until both sides agree.
