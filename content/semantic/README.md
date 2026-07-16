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

`v1/coverage.evidence.json` adds the reviewed facts that typed card programs
cannot derive themselves: rule-family ownership, exact executable Rust tests,
support/deviation annotations, and a sequenced history classifying real card
admissions as `content_only` or `kernel_changing`. Generate or verify the
canonical coverage-gap artifact with:

```bash
uv run scripts/generate_coverage_gaps.py
uv run scripts/generate_coverage_gaps.py --check
```

The rolling policy selects the latest 20 distinct real card admissions (basic
lands and referenced tokens are excluded) and uses those 20 rows as the
denominator. More than 20% kernel-changing cards is a breach. A breach remains
visible in `v1/generated/coverage-gaps.json` and passes the check only when an
IR redesign, kernel redesign, or stopped-expansion response exists and is
acknowledged through the newest history sequence. Adding another card makes a
previous acknowledgment stale, so the response cannot become a permanent
waiver.

The initial history intentionally reports the two-deck Milestone 1 expansion
as an acknowledged `0 content-only / 20 total` breach. The linked semantic
kernel design is the reviewed IR-redesign response to that historical pattern;
future admissions should move the rolling window toward content-only changes.

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

## Learning projection v1

`v1/learning_schema.json` defines the versioned, typed token graph used by
learning consumers. `manabot.semantic.learning.BoundSemanticPack` obtains a
read-only manifest from the exact environment `ContentPack`, resolves the IR's
legacy registry-name adapters once, then discards names from model inputs.

The shared catalog stores definition and program token spans plus explicit
definition-reference edges. Viewer observations carry only ragged bindings into
that immutable catalog. The `semantic_only` mode keeps the structural binding
but masks every opaque `CardDefId`; the `identity` mode makes that feature
explicit for controlled ablations. Neither mode tokenizes card names.

Semantic datasets and checkpoints must persist the artifact header returned by
`BoundSemanticPack.artifact_header()`. Semantic loaders reject missing or
mismatched learning-schema, IR, content-pack, layout, and identity-mode
provenance. Legacy artifacts without semantic inputs retain their existing
load path and are not silently upgraded.

The projection reads only `managym.Observation`, which already excludes the
opponent's hand, unrevealed libraries, and private choices. It never reads
`Game` or `MatchState`; mutable counters, damage, control, and other match facts
remain in the existing observation tensors.

Focused verification:

```bash
uv run --extra dev pytest -q \
  tests/semantic/test_learning_projection.py \
  tests/semantic/test_learning_projection_env.py
```
