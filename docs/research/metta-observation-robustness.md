# Metta observation-space robustness: history and lessons for semantic programs

Research date: 2026-07-15. Repository studied:
[`Metta-AI/metta-public`](https://github.com/Metta-AI/metta-public), including
its full Git history through the archived head.

## Conclusion

Metta is a strong precedent for making token observations operationally
robust. Its history shows that variable-length tokens are only the first step:
the durable work is explicit schema identity, padding, normalization, overflow
semantics, checkpoint binding, compatibility, and cross-language parity.

Metta does not demonstrate the stronger capability Etude wants. Its policy
can survive a known feature arriving under a different runtime ID, but it does
not infer the meaning of a genuinely new feature. Feature names bind known
facts to learned embedding slots; they are not themselves interpreted as a
semantic language. Etude's target is therefore one level higher: an unseen
card composed from known typed operations should remain intelligible without
its `CardDefId`.

The design maxim is:

> Runtime numeric IDs are transport. Versioned symbolic semantics are
> checkpoint meaning.

## Historical progression

### 1. Fixed dense channels

Metta began with box/grid observations consumed by a CNN. A fixed set of
channels was simple and fast, but observation additions or reorderings changed
model inputs. Environment variation and checkpoint portability were coupled to
one dense channel layout.

### 2. Sparse observation tokens

The May 2025
[`Robust Agent - Observation Encoder`](https://github.com/Metta-AI/metta-public/commit/af05d6ff5a05b35c29f088244198ac791dc8636a)
introduced sparse three-byte facts:

```text
[packed egocentric location, feature ID, feature value]
```

The stated objective was to support curricula over varied observation features
and action spaces. Attention, slot-attention, and cross-attention encoders were
explored. This solved shape flexibility, but not yet feature identity,
checkpoint binding, overflow, or unambiguous padding.

The transport was hardened rapidly:

- [`0xFF` became the empty-token sentinel](https://github.com/Metta-AI/metta-public/commit/7942f5e5c58a3c7955a6ba989d2410c3b0e2b3f3),
  separating padding from legitimate zero values.
- [Feature IDs were made dense](https://github.com/Metta-AI/metta-public/commit/9d53da5ae960a46b0b2fcd36f1a41894ffaec137)
  to simplify normalization and conversion. The commit explicitly called this
  silently breaking, revealing that registry order had become model meaning.
- [Nearby tokens were prioritized](https://github.com/Metta-AI/metta-public/commit/e223108b29a712dde3c1eb4180bb32678c1ede95)
  so capacity overflow dropped distant objects rather than arbitrary row-major
  suffixes.
- Feature normalizations moved into environment metadata rather than being
  implicit in model construction.

### 3. Adapter-led migration

Metta separated transport migration from model migration. It first added a
[`token_to_box` adapter](https://github.com/Metta-AI/metta-public/commit/dc9b53a481ce1cedbf9e8547537db38bdb8b1f51),
then made tokens the environment default while reconstructing the old dense
input for existing policies. Only afterward did it
[`remove box encoding`](https://github.com/Metta-AI/metta-public/commit/2ad09a8db6170ad28db91790644ebc13ed7a49a2)
and land a
[`performant token encoder`](https://github.com/Metta-AI/metta-public/commit/6cf3a607f4ab75c77a8837ec749c9502c107e2d4).

That sequence isolated environment, projection, and policy changes. It also
kept old checkpoints usable while the token-native path was measured.

### 4. Symbolic schema binding

Metta next made the feature schema explicit. A
[`feature_spec`](https://github.com/Metta-AI/metta-public/commit/c67a39c910fb7628000ba091fc890af45691ade3)
reported every runtime feature's ID, name, and normalization. The subsequent
[`initialize_to_environment` remapping](https://github.com/Metta-AI/metta-public/commit/87d8a3b8e6d1c7120b017931579b231a8c2e9a8c)
stored the policy's original `feature_name -> embedding ID` mapping and rebound
new environment IDs at load time.

Known features map back to their training-time embedding slots. Unknown
features map to ID 255 during evaluation; continued training may append them to
the stored vocabulary. This provides:

- invariance to runtime feature-ID permutation;
- tolerance of missing and additional features;
- checkpoint portability among compatible configurations;
- environment-owned value normalization.

It does not provide zero-shot semantics for an unknown feature. `UNKNOWN` is a
safe compatibility behavior, not understanding.

### 5. Token-native policy and bounded computation

In September 2025 the
[`token-based ViT became the default`](https://github.com/Metta-AI/metta-public/commit/c1ae4f24910c6deaad64a5bc8b138d7d1dd30de8).
At the archived head, the default path uses feature embeddings, Fourier
coordinate features, normalized scalar values, a twelve-latent Perceiver
bottleneck, and a recurrent/trunk policy. The current
[`observation manual`](https://github.com/Metta-AI/metta-public/blob/main/packages/mettagrid/docs/observations.md)
documents a padded `(num_agents, num_tokens, 3)` `uint8` array and warns that
feature IDs vary by configuration.

The policy-side token cap initially looked like a throughput choice. A later
[`10,000-step clipping audit`](https://github.com/Metta-AI/metta-public/commit/b3a84fe0dce8c948225238e19fd1dc1958201488)
showed otherwise:

| policy cap | steps with any clipping | dropped tokens per agent-step |
|---:|---:|---:|
| 48 | 98.58% | 10.563 |
| 80 | 26.88% | 0.799 |
| 96 | 5.17% | 0.109 |
| 128 | 0% | 0 |

The environment's 200-token buffer dropped nothing in that workload. All loss
was introduced by policy preprocessing. The cap was raised to 128.

This is a central lesson for Etude: a token budget is part of experimental
semantics, not merely model configuration. Truncating a semantic program can
change an ability's meaning rather than merely omit a distant fact.

### 6. Compatibility debt under schema growth

Inventory quantities later grew from eight to sixteen bits. Because token
values remained bytes, Metta represented larger quantities with radix tokens
such as `inv:food`, `inv:food:p1`, and `inv:food:p2`. The extra features shifted
IDs and broke older policies.

Metta restored compatibility with an
[`inventory power-token shim`](https://github.com/Metta-AI/metta-public/commit/8cf487b5f3332ef618d96b298eaaf9f1a3485c56)
that rebuilt the legacy feature ordering and mapped new power tokens to
`UNKNOWN`. The
[`legacy behavior remained the default`](https://github.com/Metta-AI/metta-public/commit/f1064cc6920ac94598adc71da26487f0fdd3ef95)
while selected policies could opt into full decoding.

This kept old policies executable, but those policies remained blind to the
new precision. Compatibility and semantic competence must therefore be
reported separately.

### 7. Checkpoint and implementation correctness

Metta eventually made the architecture contract part of the artifact. Its
[`checkpoint bundle migration`](https://github.com/Metta-AI/metta-public/commit/dceb2b79a0f0c9543d6e030a34447f49be9a9295)
bundled a policy specification with safetensor weights, then loaded weights and
reinitialized environment-dependent buffers. A lightweight typed
`PolicyEnvInterface` carries feature specs, tags, action names, observation
shape, and protocol metadata without coupling the policy to the full engine
configuration.

Structured data did not eliminate semantic implementation bugs. A November
2025 fix found that Torch had silently reversed the packed row/column nibbles
relative to the engine. The
[`coordinate-packing correction`](https://github.com/Metta-AI/metta-public/commit/d1b1887d4e0b5648d12d7e1a4ec032929180cdc6)
is strong evidence for differential fixtures across every implementation of an
input projection.

## Robustness taxonomy

Metta's work separates five concerns that are easy to conflate:

1. **Shape robustness:** variable numbers of sparse facts, explicit padding,
   masks, and a bounded attention bottleneck.
2. **Referential robustness:** symbolic feature names rebind configuration-
   specific runtime IDs to checkpoint-stable embedding slots.
3. **Capacity robustness:** deterministic overflow priority plus measured cap
   distributions and drop counters.
4. **Artifact robustness:** architecture and vocabulary expectations travel
   with weights and are rebound during environment initialization.
5. **Implementation robustness:** engine and policy implementations share
   differential fixtures for packing, ordering, normalization, and masks.

Metta is strong on these dimensions. It is weak on **novel semantic
robustness**: its model consumes learned IDs for flat spatial facts, not typed
nested programs whose familiar operations can recombine on unseen content.

## Etude design consequences

### Variable typed programs, not fixed feature vectors

The primary semantic input should preserve the executable structure of each
visible ability. A program token or node needs stable opcode identity, argument
role and kind, typed values or references, source object/card and ability
segments, and explicit tree/graph structure through parent edges, depth, or an
equivalent grammar. Padding and validity must be explicit.

`CardDefId` may remain as a separate optional identity signal, but the transfer
experiment must be able to ablate it without removing the program.

### A checkpoint-bound `SemanticInputSpec`

Every semantic-policy artifact should carry at least:

- observation/world schema version;
- training-time opcode, role, argument-kind, and tag vocabularies;
- value encodings and normalizations;
- tree/segment encoding rules;
- token/node budgets and overflow policy;
- ContentPack/compiler schema and digest compatibility requirements;
- legacy-adapter version, when present.

At load time, a ContentPack binds runtime numeric IDs to this stored symbolic
vocabulary. Reordering definitions or enum tables must not change policy
meaning.

### Unknown cards are not unknown operations

The admission contract must distinguish:

- **Unseen composition:** an unseen card made entirely from known operations,
  roles, and values. This is valid input and the desired zero-shot transfer
  case.
- **Unknown semantic primitive:** a card requiring an opcode or rule concept
  outside the checkpoint's vocabulary. The pack must fail admission, require a
  new world, or take an explicitly measured unsupported path.

Collapsing the second case to one `UNKNOWN` embedding and counting it as
semantic transfer would invalidate the claim.

### No silent semantic truncation

For each admitted ContentPack, the compiler should prove the maximum supported
program budget or reject oversized content. If a later architecture uses
chunking or hierarchy, it must preserve complete semantics. Any compatibility
cap needs deterministic semantic priority, explicit overflow telemetry, and a
separate result label; generic prefix truncation is not acceptable.

### Compatibility shims remain experimental controls

A program-to-legacy-feature adapter is valuable for controlled migration and
the baseline arm. It should not define the primary semantic representation or
silently hide new primitives from the native semantic arm.

## Required verification

Before interpreting held-out transfer, the semantic input contract should prove:

- runtime opcode-ID permutation invariance;
- CardDef and ability-table reorder invariance;
- checkpoint reload against a reordered compatible ContentPack;
- exact Rust/Python projection parity on versioned fixtures;
- complete batching, padding, masking, program-boundary, and parent-edge
  preservation;
- accepted-pack token counts within the declared budget, with zero silent
  overflow;
- declared rejection or migration behavior for an unknown opcode;
- equivalent semantic programs with and without the optional `CardDefId`;
- program-order sensitivity where execution order matters and invariance only
  where the IR declares siblings commutative;
- token-count, latency, throughput, peak-RSS, and overflow receipts for every
  training/evaluation arm.

The four-arm ladder remains the right causal experiment:

1. Card-ID embedding plus legacy action head.
2. Card-ID embedding plus structured decoder.
3. Semantic-program encoder plus structured decoder.
4. Semantic-program encoder plus structured decoder on held-out cards or an
   entire held-out compiled pack.

Arm 2 isolates the interaction grammar. Arms 3 and 4 test whether executable
meaning transfers. Add schema-ID permutation, checkpoint-rebind, `CardDefId`
ablation, and overflow receipts as robustness controls rather than additional
headline arms.

## Roadmap placement

- The viewer-safe semantic projection owns `SemanticInputSpec`, stable
  symbolic binding, structural tokens, masks, budgets, and parity fixtures.
- The held-out four-arm experiment owns identity ablation, unseen-composition
  evaluation, schema permutation/rebind controls, and throughput/RSS/overflow
  receipts.
- The acceptance-slice compiler owns pack admission: every emitted primitive
  is typed and supported, and unknown primitives cannot silently enter the
  learning path.
- World/version policy owns comparability: projection shape or semantic schema
  changes create a new world, while pure runtime table reorderings do not.

The transferable lesson is to copy Metta's migration discipline and contracts,
not its byte layout. Metta made policies portable across changing sets of known
facts. Etude should make policies compositional across unseen cards built
from known executable meanings.
