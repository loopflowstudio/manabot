# 12: Semantic Programs + Learning Input Contract

## Finish line

The checked-in typed IR feeds a viewer-safe, versioned semantic-program input.
The agent can consume what visible cards and abilities do structurally, emit
structured commands, and demonstrate whether known operations recombine on
unseen cards or an unseen compiled pack.

## Changes

### Typed Program Projection

Project visible card and ability IR into variable-length typed programs. Do not
flatten effect trees into one fixed-width per-card vector or make registry
ordinals checkpoint meaning. Preserve:

- stable opcode, role, argument-kind, selector, condition, and value identity;
- source object/card and ability boundaries;
- parent/child, depth, segment, or equivalent program structure;
- execution order where order is semantic;
- explicit lengths, validity masks, and viewer-safe visibility;
- coverage for visible cards in hand, battlefield, stack, graveyard, exile,
  and any other admitted zone relevant to planning.

Keep `CardDefId` as a separate optional identity feature so it can be ablated
without removing the semantic program.

### SemanticInputSpec + Checkpoint Binding

Bundle the input contract with every semantic-policy checkpoint:

- world/schema version;
- training-time symbolic vocabularies and runtime-ID remapping;
- value encodings and normalizations;
- structural/segment encoding rules;
- token/node budgets and overflow policy;
- ContentPack/compiler compatibility requirements and digests;
- legacy-adapter version when present.

Loading against a compatible ContentPack rebinds runtime IDs by symbolic
meaning. CardDef, ability, opcode, and tag table reordering must not change the
projected program or policy result.

### Admission, Unknowns, and Capacity

An unseen card composed entirely from known typed primitives is valid and is
the transfer target. A genuinely unknown opcode or role must fail ContentPack
admission, require a new world/checkpoint migration, or take an explicitly
measured unsupported path. It must not silently collapse to `UNKNOWN` during a
run that claims semantic transfer.

The compiler proves the maximum semantic-program budget for every admitted
ContentPack or rejects oversized content. The primary semantic path never
silently prefix-truncates an ability. Any bounded compatibility path reports
token counts, clipping, and deterministic semantic priority separately.

### Migration Adapter

Preserve a semantic-program-to-legacy-observation adapter long enough to keep
the current policy as a controlled baseline. The adapter is not the native
semantic representation and may not hide new primitives from the semantic arm.

### Four-Arm Transfer Experiment

Run the causal ladder:

1. Card-ID embedding plus legacy action head.
2. Card-ID embedding plus structured decoder.
3. Semantic-program encoder plus structured decoder.
4. Semantic-program encoder plus structured decoder on held-out cards or an
   entire held-out compiled pack.

Arm 2 isolates the choice grammar. Arms 3 and 4 test the hypothesis that the
policy can read abilities as a language of known executable operations.

Report more than training-deck win rate:

- zero-shot and limited-retraining strength on held-out cards/packs;
- known-operation recombination and targeting/choice-role transfer;
- `CardDefId`-present versus identity-ablated results;
- legality and complete choice coverage beyond 32 candidates;
- runtime-ID permutation and compatible-pack reorder invariance;
- checkpoint restore/rebind against a reordered compatible pack;
- token-count distributions, overflow, latency, throughput, and peak RSS;
- pre-registered confidence intervals and seed/deck/seat balance.

### DSL Expressiveness Audit

Review all implemented cards against the DSL. Document:

- cards fully representable declaratively;
- cards requiring special handling or DSL extensions;
- patterns that appear frequently and deserve first-class DSL support.

Also record the ratio of new-card content changes to kernel/IR changes. If most
admitted cards require a new semantic primitive, stop and redesign rather than
treating vocabulary growth as transfer.

### Contract Verification

- Exact Rust/Python projection parity on versioned fixtures.
- Opcode-ID, CardDef-order, ability-order, and tag-ID permutation tests.
- Program boundary, masking, padding, parent-edge, and ordering tests.
- Unknown-primitive admission and world-migration tests.
- Accepted-pack maximum-budget proof with zero silent overflow.
- Checkpoint round trip with its complete `SemanticInputSpec`.
- Differential outcomes between the legacy adapter and native semantic path
  wherever they claim equivalent information.

## Done when

- Every visible admitted ability has a complete viewer-safe typed program and
  the native input has no dependency on incidental runtime table order.
- Checkpoints bundle and successfully rebind the full `SemanticInputSpec`.
- Compatible reordering and cross-language parity gates pass; unknown
  primitives and oversized programs cannot enter silently.
- The four-arm experiment is reproducible and reports transfer, identity
  ablation, legality, overflow, latency, throughput, and peak RSS.
- Positive semantic generalization requires arm 4 to outperform the
  identity-only controls on pre-registered held-out evidence; a null result is
  still a completed, decision-bearing experiment.
- The DSL expressiveness audit and content-change-to-kernel-change ratio are
  published with documented gaps and a continue/redesign decision.

Historical motivation and the source-backed comparison are in
`docs/research/metta-observation-robustness.md`.
