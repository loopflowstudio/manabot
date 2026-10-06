# Worlds

An observation/action-shape change is a **world version**. Checkpoints,
shards, and reports are only comparable within a world; cross-world numbers
never share a table untagged. New worlds are frozen deliberately (batch the
shape changes), and on each freeze the headline baselines are re-run before
new claims are made. (Cost of ignoring this, measured: every exp-07 artifact
was dead on arrival in exp-10.)

| world | frozen at | shape (player / card / permanent / action-types) | live artifacts | headline baselines |
|---|---|---|---|---|
| **w0** | pre-2026-07-09 (first-light era) | 26 / 29 / 5 / 7 | none (all superseded) | exp-04 terminal-only 60–75% |
| **w1** | rules stage 1–2 (`2d124c9`..`50e0a1f`) | 27 / 37 / 7 / 14 | none (exp-07 artifacts dead per exp-10) | exp-07 student_r0 87%, ladder ≈N=7 |
| **w2** | rules stage 3–4 + conformance (`cb80331`..`a9f1f91`, max_actions 32 @ `55a0b4b`) | 28 / 38 / 24 / 14 | exp-10 V + BC student; exp-11 arms (incl. ported student_r0, validated 86.5%) | exp-06 PPO 60–77%; exp-10/11 (pending merge) |
| **w3** | corrected Learn evidence, 2026-09-24–29 (certification unfinished) | 28 / 39 / 24 / 16 | frozen Learn/checkpoint evidence; see [Learn record](docs/rules/learn-lesson.md) | no completed new baseline certification |
| **w4** | ETU-88, 2026-09-29 | 28 / 39 / 25 / 16; 13 decision kinds | Gran-Gran, Proft's Eidetic Memory, Combustion Technique and required base rules | no strength baseline yet |
| **w5** | ETU-107, 2026-10-06 | unchanged tensor widths; semantic decision schema 7, typed choice contexts and canonical prefix order 1 | opt-in object compound decoder; no ported artifacts | pending; no strength claim |

**Current world: w5.** `managym.WORLD_VERSION` identifies live runs. Earlier
checkpoints (including w4) are **not comparable** to w5; do not reuse their
win rates or load them as current-world policies. Shape alone is insufficient
identity. Exp-11 historically ported opponent components with
`port_legacy_state_dict` and behavioral validation; that utility is now retired.
Current loaders require current-ABI checkpoints. Cross-world *measurements*
are regenerated, never ported (exp-10's precedent).

Update this table in the same PR as any shape change.

ETU-75 completes the ordinary policy input binding against **w4**, preserving
the native rules world from ETU-88. The base feature widths above remain
unchanged. Additional derived `semantic_cards` (two sides × card capacity) and
`known_hand` (two sides × card capacity × definition/count) arrays join the
already-public native projection to complete typed programs. Ordinary policy
input version 1 binds these arrays and the learning schema explicitly; earlier
unbound w4 files are rejected, not treated as equivalent policies. Historical
w3 and w4 results retain their original input contracts and evidence limits.

## w4: complete UR additions

Jack Heart accepted a new world on 2026-09-29 for ETU-88. This world adds
mandatory discard and legend-rule decisions, seven-card cleanup and Proft's
hand-limit exemption, conditional spell-cost reduction, combat/draw-count
abilities, and temporary death-to-exile replacement. The permanent tensor
adds the replacement flag at index 23; validity moves to index 24. Action
kinds remain 16; decision kinds add Discard (11) and LegendRule (12), both
using the existing SelectCard action. Match-state hashes use v3 and semantic
decisions/observations use v6. Visible card records now carry registered rules
text, including owned outside candidates.

The authored UR and GW decklists are unchanged. Jack's revised 40-card UR list
is a custom setup for ETU-87 to measure, not an adopted authored deck. The
new definitions use the ordinary declarative card registry and effect
interpreter; they are not added to the frozen compiled two-deck semantic IR.
Semantic-program checkpoints that require that IR cannot claim these cards
as admitted vocabulary. Random and search consumers use the full native
rules and legal decisions.

Frozen receipts, checkpoints, and prior win-rate measurements keep their
original identities. The earlier 38% Search-64 result is context, not a w4
baseline. Re-run both authored and candidate lists under matched w4 consumers
before comparing strength. See [ETU-88 evidence](docs/rules/ur-lessons-increment.md).

## Semantic input compatibility

The semantic-program input adds meaning-level compatibility requirements on
top of tensor dimensions:

- A checkpoint bundles its complete `SemanticInputSpec`: symbolic vocabularies,
  value/structure encodings, masks, budgets, compatibility rules, and relevant
  ContentPack/compiler digests.
- Runtime CardDef, ability, opcode, role, or tag table reordering is transport
  churn, not a new world. Load-time symbolic rebinding must preserve the exact
  projected program and policy result.
- Adding or changing a semantic primitive, structural encoding, visibility
  rule, normalization, or budget behavior creates a new world unless an
  explicit migration proves equivalence.
- An unseen card composed entirely from known primitives may be evaluated in
  the same world. A card requiring an unknown primitive is rejected or moves to
  a new world; mapping it to `UNKNOWN` does not count as semantic transfer.
- Every admitted ContentPack must fit its declared semantic-program budget with
  zero silent truncation. Token-count and overflow receipts travel with
  experiment results.

The full rationale and required controls are in
[`docs/research/metta-observation-robustness.md`](docs/research/metta-observation-robustness.md).

## w5: shared choice meaning

ETU-107 extends the shared InteractionOffer with fixed public action parameters,
program-bound mode/ability references and role contexts. Compound projections bind
schema 7, revision and canonical factorization order 1. managym owns constructive
prefix support and next-Command prefix partitions. Padded object layout and the
historical flat/label-compound equations remain unchanged; object features are
selected explicitly by AgentSpec and add their own parameters.

The serialized semantic contract changes even though base tensor widths do not.
Ordinary admission retains exact world/setup/input checks: historical w4 artifacts
remain evidence and require their original runtime. No weights or measurements
are ported. New workflow fixtures and the w5 conformance corpus are software
acceptance only; headline strength baselines remain pending.
