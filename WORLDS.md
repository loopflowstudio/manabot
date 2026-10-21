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
| **w5** | attack declaration encoding, 2026-10-15 | 28 / 39 / 25 / 16; action width 18 including declaration and validity | no promoted checkpoints | no strength baseline yet |
| **w6** | action identity audit, 2026-10-15 | 28 / 39 / 26 / 16; action width 20 | no promoted checkpoints | no strength baseline yet |
| **w7** | in development, MTG-161, 2026-10-20 | 28 / 40 / 27 / 16; hybrid cost metadata and Elemental native rule primitives | compiled Elemental validation pack; no promoted artifacts | no strength baseline |

**Current development world: w7.** `managym.WORLD_VERSION` identifies live runs. Earlier
checkpoints (including w6) are **not comparable** to w7; do not reuse their
win rates or load them as current-world policies. Shape alone is insufficient
identity. Exp-11 historically ported opponent components with
`port_legacy_state_dict` and behavioral validation; that utility is now retired.
Current loaders require current-ABI checkpoints. Cross-world *measurements*
are regenerated, never ported (exp-10's precedent).

Update this table in the same PR as any shape change.

## w7: Elemental rules and compiled content (not yet frozen)

MTG-161 adds two-color hybrid costs, private basic-land search, printed discard
counts, noncreature-cast triggers, conditional combat eligibility, and additional
native Elemental definitions. Hybrid costs retain their pips in native/Python
observation metadata. Card rows add indestructible at 38 and move validity to
39; permanent rows add indestructible at 25 and move validity to 26. The outside
marker remains card index 37. Match-state hash version 4 includes spell mode,
target clauses, scry ordering, and per-turn cast history as well as counted
discard continuations. An activated ability's
independent effects now survive its source leaving the battlefield (CR 113.7a).

All 42 owned cards execute natively and through the `elemental-combat-v1`
compiled pack. Its two 46-card validation rosters cover the owned pool and five
basics; four token definitions complete the closure. These rosters do not adopt
the final ten product decks. Learning schema 3 in
`content/semantic/w7/learning_schema.json` adds the corresponding typed
operations, quantities, hybrid pips and nested choices. The frozen learning
schema and earlier authored sources remain unchanged. Model permanent features
retain indestructible while excluding only the categorical definition join.

The current rules corpus is
[semantic-kernel-w7-v1](conformance/semantic-kernel-w7-v1/), including inherited
matchup cases and compiled Elemental cases in both seat orders. Earlier corpora,
`erd1` receipts, authored IR and checkpoints retain their identities. Release
validation and world freeze remain with gate; no new training or strength
comparison is certified by these checks.

Gate's current limits are recorded in
[Elemental contracts](docs/rules/elemental-combat.md#gate-evidence-and-remaining-limits--2026-10-20).
The w7 authored-match authority and replay parity have separate versioned
receipts; historical public-commitment/RUL-9 checks and pinned Linux visual
validation remain unresolved. w7 is not frozen.

## w6: preserve action and object identity

The ordinary action row retains the 16 family bits and declaration bit, adds
mode index at 17 and activated-ability index at 18, and moves validity to 19.
Selectors encode index + 1, reserving zero for absence; both encoders reject
indexes above 16,777,215 to preserve exact float32 integers. The native action
order and rules outcomes are unchanged.

Permanent rows add backing card definition + 1 at index 24; validity moves to
25. The model treats this field as a categorical join, not a numeric feature.
With semantic inputs enabled, each permanent receives its own card's checked
program embedding, including tokens. Without a semantic pack, visible cards,
owned outside Lessons and permanents share a learned definition embedding bound
to the match content manifest. This fallback distinguishes definitions but does
not claim transfer to unseen cards. Zero denotes padding. Unadmitted definition
IDs fail explicitly. Architecture layout version 2 records the new ownership.

These changes remove forced ties between different permanent targets, modal
choices, same-source activated abilities, and generic card choices. Earlier
checkpoints are rejected before loading weights. Frozen recipes and receipts
retain their original identities; new training must explicitly select w6.
The rules conformance corpus remains w4. Correctness and learnability tests
establish representational capacity, not improved full-game strength.

## w5: distinguish attack from decline

The ordinary action tensor now includes the existing `declared` choice at index
16, after the 16 action-family bits. Validity moves from index 16 to 17.
Attack and decline previously had identical family/focus inputs and therefore
identical policy scores. The extra bit allows the shared policy head to learn
either preference. Blocking declarations also carry their existing choice bit;
other actions use zero. Native legal actions and their ordering are unchanged.

Both native/vector and Python encoders share this layout. Family priors and
diagnostics read only the 16 family bits. Ordinary checkpoint admission rejects
w4 bindings before loading weights; no weight padding or compatibility relabeling
is provided. Frozen w4 recipes, checkpoints and experiment receipts retain their
identities and require their historical runtime. New training recipes must name
w5 explicitly. The rules conformance corpus remains w4 because rules and native
Commands have not changed. This input fix makes no new strength claim.

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
