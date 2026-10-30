# Worlds

A change to rules meaning or observation/action shape requires a **world version**. Checkpoints,
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
| **w8** | in development, MTG-162, 2026-10-20 | 28 / 40 / 27 / 16; sacrifice/payment continuations and native choice descriptions | compiled sacrifice validation pack; no promoted artifacts | no strength baseline |
| **w9** | in development, MTG-163, 2026-10-21 | 40 / 41 / 27 / 16; restricted mana colors and copied-token legendary status | compiled Temur validation pack; no promoted artifacts | no strength baseline |
| **w10** | in development, MTG-164, 2026-10-21 | 40 / 41 / 28 / 16; temporary Avatar subtype feature | ten exact Elemental setups in one compiled pack; no promoted artifacts | no strength baseline |
| **w11** | in development, optional payments, 2026-10-21 | 40 / 41 / 28 / 16; shared typed optional-payment state and learning schema 7 | `elemental-world-v2`; no promoted artifacts | no strength baseline |
| **w12** | in development, MTG-177, 2026-10-21 | 40 / 41 / 28 / 16; permanent-wide hexproof targeting enforcement | unchanged authored packs; fresh w12 conformance receipts | no strength baseline |
| **w13** | in development, MTG-177 integration, 2026-10-22 | 40 / 41 / 28 / 16; targeting enforcement plus shared-payment repair | unchanged authored packs; fresh combined-runtime conformance receipts | no strength baseline |
| **w14** | in development, MTG-195, 2026-10-26 | 40 / 41 / 28 / 17; controller-selected trigger placement and policy-input v3 occurrence joins | unchanged authored packs; fresh w14 conformance and play fixtures | no strength baseline |
| **w15** | in development, MTG-177, 2026-10-27 | 40 / 41 / 28 / 17; retained resolution eligibility, policy-input v4 relations 64 × 33 | unchanged authored packs; fresh integrated conformance and play fixtures | no strength baseline |
| **w16** | in development, Task 892c47, 2026-10-28 | 40 / 41 / 28 / 17; typed Rust cards compiled in process, first-printing set keys, one default pack | `tla-pool-v1`: every defined card (115 supported); conformance receipts not yet regenerated | no strength baseline |
| **w17** | in development, Task 892c47, 2026-10-29 | 40 / 41 / 29 / 17; cards in engine rules types, learning schema 9 tokens derived from them, attacking permanent column | `world` (the default pack); `semantic-kernel-w17-v1`, `authored-match-w17-v1` and `authored-match-parity-w17-v1` receipts; Academy w17 demo, navigation and trigger-order fixtures | no strength baseline |
| **w18** | in development, Tasks 87e012 and 6875f0, 2026-10-29 | 63 / 55 / 52 / 17 (policy input v5; v1–v4 keep 40 / 41 / 29); TLA casting, costs, zone permissions and private zones; exile cast permissions, hand-only lock, restricted mana by class, foretell, control; counters by kind, Equipment and Auras, Vehicles, layered type/keyword/P/T grants, ability loss, activation X and crew | `world` (the default pack): every defined card; `semantic-kernel-w18-v1`, `authored-match-w18-v1`, `authored-match-parity-w18-v1` and `public-commitment-parity-w18-v1` receipts; Academy w18 demo, navigation and trigger-order fixtures | no strength baseline |

**Current development world: w18.** `managym.WORLD_VERSION` identifies live runs. Earlier
checkpoints (including w17) are **not comparable** to w18; do not reuse their
win rates or load them as current-world policies. Shape alone is insufficient
identity. Exp-11 historically ported opponent components with
`port_legacy_state_dict` and behavioral validation; that utility is now retired.
Current loaders require current-ABI checkpoints. Cross-world *measurements*
are regenerated, never ported (exp-10's precedent).

Update this table in the same PR as any rules-meaning or shape change.

## Running an older world

The engine implements exactly one world. To run an older world, check out the
revision that recorded it and build that engine; current code carries no rules
compatibility shims. Frozen evidence (IR, receipts, checkpoints) stays readable
in place and is never rewritten.

## w18: TLA casting and characteristics (not yet frozen)

The cards of TLA batches T2 (Task 87e012) and T3 (Task 6875f0) are defined in
`cardsets/tla.rs`. T2's rules meaning: casting from exile under exact-object
grants (airbend, impulse exile, linked exile, foretell face down), a hand-only
lock that overrides those grants until its controller's next untap, printed
and granted flashback, announced X and kicker as one cast announcement, free
casts during resolution, restricted mana keyed by spending class (Lesson,
Shrine, Ally spells and ability sources), affinity and activation cost
reduction, exhaust and once-each-turn activation limits, hand- and
graveyard-zone activated abilities (landcycling), private library searches and
choices, and control (a permanent spell enters under its controller).

T3 adds characteristics, counters, replacements and attachments. Its cards and
rules are `CardDefinition`, `Ability` and named `Effect` variants like every
other card, and learning tokens follow from them. A permanent's
characteristics are what `Game` derives from its card and permanent state,
never those inputs read alone: static type, keyword and P/T grants apply in
layer order (`ScopeView`); `Game::has_lost_abilities`, `permanent_name`,
`permanent_has_subtype`, `activated_abilities_for` and `permanent_can_tap` are
the single readers. Counters other than +1/+1 have kinds (stun, fire, double
strike). Equipment and Auras attach to an exact object and grant abilities or
set characteristics; Vehicles crew by tapping creatures. Activated abilities
may announce X and choose several targets that share a creature type; a
permanent may let its controller cast the top card of their library.

Observations add, omitted at their defaults: cast permissions, a stack spell's
announced X, a player's restricted mana, hand-only lock, no maximum hand size
and kept red mana; a permanent's used exhaust and once-each-turn counts,
granted firebending, counters by kind, attachment, set name and creature types
(`is_ally` reads them), granted land type, `blocking`, `abilities_lost` and
`used_abilities`; a stack object's X and a spell's granted keywords.

Policy input **v5** is the default; v1–v4 keep their widths. In every widened
row the new terms start at the old validity column and validity stays last.
Player rows add restricted mana by spending subtype × color (18), the
hand-only lock, no maximum hand size, turn number, seat and kept red mana
(40 → 63). Card rows add 12 cast-permission terms and a stack spell's
announced X (41 → 55); a face-down card is foretold, so the foretold term marks
it. Permanent rows add used exhaust abilities, used once-each-turn abilities,
granted firebending, counters by kind, an attachment join, effect-set
characteristics, granted land, temporary subtypes, a blocking join, lost
abilities and spent activation limits by ability (29 → 52). Relation rows add a
frame's X and a spell's granted keywords (33 → 48), so the relation width alone
tells v5. Learning schema 9 is unchanged: tokens derive from the definitions.
Viewers see an opponent's foretold card only as an exile count, in JSON, text
and tensor alike. `managym/tests/cards/observability.rs` names each field's
columns and tokens. Action shapes are unchanged.

Fresh `conformance/semantic-kernel-w18-v1` replays (same cases as w17), the
authored-match receipt and its live/headless parity (`authored-match-w18-v1`,
`authored-match-parity-w18-v1`: the same 142-Command tape as w17, new digests)
and `public-commitment-parity-w18-v1` bind this world. Academy `*-w18.json`
demo, choice-navigation and trigger-order fixtures reach the same decision
revisions as their w17 predecessors. Belief encoding receipts move with
possible-world space v3.

## w17: one card representation (superseded by w18)

Each card is a function returning the engine's own `CardDefinition`, written in
its rules types, in the set file of its first printing; the authoring language
(`cards/authoring.rs`), the in-process compiler (`semantic/compile.rs`) and IR
parsing and interpretation for current content are deleted. All 161 definitions
were generated once from w16's lowering and proven JSON-identical to it. Each
definition carries its key and the Scryfall id of that printing; neither enters
content digests, so default and authored pack digests equal w16's. Five cards
moved to their actual first printing (`roe.prophetic_prism`,
`mh2.tireless_provisioner`, `ptk.corrupt_court_official`, `eld.epic_downfall`,
`ths.lightning_strike`); every set-file card is supported unless marked
incomplete.

Learning tokens are derived in Rust (`managym.learning_catalog`) by walking each
definition with a `serde` serializer: learning schema **9**, token values are
stable hashes of Rust and printed names, and IR opcodes, roles and enums are
gone. `BoundSemanticPack` binds by content digest and registry name. Manifests
carry `pack_key` (replacing `compiled_semantics` with its IR and source hashes);
`content_pool` rows add `scryfall_id` and name the card key `key`. The world
pool is `world` (formerly `tla-pool-v1`): it describes the default pack, so it
shares that pack's `CardDefId`s and digest; rules fixtures are unsupported rows
without a key. Card catalog modules (`setups`, `pool`, `tokens`) live under
`cards/`. Permanent observations add an
`attacking` column (width 28 → 29) and LLM text, so the state is no longer
visible only through the event window. `tests/cards/observability.rs` guards
that every keyword, permanent and player status and `Effect`/`TriggerCondition`
/`Ability` variant names its tensor column or token.

Fresh `conformance/semantic-kernel-w17-v1` replays (same cases as w15),
`authored-match-parity-w17-v1` (the 142-Command w17 authored tape) and Academy
`*-w17.json` demo, choice-navigation and trigger-order fixtures bind this world;
each w17 demo reaches the same decision revision as its w15 predecessor. The
public-commitment parity proof now replays the w17 authored tape, but has no w17
receipt yet: belief transport rejects Command 68, where a card revealed from the
library enters the villain's hand.

## w16: one card authority (superseded by w17)

Cards are typed Rust functions in the file of the set that first printed them
under `managym/src/cardsets/` (`tla.rs`, `stx.rs`, `lea.rs`, …), each defined
once. `semantic::compile` lowers them in process into the IR the runtime, pool
descriptor and learning projection read; no card catalog or IR is generated on
disk. Definition keys name that set (`lea.forest`, `stx.pop_quiz`, `token.clue`);
compiled IR was proven identical to w15's content apart from those renames. The
few cards the typed language cannot express are native `CardDefinition`s in their
set files; with the rules fixtures they keep the default pack's legacy
`CardDefId` prefix and content digest.

`managym::world` owns `WORLD_VERSION` and the card/deck support check
(`managym.unsupported_cards`). The world pool is every defined card from every
set. A card is supported when its set file defines it with a typed program and
does not mark it incomplete; tests never gate support. `managym.set_cards`
narrows the pool to chosen sets, such as `["tla", "lea"]`.

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

The w7 rules corpus is
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

## w8: Sacrifice and landfall (not yet frozen)

MTG-162 adds selected sacrifice costs, payment alternatives, immediate chosen-color
mana abilities, sacrifice triggers and per-incarnation turn limits. Basic-land
search retains one native operation with count, entry and continuation options.
Last-known facts retain animation for creature-death triggers and effective
keywords for damage after a source leaves. Match-state hash
version 5 includes the additional trigger state. Choice descriptions are native
presentation metadata; bounded tensor widths stay unchanged.

The `elemental-sacrifice-v1` pack contains all 17 owned real cards, five basics
and five tokens. Its two validation rosters are separate from the final ten
decks. Learning schema 4 is at `content/semantic/w8/learning_schema.json`.
The current rules corpus is [semantic-kernel-w8-v1](conformance/semantic-kernel-w8-v1/).
Earlier worlds, sources, schemas, receipts and experiment evidence remain unchanged.
World freeze and release certification remain separate from implementation.

## w9: Temur mana and graveyard casting (not yet frozen)

[The Temur rules contract](docs/rules/elemental-temur.md) covers six added real
cards, granted mana abilities, Lesson-restricted mana, Iroh flashback, and Ran
and Shaw's nonlegendary token. Player features grow from 28 to 40 to retain both
ordinary and Lesson-restricted W/U/B/R/G/C pool counts. Card features grow from
40 to 41 to expose actual legendary status, including copied tokens. Action and
permanent widths remain 20 and 27. Mutable state hash version 6 includes copied
token characteristics; cast provenance lives on stack and battlefield objects.

Learning schema 5 is `content/semantic/w9/learning_schema.json`; earlier schemas
and receipts retain their original contents. The current conformance corpus is
[semantic-kernel-w9-v1](conformance/semantic-kernel-w9-v1/). The validation pack
is separate from MTG-164's final ten-deck admission. These are rules and
integration changes, without a strength claim or a promoted checkpoint.


## w10: one ten-deck Elemental world

`elemental-world-v1` adds the exact ten accepted 40-card decks and outside-game
Lesson pools in [the Elemental contract](docs/rules/elemental/world.md). Its closure
contains 89 selected nonbasic cards, five basics and six token definitions.
Earlier authored packs and receipts retain their bytes and setup meanings.

Temporary battlefield subtype grants are now incarnation-local state, part of
match-state hash version 7 and both viewer projections. The native and Python
permanent encoders add an Avatar-grant feature at index 26; validity moves to
index 27. Player/card/action widths stay 40/41/20. Learning schema 6 adds typed
mandatory discard, death-exile marking and temporary subtype instructions;
`content/semantic/w9/learning_schema.json` retains schema 5. Earlier checkpoints
fail world admission before weights load.

A selected Elemental checkpoint binds the entire ten-setup native roster,
including each Lesson pool, the complete compiled manifest and schema 6. Any
pair or mirror from that roster can execute with explicit pack identity;
foreign or modified setups fail admission. This is compatibility, not a claim
that every admitted matchup was trained or that the policy is strong.

The current conformance corpus is `conformance/semantic-kernel-w10-v1`.
Native all-pairs smoke and a predeclared 14-game same-Command cohort cover
integration. World freeze, full diagnostic, release-browser certification and
strength evaluation remain distinct outcomes.

## w11: shared optional payment continuations

Ordinary optional mana and waterbend share typed progress and outcome effects.
Match-state hash 8 identifies the changed serialized state; learning schema 7
and opcode 65 identify authored payment branches. Tensor dimensions stay fixed.
Frozen opcode 61 lowers to the shared runtime. Earlier sources, schemas and
receipts remain unchanged; w10 checkpoints are rejected. See
[optional payments](docs/rules/optional-payments.md) for ownership and authoring.

The shared-payment repair uses executable witnesses for offers and nested mana
costs within development w11. Its source-bound corpus is
`conformance/semantic-kernel-w11-payment-v1`; earlier conformance and scientific
receipts remain unchanged. This correction changes legal transitions and payment
event ordering, without changing tensor dimensions or the authored w11 pack.
A later scientific owner must bind the actual repaired source and runtime.

## w12: permanent-wide targeting restrictions

CR 702.11b now applies to every permanent target specification, including
`SpellOrPermanent`. Battlefield presence and opposing hexproof are checked once
before wording-specific restrictions, at both selection and resolution.
Own hexproof permanents remain legal when they satisfy the target specification.
A response that grants hexproof can invalidate a selected target under CR 608.2b;
a departed permanent slot cannot follow a new incarnation of the same card.

Tensor shapes, compiled authored content and learning schema 7 are unchanged.
The new [w12 corpus](conformance/semantic-kernel-w12-v1/) records fresh runtime
receipts; w11 and earlier corpora, checkpoints and `erd1` evidence stay frozen.
This bounded repair does not supply trigger-order choices, general protection,
or stack-spell incarnation guarantees. No strength baseline is claimed.

## w13: combined targeting and shared-payment repairs

The w12 targeting repair and the separately landed shared-payment repair now
execute together. The latter changes payment transitions and event ordering:
the `temur-164` checkpoint after command 10 matches the payment corpus, but
differs from the earlier w12 targeting-only corpus. Neither historical receipt
describes the combined source. w13 identifies this combined runtime; it adds no
new tensor shape, learning schema, authored cards or payment implementation.

[Fresh conformance evidence](conformance/semantic-kernel-w13-v1/) uses the same
bounded case manifest and phase matrix. New w13 Academy demo prefixes use the
ordinary Command path. All earlier corpora and demo files retain their bytes.
Current-world admission and monitoring must validate this combined runtime;
historical w12 passes are scoped to their recorded sources.


## w14: controller-selected trigger placement

Controllers explicitly choose the next waiting occurrence to place, within each
of CR 603.3b's two APNAP passes. Targets are chosen at placement; a suspended
placement round resumes before further state-based actions or collection.
Trigger-on-trigger condition matching remains unsupported; typed native fixtures
establish second-pass scheduling only. MTG-196 owns replacement ordering.

Semantic decisions use schema 7, decision capture version 3, and canonical mutable
state hash version 9. Fresh policy input v3 adds the seventeenth action family and
an action-to-continuation frame join. All relations of that frame inform its action;
occurrence counters never become numeric policy features. Missing checkpoint input
versions still decode as v1 and explicitly saved v2 configurations retain their
serialization. Historical-world checkpoints are rejected for current-world play.

Native offers say “Put on stack next” and describe the exact triggered ability
or delayed payload. Public stack inspection and text capture reuse that derived
description; full card text remains available in source inspection.

The new conformance root is `conformance/semantic-kernel-w14-v1`. Existing w13 and
earlier corpora, play roots, decks and checkpoint evidence are retained unchanged.
The existing-card trigger witness uses the explicit scenario recipe in
`manabot.verify.trigger_order` before ordinary human Commands and agent actions;
`academy/fixtures/trigger-order-w14.json` retains its root and content identities.
This is a constructed root, not a seed-only authored-match prefix.


## w15: resolution-start target eligibility

CR 608.2b eligibility is evaluated once at resolution entry for each chosen
clause/target slot, across spells, activated abilities and triggers. Instructions
and suspended choices retain that result; live presence checks separately reject
departed objects. Illegal slots remain positional, all-invalid targeted objects
perform no instructions, and zero optional targets still resolve. Controller-selected
trigger placement from w14 remains intact.

Canonical mutable-state hash 10 includes the retained eligibility slots. Policy
input v4 extends the 64 relation rows from 32 to 33 columns, adding unknown/legal/
illegal eligibility at column 32. Validity stays at column 31; v2/v3 layouts and
checkpoint omissions remain unchanged. Capture/text v4 carries the same native
fact, with unchanged embedded decision schema 7. Earlier checkpoints are rejected
for current-world play, and historical captures retain their recorded semantics.

Fresh `conformance/semantic-kernel-w15-v1` receipts and Academy w15 prefixes bind
the integrated source. Earlier corpora, demos, authored packs and experiment
artifacts remain unchanged. These bounded fixtures do not establish exhaustive
rules correctness, scientific baselines or final release certification.
