# What large Magic engines teach the Standard milestone

Research date: 2026-11-04. Status: evidence and recommendations, not an approved
engine redesign. This report addresses Jack Heart's request to compare other
engines for composability, implementation scale, correctness and performance.
No external engine was built or benchmarked.

## Identity, scale and limits of the comparison

Jack Heart confirmed [XMage](https://github.com/magefree/mage) and
[Forge](https://github.com/Card-Forge/forge) by supplying their links on
2026-11-04.
**DeepScry** identifies itself as a Rust
reimplementation inspired by Forge. Magarena provides an additional comparison
with a long-lived search-oriented engine.

The inspected source revisions are:

| Project | Pinned revision | Evidence of card scale |
|---|---|---|
| [Forge](https://github.com/Card-Forge/forge/tree/a4281f7e58b50542decce8971d81bca4307fd8b4) | `a4281f7e58b50542decce8971d81bca4307fd8b4` | 34,134 `.txt` files under `forge-gui/res/cardsfolder/` |
| [XMage](https://github.com/magefree/mage/tree/987fd4e9c73e9f113e235f8d8f98628f2d391667) | `987fd4e9c73e9f113e235f8d8f98628f2d391667` | 32,782 `.java` files under `Mage.Sets/src/mage/cards/`; README claims over 32,000 unique supported cards |
| [Magarena](https://github.com/magarena/magarena/tree/efa0aba85e681816a92b4938b28741d540384e35) | `efa0aba85e681816a92b4938b28741d540384e35` | 13,071 `.txt` definitions and 1,917 `.groovy` files under `release/Magarena/scripts/` |
| [DeepScry](https://deepscry.net/guide/introduction.html) | Deployed guide retrieved 2026-11-04 | Guide references approximately 31,000 upstream scripts; actual engine coverage unverified |

Counts above were independently calculated from complete, non-truncated GitHub
recursive trees, using the stated paths and suffixes. They measure authored
inventory, not distinct legal deck entries or correct implementations. Tokens,
special cards and supporting classes can affect the counts. XMage's supported
total is its maintainers' claim, not our test result. DeepScry's publicly indexed
`DeepScryAI/DeepScry` repository returned HTTP 404 through both GitHub and its API;
its deployed primary documentation remained readable. A large imported library
must not become a claim of a large correctly executable library.

## XMage: explicit card code over a large reusable rules library

XMage's ordinary card is a Java class assembling targets, costs, abilities and
effects. Lightning Helix adds `TargetAnyTarget`, `DamageTargetEffect(3)` and
`GainLifeEffect(3)` to its spell ability. The damage effect itself accepts a
`DynamicValue`, reads target pointers, and routes damage to the appropriate game
object. These are meaningful reusable components rather than one engine handler
per card. [Card definition](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage.Sets/src/mage/cards/l/LightningHelix.java),
[shared damage effect](https://github.com/magefree/mage/blob/987fd4e9c73e9f113e235f8d8f98628f2d391667/Mage/src/main/java/mage/abilities/effects/common/DamageTargetEffect.java).

Its contributor guide tells authors to find a similar card, reuse its ability
construction, adjust filters or conditions, register the new printing, and test.
That is an effective discoverability pattern: the existing catalog is a library
of examples. However, copying construction code can also preserve an accidental
assumption from the example. Reusing an effect implementation is valuable;
copying an old card's timing, target scope or controller binding without auditing
the new text is hazardous. This is an inference about the workflow, not a claim
that every copied implementation is wrong. [Authoring guide](https://github.com/magefree/mage/wiki/Development-HOWTO-Guides).

**Keep:** strongly named semantic building blocks, searchable real-card examples,
independent card behavior tests, and explicit set implementation inventories.
XMage's Wilds of Eldraine tracker demonstrates per-card ownership and completion
accounting. [WOE tracker](https://github.com/magefree/mage/issues/10759).

**Avoid importing:** class-per-effect/card inheritance and copy boilerplate merely
because they achieved breadth. Managym already gets explicit authored semantics
and compiler checking through Rust values. Its `EffectValue`, `ObjectSelector`,
`TargetRequirement` and `Modification` should supply the same reuse with a
smaller number of authoritative evaluators. Familiar card names should help find
examples, never become dispatch keys in engine logic.

## Forge: broad composition with cheap authoring, expensive semantic surfaces

Forge's Lightning Helix is a compact script: a damage operation with an any-target
selector and amount, followed by a named life-gain subability. A card author can
combine existing capabilities without adding Java. This is direct evidence for
the desired multiplication of reusable dimensions: operation × selector × amount
× sequencing. [Helix script](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-gui/res/cardsfolder/l/lightning_helix.txt).

The tradeoff becomes visible in execution. `DamageDealEffect` branches on many
string parameters, including division, choices, relative targets, optional
deciders, remembered damage and excess damage. Its flexibility is real, but the
meaning and valid combinations of those parameters must be maintained across
callers. A concise definition can conceal a large semantic compatibility surface.
Managym can learn the useful orthogonal dimensions while representing them as
typed fields and closed variants, without adopting the script format.
[Damage execution](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-game/src/main/java/forge/game/ability/effects/DamageDealEffect.java).

Execution reuse does not automatically give AI reuse. Forge has a separate
`DamageDealAi` implementation, with `AILogic` parameters, affordability decisions
and special heuristics. The inspected file is 1,208 lines. That is not a speed
measurement or proof of bad quality; it is concrete evidence that a generic
effect may acquire a second, substantial interpretation in the AI. Managym's
policy tokens derived from the actual `CardDefinition` are a valuable difference.
New effects must remain learnable through the same representation rather than
gaining an independent card-specific AI language.
[Damage AI](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-ai/src/main/java/forge/ai/ability/DamageDealAi.java).

Forge's simulation copier is another instructive cost. It explicitly remaps
players, targets, copied characteristics, remembered objects, stack entries,
departed-object state and effects. These are exactly the relationships a search
fork must preserve. The source's detailed handling shows why clone correctness
belongs to the domain model, not a later search-only approximation.
[GameCopier](https://github.com/Card-Forge/forge/blob/a4281f7e58b50542decce8971d81bca4307fd8b4/forge-ai/src/main/java/forge/ai/simulation/GameCopier.java).

**Keep:** cheap assembly of shared operations, explicit result/context references,
and a clear boundary between a definition and its executor. **Avoid:** stringly
typed option growth, duplicated semantic interpretation in AI, and a separate
approximate game representation for simulation. The current native-definition
contract already permits the good parts without a DSL migration.

## DeepScry: relevant engineering principles, incomplete verification

DeepScry's official guide says it consumes Forge's scripts and that Forge owns
their semantics. This saves authoring effort but moves the burden to complete
compatibility with an evolving external language. The guide explicitly separates
the upstream language from DeepScry's unfinished coverage.
[Card-script introduction](https://deepscry.net/guide/part3/card_script_intro.html).

Its cache policy is directly relevant: derived data belongs behind the owning
object's API; every write must invalidate or update it; callers should not manage
freshness. The dated June survey identifies a direct tap-state write that bypassed
mana-cache invalidation and an unused parallel mana index. Without current source
access these are historical documented findings, not claims about today's code.
The lesson applies to managym's revision caches: enforce ownership structurally,
keep slow recomputation as an oracle, and remove an abandoned optimization rather
than retaining two authorities.
[Caching policy](https://deepscry.net/guide/part2/caching_policy.html).

Its snapshot design records completed choices and stops before presenting the
next one. Its determinism rules require persistent state across a choice boundary
to survive reconstruction, while short-lived resolution data travels explicitly.
These closely match managym's need for exact suspended decisions, replay and
search forks. They reinforce existing direction rather than justify a new undo
system. [Snapshot contract](https://deepscry.net/guide/part2/snapshot_architecture.html),
[determinism contract](https://deepscry.net/guide/part2/deterministic_simulation.html).

There is a material evidence limit: the deployed script reference still contains
a very early implementation-status checklist, inconsistent with the broader
features described elsewhere. Its upstream count cannot establish current card
support, and its high-performance description supplies no comparable benchmark.
We should neither dismiss its design nor repeat an unverified speed or coverage
claim. [Script reference](https://deepscry.net/guide/part3/card_script_spec.html).

## Magarena: search-oriented reuse with another maintenance tradeoff

Magarena's Lightning Helix uses an effect-expression definition plus evaluation
and timing hints. More complex implementations can use Groovy. Its MCTS code
clones the game for simulation and performs bounded playouts. This is useful
evidence that declarative common cases can coexist with search, but an escape
hatch for arbitrary per-card code and separate evaluation hints can accumulate
their own maintenance surface. The inspected Helix text also retains older
target wording: a large catalog needs ongoing Oracle/version maintenance, not
just initial implementation. [Helix definition](https://github.com/magarena/magarena/blob/efa0aba85e681816a92b4938b28741d540384e35/release/Magarena/scripts/Lightning_Helix.txt),
[MCTS implementation](https://github.com/magarena/magarena/blob/efa0aba85e681816a92b4938b28741d540384e35/src/magic/ai/MCTSAI.java).

## Implications for managym's implementation plan

1. **Batch by shared semantic authority, across sets.** A batch should extend one
   operation, selector, amount, event, cost, duration or continuation contract,
   then implement the cards that compose it. Require at least two materially
   different examples when claiming a reusable abstraction. A truly unique
   mechanic can still deserve a primitive; arbitrary reuse quotas would distort
   rules.
2. **Make rule distinctions explicit.** Trigger event versus intervening condition,
   target selection versus resolution legality, physical card versus incarnation,
   and remembered result versus live query are independent dimensions. Treat
   their cross-products as directed test cases, not informal author knowledge.
3. **Keep one executable definition and one learning interpretation.** Native
   typed values, exhaustive visitors, derived learning tokens and viewer-safe
   observations are an advantage. General constructors should assemble those
   values; they should not hide a second card language or evaluator.
4. **Record four separate progress measures.** Count identities inventoried,
   definitions authored, mechanics executable, and independent behavior evidence.
   A report can expose known bugs without blocking playable cards. Track shared
   defect families and affected cards alongside progress.
5. **Require a deletion story for refactors.** When generalizing a special case,
   migrate every applicable existing caller and remove its old executor and
   exclusive machinery. A generalized path beside permanent special paths does
   not deliver the requested simplification.
6. **Measure composition's cost at actual workloads.** Retain same-runner regular
   steps and exact Search-64 comparisons; add bounded representative Standard
   cases when they introduce new expensive mechanics. Attribute time to action
   generation, payment, derived characteristics, stabilization, cloning and
   observation only after profiling. Keep complete legal choices and game
   semantics fixed for equivalent-workload speed claims.
7. **Treat cheaper forks as a hypothesis.** Persistent collections, copy-on-write
   and undo each move costs elsewhere. Managym already has exact full-clone forks
   and cached derived views. Any replacement needs measured net benefit and
   equivalence across suspended costs, triggers, LKI, hidden knowledge and RNG;
   another project's architecture is not that evidence.

The common success is a reusable semantic vocabulary with years of concrete
examples. The common risk is that adding breadth also adds hidden interpretation
surfaces: parsers, parameter combinations, AI heuristics, copying logic and
coverage rules. Managym should grow the former while deliberately controlling
the latter.

Validation: source and documentation inspection only; GitHub inventory counts
used non-truncated trees at the revisions above. No engine executions,
performance comparisons, conformance certification or production changes.
