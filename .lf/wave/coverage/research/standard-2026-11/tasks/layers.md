Complete composable layer evaluation with concrete interaction witnesses

Jack Heart explicitly authorized a separate layers implementation Task on
2026-11-04, in parallel with the three engine-refactor Tasks. This is ONE Task
for layers, not a Task per card pair and not an FDN-set implementation. Use the
code Graph, publish for review and stop at the interactive boundary. No merge
or training experiment launch is authorized.

Outcome: the one Game::characteristics authority correctly composes interacting
type/color/ability/power/toughness effects, including dependencies within a layer,
and supplies consistent results to all rule and policy consumers. Preserve a
cheap common path and revision caching with debug recomputation.

Current source (5dd11cec4; recheck main) explicitly supports a layer/timestamp
subset and excludes layer 7d switching and CR613.8 dependencies. New semantics
belong behind the existing derived view, not separate effective-property readers
or mutable characteristic ledgers scattered among callers. The runtime currently
does not expose effective colors through Characteristics; examine this while
designing complete characteristic composition rather than retaining divergent
color readers.

Implement dependency-aware ordering and re-evaluation within layers, correct
timestamp fallback for dependency loops, continuation of effects across layers,
and discovery of abilities/effects granted during evaluation. Distinguish
characteristic-defining abilities and other setters. Use explicit evaluation
context so layer internals do not recursively invoke public characteristics.
Review cache granularity and every dependency read, including tap state, zones,
history and ability sources. Keep candidate indexes as discovery only and the
derived rules computation as authority. Do not merely hard-code witness names.

Start with Beorn the Fierce (HOB) + Kellan, Planar Trailblazer (FDN), suggested by
Jack Heart. This witnesses layer-4 addition versus replacement of creature types,
later Bear-based +2/+2, persistent trample counters and granted abilities. It
tests timestamps and cross-layer composition, not by itself a CR613.8 dependency.
Compare upgrading Kellan to Rogue before versus after Beorn adds Bear. Include
both cards' real definitions and complete visible behavior, reusing Adventure
or other unrelated mechanics only if actually needed (Beorn the Fierce itself
is not the Beorn Adventure card).

Jack Heart also nominated Blood Moon, Urza's Saga, Humility, Dress Down and
Opalescence. Use a compact interaction matrix spanning true dependency override,
timestamp order, ability removal, granted abilities, CDAs, counters and SBAs.
Historic non-Standard pairs may be isolated rules fixtures; do not confuse these
with implementing a full non-Standard card pool. Recheck CURRENT official rules:
old Blood Moon/Urza's Saga auto-sacrifice claims changed with the 2025 FINAL
FANTASY rules update. Older forum posts are leads, not current expected results.

Ten nominated pairs beyond Beorn/Kellan, independently checked against primary
sources and judge analyses; confirm against the pinned rules at implementation:
- Blood Moon/Urborg: true layer-4 dependency; Urborg does not add Swamp in
  either timestamp order because Moon removes its generating ability first.
- Blood Sun/Urborg: contrast layer 4 then 6; lands still gain Swamp.
- Blood Moon/Urza's Saga: resolve chapters I and II, then Moon before III.
  Saga retains the separately granted mana/Construct abilities and remains
  without chapter abilities. Do not use the ambiguous initial-entry-counter
  sentence in the 2026 article as an expected value; distinguish current CR.
- Dress Down/Magus of the Moon: Magus loses abilities but nonbasics remain
  Mountains because its type effect already applies in layer 4.
- Dress Down/Tarmogoyf: without other modifiers, Goyf is 0/1; its CDA is
  removed before layer 7a.
- Dress Down/Opalescence: with Opalescence already present, Dress Down enters
  as an abilityless 2/2, neither draws nor triggers its end-step sacrifice.
- Humility/Opalescence: one Opalescence; Humility is an abilityless enchantment
  creature, 1/1 when Humility has the later timestamp, 4/4 when Opalescence
  does. Opalescence does not animate itself.
- Ygra, Eater of All/Imprisoned in the Moon: enchant your own Ygra; it becomes
  a land with the granted mana ability and loses ward/death trigger, while
  other creatures remain Foods with the granted Food ability (CR613.6).
- Mycosynth Lattice/March of the Machines: true layer-4 dependency; ordinary
  previously noncreature lands become 0/0 artifact land creatures and die
  absent toughness bonuses. Lattice and March become 6/6 and 4/4 respectively.
- Painter's Servant/Dress Down: choose a color with Painter before Dress Down;
  the color addition persists through ability removal (layer 5 then 6).

Useful extra integration witness from the supplied Reddit thread: Merfolk
Trickster/Dryad of the Ilysian Grove removes the additional-land permission but
not the lands' five basic land types. Do not turn every historic card into full
world support: isolate appropriate rules fixtures and keep Beorn/Kellan real.

Primary anchors include the official Time Spiral Remastered, Rivals of Ixalan,
Modern Horizons 2, Bloomburrow and FINAL FANTASY release notes; CR613 and CR714.
Specific Dress Down analysis:
https://blogs.magicjudges.org/searegion/2024/04/15/delving-deeper-dress-down/
Current CR at research time:
https://media.wizards.com/2026/downloads/MagicCompRules%2020260925.txt

User-provided starting sources:
https://www.magic.gg/news/metagame-mentor-the-30-weird-interactions-modern-players-should-know
https://www.reddit.com/r/magicTCG/comments/x9zicr/weird_layers_interaction_im_trying_to_understand/
https://www.reddit.com/r/EDH/comments/8ie5bs/what_is_the_weirdest_card_interactions_you_have/
https://draftsim.com/strange-mtg-rules-interactions/
https://www.mtgsalvation.com/forums/magic-fundamentals/magic-general/331972-most-confusing-rules-interactions
https://www.tcgplayer.com/content/article/20-Weird-Modern-MTG-Card-Interactions-You-Need-to-Know/31534b87-fbd3-42bb-a40c-43edc58f34a6/

Acceptance: independent expected outcomes for a concise matrix, both relevant
timestamp orders, source removal/re-entry, copy/grant/ability-loss combinations,
and a clone at a real decision boundary. Check combat/targeting/payment consumers
and current native/Python policy/text observations against the same derived view.
Exercise cache invalidation with real event-driven mutations, not only direct
fixture state. Use bounded debug/reference recomputation tests and preserve
frozen observation/evidence contracts. No claim of exhaustive Magic conformance.

This Task owns statics/layer evaluation and characteristic cache dependencies.
Composable execution owns pending frames, replacement proposals and contextual
source/entry callers. Mana consumes layered source abilities; naming owns large
choice representation. Coordinate narrow interfaces and integrate through normal
Task sync, without building alternate authorities or blocking unrelated simple
card definitions on this work.

Measure ordinary steps and search total/tail under the existing same-runner
Engine Budget, retaining old TLA workloads and a bounded layer-rich witness.
Explain changed rule digests; don't count fewer choices as speedup. Keep debug
correctness oracles and follow AGENTS.md and docs/development.md checks. No
artificial load or speculative state-storage rewrite. Bugs are expected; focus
on generally useful training and preventing invalidated learning conclusions.
