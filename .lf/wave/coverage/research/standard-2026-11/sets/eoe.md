# Edge of Eternities: shared-engine implementation plan

Research date: 2026-11-04; checkout `5dd11cec4`. All **266** input identities
were read in full and individually assigned in [eoe.jsonl](eoe.jsonl). This is
planning evidence, not implementation or conformance certification. No games,
benchmarks, builds or runtime changes were made.

## Inventory and existing support

The input is `.cache/standard-research/eoe.json`, a paginated Scryfall
`e:eoe`, `unique=cards` snapshot with source metadata. It contains no multifaced
cards. The [official release notes](https://magic.wizards.com/en/news/feature/edge-of-eternities-release-notes)
were checked for station, warp and difficult card interactions. EOE remains in
the initial post-rotation priority. Commander and Stellar Sights inventories
must not be imported merely because their products accompany EOE; format
membership needs its own reprint-aware reconciliation.

A textual exact-name scan of the current card modules finds only the five basic
lands. Preserve those Alpha definitions; the other 261 names are authoring
candidates, not 261 new engine operations. Existing composition can express
many ordinary effects, while the difficult cards expose common infrastructure
that other sets also need.

## Batches and dependency order

| Primary batch | Cards | Representative requirements |
|---|---:|---|
| Context/history/values | 51 | Void, second cast versus second resolution, actual mana spent, distinct land names, damage history |
| Existing composition | 49 | Honor, Annul, Bombard, removal unions, ordinary triggers and keywords |
| Warp/play permissions | 35 | Alternative casting, provenance, exact delayed exile, later-turn grants, granted warp |
| Costs/mana | 31 | Convoke, composite sacrifice/tap/counter costs, restricted spending, conditional reductions |
| Station/levels | 28 | Charge counters, resolution-time contribution, conditional abilities and characteristics, Planets |
| Layered grants/durations | 19 | Temporary ability grants, copy exceptions, controller-relative expiry, attachments |
| Exact objects/results | 19 | Counter LKI, linked exile, simultaneous moves, retained controller/characteristics |
| Resumable selections | 17 | Dynamic surveil, ordered private selection, modal cardinality, opponent choices |
| Replacements | 14 | Entry choices, draw/token/counter modification, prevention prohibition |
| Player procedure | 3 | Planeswalker/emblem, opponent-turn control, poison loss |

These are primary ownership queues; dependencies cross rows. Start with a small
ordinary-card slice and contextual read/history contracts, then payment and
selection, warp and station, and finally replacement/lifetime combinations.
Procedure cards have explicit separate dependencies. Do not author an entire
easy tier before exposing the difficult invariants with real cards.

## Highest-value shared mechanisms

**Station is not crew with a different threshold.** A station activation taps
one other creature, then its resolving ability reads that creature's power or
LKI. Further activations remain legal above the final level, and a Planet need
never become a creature. Share exact-object tap costs and value readers with
crew/saddle, but preserve their different timing. Tapestry Warden substitutes
toughness only under its specified comparison. Charge-level abilities belong in
the existing characteristics authority, not duplicated `is_stationed` fields.
The current `CounterKind` lacks charge; append vocabulary and extend all shared
projections. A level dropping below threshold must remove its derived grants.

**Warp composes costs, provenance and permissions.**
[`ExileCastGrant`](../../../../../../managym/src/flow/cast_permission.rs) already has
exact object identity and an `after_turn` term. Reuse them; do not create a
parallel warp-card store. Extend cast metadata, delayed effects and alternative
costs together. Timeline Culler adds graveyard access and life payment; Tannuk
grants warp to a class of hand cards; enchantments demonstrate that warp is not
creature-only. Astelli Reclaimer, Unravel and Dyadrine require actual mana spent,
which current [`SpellOnStack`](../../../../../../managym/state/src/stack_object.rs)
does not retain. Announced X, nominal cost and paid mana are different values.

**History must describe events precisely.** Void observes nonland battlefield
departure or a warped cast. Tannuk, Memorial Ensign counts resolution of its own
ability, while Terrasymbiosis limits acceptance of an optional effect. Neither
is ordinary once-per-turn triggering. Model scoped event/ability facts rather
than one new player field per card. Haliya measures life gained, Faller's
Faithful asks about damage to the target, and artifact-creature unions count
each object once. These distinctions also determine cache invalidation.

Cross-set reconciliation: TLA already has `OnNthResolutionThisTurn` (South
Pole Voyager) and `MayOnceEachTurn` (Earth Kingdom General). Tannuk and
Terrasymbiosis should first reuse and validate these existing owners; their
different histories do not by themselves justify new machinery.

**Unify typed costs and selection continuations.** Steelswarm Operator restricts
mana by spell type or activation-source type; Mm'menon additionally restricts
casting origin. The three-subtype bit mask in
[`ManaUse`](../../../../../../managym/state/src/mana.rs) cannot represent these terms.
Gene Pollinator requires a second tap contribution; Secluded Starforge binds X
to tapped artifacts; Dyadrine selects two counter donors. Generalize contribution
selection and reservations without greedy choice pruning. Cerebral Download and
Starbreach Whale need multi-card surveil ordering. Migrate the restricted
surveil continuation and repeated search/sacrifice composites into shared
selection/result machinery instead of adding per-card branches.

**Preserve exact results through layers and replacement.** Ouroboroid samples
source power once; it must not increase its own amount midway through a counter
batch. Dockworker Drone and Broodguard Elite need counter-kind/count LKI.
Pinnacle Starcage counts actual linked-card movements. Moonlit Meditation,
Quantum Riddler, Loading Zone and Famished Worldsire require pending replacement
choices and renewed applicability checks, not fixed ordering. Mouth of the Storm
needs a fixed affected set lasting until the controller's next turn; current
continuous-effect durations do not express that boundary. Share it with other
sets' duration records rather than encoding another delayed reversal.

## Focused acceptance and performance

- Activate station with a zero-power, summoning-sick creature, then change its
  power or remove it before resolution. Verify LKI, optional further activations,
  Planet behavior and downward level transitions through Commands and forks.
- Warp under changed control, blink before the end step, counter the delayed
  trigger, and try casting from exile on the same versus later turn. Verify the
  correct owner and exact-object grant; unrelated exile must grant nothing.
- Compare nominal cost, X, discounts and actual spent mana with Astelli and
  Unravel. Exercise Gene Pollinator without reusing a reserved tapped permanent.
- Surveil three, keep two in reversed order, and reconnect or fork mid-choice
  without revealing the inspection to the opponent.
- Counter the first Tannuk trigger; resolve later ones. Decline Terrasymbiosis
  once, then accept later. Confirm the two histories remain distinct.
- Combine token-copy replacement and doubling, or enter Worldsire with a
  sacrifice choice. Keep one committed event stream and independent expected
  outcomes, then check tensor/text/token and Academy projections.

Measure existing TLA Search-64 and ordinary steps alongside a bounded EOE cohort
covering station, warp and artifact triggers. Profile action generation, payment,
layer invalidation, history queries and suspended-frame cloning. Reuse immutable
definitions and indexed event predicates; do not scan the entire card catalog
per step. Retain exact results for speed comparisons and separately disclose
changed-rule workloads. No performance improvement is claimed by this review.

## Object and procedure closure

Register Lander, Robot, Human Soldier, restricted-blocking Drone, Munitions and
Sliver token definitions; copy effects preserve arbitrary payloads. Lander has
a genuine composite mana/tap/sacrifice search ability. Tezzeret's emblem is a
persistent rule source after the planeswalker leaves. Virulent Silencer requires
player poison counters and loss checks. The Dominion Bracelet requires shared
opponent-turn decision control and visibility, including private choices; a
browser-only override cannot implement it. These are explicit prerequisites to
full EOE support, not reasons to silently omit difficult cards.

Validation: 266 unique ledger Oracle IDs and names exactly match the input;
zero missing, duplicate or unmatched identities. Card-specific planning
obligations cover every input row. All proposed new behavior remains untested.
