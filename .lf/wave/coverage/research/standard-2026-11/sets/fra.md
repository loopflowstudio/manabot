# FRA: loyalty authority, prepared identities, and event results

Research date: 2026-11-04. Read all **285 identities**, including every face
of **21 preparation cards**: **306 face entries** total. The
[ledger](fra.jsonl) matches `.cache/standard-research/fra.json` exactly by Oracle
ID, full name and ordered face names. The inventory contains 264 normal layouts
and 21 prepare layouts; shared prepare spell names do not merge identities.
Five exact definition-name matches are already present, all basics in `lea`.
Reprints such as Tarmogoyf and Chandra, Torch of Defiance still belong in their
first-printing modules. This review covers FRA membership, not adjacent FRC/SPG
products or a certification of Standard legality. No games or benchmarks ran.

## Existing foundation and actual gaps

Reuse `ObjectSelector`, continuous records, target clauses, exact-object cast
grants, flashback, optional payments, reflexive triggers, copy/retargeting and
remembered delayed subjects. `PutCountersFrom` already expresses Graft Surgeon's
counter inheritance; exhaust and finality already exist. Ruric Thar, Biomagus
needs two ordinary prowess trigger instances, not a duplicated keyword bit.
Existing `OnceEachTurn`, `OnNthResolutionThisTurn` and `MayOnceEachTurn` have
different meanings and should remain distinct.

The [activated ability definition](../../../../../../managym/state/src/card.rs)
has no loyalty cost or general scoped loyalty grant. Planeswalker card types,
loyalty counters, damage removal and the
[zero-loyalty SBA](../../../../../../managym/src/flow/sba.rs) exist, but do not provide
planeswalker execution. [ManaUse](../../../../../../managym/state/src/mana.rs) currently
restricts by three named subtypes, not card type or cast origin; mana-choice
recognition assumes fixed output. Existing `Surveil` preserves kept-card order,
which needs expansion for Yuriko's surveil two. The current replacement pipeline
explicitly documents deterministic source order rather than affected-player
ordering choices.

## Shared candidate batches

Counts are primary ownership, not independent engine implementations. Ledger
`implementation_kind` classifies the principal obligation; dependencies may
require additional contracts. “Existing composition” means a candidate using
present primitives, not verified complete execution or an implemented card.

| Primary batch | Cards | Shared owner and witnesses |
|---|---:|---|
| Existing composition | 93 | Card definitions, selectors, layers and existing continuations |
| Resumable results and choices | 34 | Resolution frames: Confidant, Seasoned Cryomancer, Extrapolate |
| Event history | 40 | Event authority: Command the Stage, Danitha, Tinybones |
| Casting, payment and permissions | 29 | Cast transaction/mana solver: Emrakul, Loot, Uldaros |
| Loyalty, tokens and emblems | 40 | Activation, characteristics, token entry and SBA authorities |
| Preparation | 24 | SOS persistent-copy contract: 21 cards and three consumers |
| Characteristics and identity | 13 | Derived view/combat: Tarmogoyf, Loot, Hall of Echoes |
| Replacements, restrictions and procedure | 12 | Proposed-event/priority authorities: Karn, Tomik, Draconic Visitor |

**1. Build loyalty once.** Typed signed loyalty costs, X choices, per-permanent
activation limits and timing permissions belong to ordinary activation/payment.
Scoped granted abilities flow through `Characteristics::activated_abilities`,
retaining grantor identity when stacked. A mana-producing loyalty ability still
uses the stack. Way of the Warlord additionally requires independent target
clauses on activated abilities, whose current accessor assumes one requirement.
Emblems need persistent command-zone sources, not immortal battlefield cards.

Empower creates a zero-loyalty Jace token only when none qualifies, otherwise
chooses an existing token, including an eligible copied Jace. Counter placement
finishes before SBA checks. Jace's Machinations changes timing without granting
another activation; Sanctum Lurker suppresses only zero-loyalty death. These
boundaries are clarified by the
[official FRA release notes](https://magic.wizards.com/en/news/feature/reality-fracture-release-notes).
Generalize creation/selection plus counter placement, not forty card handlers.

**2. Reuse SOS preparation.** The [SOS review](sos.md) owns linked persistent
spell copies, permission following current control, and base/alternative
characteristics. FRA adds explicit unpreparing, repeated prepare-spell programs,
and preparing after attack-history checks. Immutable programs may be shared;
castable copy identity may not. Infinite Coursework's explicit unprepare must
remain distinct from losing abilities. Codie's copy trigger reuses existing
copying and retargeting. Named prepare spells must enter naming queries without
becoming independent deck entries.

**3. Retain instruction results.** Capture actual moved/discarded/drawn objects,
counts and LKI in typed frame bindings. Confidant and Chandra recover only their
surveilled card; Cryomancer targets after discarding; Overwrite counts actual
exiles; Vindictive Triumph binds its delayed exile to the returned incarnation.
Extrapolate extends existing outside-game candidates with exactly two distinct
names, public reveal and opponent choice. Arc of Fortune and Garruk require
resumable per-player decisions. Tam requires repeated proliferations, each with
its own complete selection, not a multiplied counter increment.

**4. Extend payment and permission predicates.** Reuse the witness-producing
mana solver for convoke, monocolored hybrid, behold, nonmana ward payments and
compound flashback costs. Add spell-type/origin restrictions and variable output
for Loot's distinct-power mana. Uldaros needs shared card-copy casting with an
aggregate mana-value budget and per-type target constraints. Emrakul's granted
mana lasts until the specific exile cast, remains usable during that payment,
and does not expire merely because the exiled card moves elsewhere; the
[release notes](https://magic.wizards.com/en/news/feature/reality-fracture-release-notes)
make this a cast-event duration, not ordinary source-presence expiry.

**5. Share event facts and rules exceptions.** Count gross life gained,
actual library-to-graveyard moves, scry/surveil instructions, batch discards,
loyalty activations/costs and prior-turn noncombat damage at their event owner.
Retain exact-incarnation lifetime history for Ruric Thar's first combat damage.
Distinguish first qualifying cast from first cast overall, and attacking one
player alone from being the only attacker across every destination.

Signed power, distinct type/color/power counts and floor division belong to
shared values and contextual characteristics. Loot's combat assignment does
not rewrite power; Ghalta's toughness assignment shares combat authority.
Hall of Echoes needs a player-scoped legend exception. Karn suppresses entry
triggers, not replacement effects. Split second and Yuriko's combat prohibition
must constrain both normal priority and nested casts. Draconic Visitor, Yoshimaru
and Tomik require real replacement selection, without source-order shortcuts.

## Closure and acceptance

Close 16 linked token kinds: Jace, Cadet, Heartwood, Illusion, Beast, Dragon,
Forest Tentacle, Lotus, Ajani's Pridemate, Angel, Thopter, Mowu, Leviathan,
Sculpture Treasure, Treasure and dynamic Copy. Close three distinct persistent
emblems: Ajani's team buff and the two Chandras' draw/damage triggers. Tokens
retain colors, subtypes, legendary status, granted abilities and summoning
sickness; Heartwood and Lotus produce chosen colors through one mana ability.

Use bounded adversarial witnesses: Jace creation with counter replacement;
loyalty activation followed by control change; prepare/copy/unprepare;
replacement during retained-result selection; duplicate prowess; simultaneous
Karn entry; split-second nested cast; signed-power combat; and Fblthp winning
before deferred empty-draw loss. Project all new state and decisions through
observations, native/Python tensor parity, text, learning tokens and replay/fork
identity without silently changing frozen versions.

Index event subscriptions, retain compact histories and immutable programs,
and invalidate characteristic caches precisely. Avoid scanning all history or
expanding every proliferation subset eagerly. Variable mana and additional
loyalty choices need bounded ordinary-step and search-fork workloads using
existing paired Engine Budget digests and cumulative cost tracking. No speedup
is established here. Numeric capacities, replacement ordering, persistent
sources and complex cross-mechanic interactions remain medium-confidence
planning claims until independent directed proofs exist.
