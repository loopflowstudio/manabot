# SOS: casting facts, persistent spell identities, and shared choices

Research date: 2026-11-04; checkout `5dd11cec4`. Read all **271 identities**
and every face in `.cache/standard-research/sos.json`: 235 normal cards and
36 preparation cards, totaling **307 face names**. The [ledger](sos.jsonl)
contains one checked Oracle-ID/name row per input identity. This is a planning
review, not execution certification; no games or benchmarks ran. Exact front-name
matching finds only the five basic lands already defined. Keep those in `lea`;
reprints belong in their actual first-printing modules. SOS membership does not
admit adjacent SOC, SOA or SPG products, nor make prepare spells independent
deck entries.

## Reuse before extension

The existing native rules vocabulary already covers selectors, continuous
records, ordinary flashback, exact-object exile permissions, graveyard abilities,
intervening conditions, optional payments, reflexive triggers and delayed subjects.
`CopyTriggeringObject` already copies and retargets stack objects; it is not a
missing feature. Reuse those owners, including `OnNthResolutionThisTurn` and
`MayOnceEachTurn`, rather than replacing TLA's work with SOS-only interpreters.

The important gaps are narrower. [SpellOnStack](../../../../managym/state/src/stack_object.rs)
stores origin, flashback, kicked, X and one mode, but no immutable receipt of
actual mana paid. Existing [cast grants](../../../../managym/src/flow/cast_permission.rs)
bind a real exiled object to a fixed player; preparation needs a persistent
spell copy and a permission following a permanent's current controller.
[Ability/effect types](../../../../managym/state/src/ability.rs) provide the
composition points, but ordered selections, repeated modes, new history queries
and copy lifetimes need extensions.

## Candidate batches

Counts designate primary ownership, not independent implementations or guaranteed
definition-only work. Dependencies cross these batches; implement shared
authorities before their consumers, then close the complex combinations.

| Batch | Cards | Shared work and representative witnesses |
|---|---:|---|
| Existing composition | 60 | Ordinary spells, activated costs, flashback, dual lands; isolate remaining value-expression gaps |
| Resumable results and choices | 44 | Flow State, Pox Plague, Moment of Reckoning, Zimone's Experiment |
| Casting and payment | 44 | Converge, opus, increment; Mana Sculpt, Molten Note, Wildgrowth Archaic |
| Event history | 44 | Repartee, infusion, grave departures, counter and life events |
| Casting copies and permissions | 21 | Mica, Nita, five elder-dragon grants, Zaffai |
| Preparation | 38 | 36 preparation cards plus Tomekeeper and Waypoint |
| Paradigm | 5 | Recurring resolved spell snapshots, including Improvisation Capstone |
| Characteristics and identity | 12 | Fractalize, Applied Geometry, counter inheritance, Resonating Lute |
| Procedure and naming | 3 | Ral, Dellian's emblem, Petrified Hamlet |

**1. Make choices compositional.** Generalize existing continuations around
private candidate sets, cardinality, retained ordering, destination and actual
result bindings. Flow State and surveil two need ordered retained cards;
Zimone's Experiment needs typed destinations and random remainder. Mind Roots
may recover only a land actually discarded by its preceding instruction.
Pox requires suspended choices within simultaneous player batches. Moment of
Reckoning requires zero to four mode occurrences with repetitions and separate
targets. Retire migrated card-shaped selection continuations and the assumption
that iteration children cannot suspend. Preserve deliberate finish choices,
including excess crew contributors for Strixhaven Skycoach.

**2. Record payment once.** The casting transaction should retain total mana,
paid colors, announced X, chosen alternative/additional costs and origin.
Converge, opus and increment then become predicates/values over the same receipt.
Mana Sculpt retains another spell's receipt after countering it. Wildgrowth
Archaic binds an entry modification to the particular creature spell.
Generalize `ManaUse` restrictions to spell predicates for Hydro-Channeler and
Resonating Lute; retain the existing mana solver. Cost reduction, free casting,
life payment, copying and multiple X symbols must remain distinct. Checked
value composition handles Procrastinate and Mathemagics without card-specific
arithmetic handlers.

**3. Extend typed history, not keyword handlers.** Repartee queries targets at
cast time; infusion queries life gained, not net life; grave-departure triggers
distinguish one batch from several events. Existing nth-cast/draw and resolution
history should share event ownership, while preserving their different counting
rules. Increment uses an intervening condition, not a permanently captured
boolean. A life-history change affecting static characteristics must invalidate
the correct cache revision. Do not add individual flags for every beneficiary.

**4. Separate castable objects, permissions and copies.** Expand existing copying
and retargeting for explicit targets, creature-spell copies, copy prohibition,
and multiple copied instances. Compose storm, cascade, casualty, miracle and
affinity from their distinct trigger, selection and cost parts; they are not one
generic “cast free” effect. Zaffai can reuse accepted-versus-declined quota
semantics, while Nita needs exact ownership and destination replacement.

Preparation creates a linked exile copy whose permission follows current
control. Ability loss, type change and subsequent copying do not erase an
existing prepared spell; preparation itself is not copied. Paradigm's recurring
trigger survives removal of the original exiled card. These are explicit
requirements in the [official release notes](https://magic.wizards.com/en/news/feature/secrets-of-strixhaven-release-notes).
Use that distinction to extend object lifetime authority; do not bolt another
temporary boolean onto the ordinary cast grant. Paradigm can reuse resolution
counting concepts, but its first-name, game-long recurrence is not the current
per-turn counter. Resolve exact name-history and snapshot rules against current
comprehensive rules during implementation.

**5. Preserve one characteristics and scheduling authority.** Fractalize and
Applied Geometry need correct color/type/base-stat layers and copy exceptions.
Ambitious Augmenter and Scolding Administrator transfer complete counter multisets
from LKI. Root Manipulation and the Great Hall grant triggers through the shared
derived view. Ral's coin results schedule skipped turns; Dellian's emblem outlives
its planeswalker. Petrified Hamlet shares naming infrastructure with The Rise
of Sozin, but applies its prohibition to named sources beyond battlefield lands.
Do not create another catalog-sized owned modal program or silently shortlist
legal names.

## Closure, acceptance and performance

Close six concrete token kinds: red-white Spirit, flying white-black Inkling,
flying blue-red Elemental, green-blue zero-base Fractal, attack-life Pest and
Treasure. SOS's Pest is not the older death-life Pest. The linked Copy marker
represents dynamic copiable values, not a fixed token program. Also account for
Dellian's persistent emblem, prepared spell copies and paradigm snapshots.

Use small adversarial witnesses: different-color payment of identical generic
costs; copied versus cast spells; increment changing which stat satisfies its
condition; prepare/control transfer/ability loss/blink; repeated mode targets
becoming illegal; replacement interruption inside Pox; counter LKI after source
death; and declined paradigm casting followed by the next first main phase.
Project payment facts, prepared-copy joins, permissions, histories, emblem and
pending choices through native observations, Python/tensor parity, text, tokens,
replay/fork and Academy without widening frozen versions silently.

Cache immutable programs and candidate identities; index event subscribers;
store compact payment receipts instead of rescanning past actions. Measure both
ordinary steps and search forks: persistent copies, histories and wide choices
can make cloning or legal-action generation expensive. Retain existing paired
Engine Budget workloads and add bounded representative fixtures as mechanics
land, with exact decision digests and cumulative milestone accounting. This
review establishes no performance improvement. Low confidence remains around
new persistent-object semantics, numeric capacity, naming universe and complex
replacement interactions until directed rules proofs exist.
