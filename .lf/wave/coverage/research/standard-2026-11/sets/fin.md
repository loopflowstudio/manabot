# Final Fantasy: implementation batches

Research date: 2026-11-04; source checkout `5dd11cec4100aa0e1f4a23c063dfd5778e2ba28f`.
Status: complete full-text planning pass, not rules-conformance certification.
Every one of the 312 fetched identities has a card-specific assessment in
[fin.jsonl](fin.jsonl), including all component faces. Source pins are in
[the inventory](../source-inventory.json).

## Inventory and existing support

The input contains 262 normal, 15 Saga, 27 transforming, five Adventure and
three meld records. **Ragnarok, Divine Deliverance is a meld-result record, not
an independent deck entry.** Fang and Vanille remain separate deck cards. The
input currently marks Vivi Ornitier banned; preserve its implementation in the
world while applying dated format legality separately. Thus 312 is not the
number of legal deck entries.

The [official release notes](https://magic.wizards.com/en/news/feature/final-fantasy-release-notes)
confirm FIN Starter Kit cards belong to the Standard product boundary. FIC
Commander and FCA bonus-sheet contents do not acquire that status merely by
sharing the release. They also establish that meld represents two physical
cards as one permanent, unlike ordinary transformation, and that departure
moves both cards. That needs explicit identity support rather than a fabricated
single-card back face.

[fin.rs](../../../../../../managym/cards/src/fin.rs) currently defines only Sazh's
Chocobo. Exact-name scanning finds it and five basics elsewhere: six matches,
not six independently revalidated implementations. Existing selectors, effect
values, optional payments, flashback, Saga chapters, transforms, finality/stun
counters and delayed triggers are useful starting points. Their presence does
not establish FIN's broader combinations.

## Ordered shared batches

The ledger assigns each identity one primary batch; dependencies cross batches.
These are shared-engine work packages, not separate FIN-only executors.

| Batch | Primary identities | Principal reusable work |
|---|---:|---|
| Core compositions | 25 | Existing verbs/keywords and ordinary tokens |
| Contextual queries | 24 | Captured values, unions, counts, dynamic characteristics |
| History and triggers | 31 | Paid-mana, turn/event ordinals, scoped trigger multiplication |
| Resumable choices | 48 | Ordered/private selections, simultaneous sets, result bindings |
| Costs and casting | 22 | Tiered costs, kicker, activation reductions and provenance |
| Mana | 21 | Spending predicates, production choices, granted mana abilities |
| Attachments | 35 | Job select, granted abilities and complete crew selections |
| Permissions | 29 | Cast/play from zones, duration, immediate casting, Adventure |
| Replacements | 16 | Entry/damage/life/mill/token/counter transformations |
| Faces and Sagas | 44 | Saga creatures, transforms, meld and contextual identity |
| Copies | 13 | Copiable exceptions, stack abilities, generated objects |
| Scheduling | 4 | Additional steps/combat, turn termination |

**1. Context and resumable execution first.** Ship straightforward definitions
alongside scoped values/results, not before their semantics are available.
Memories Returning alternates chooser over one revealed pool; Choco selects a
hand card, then a land subset; Kefka counts distinct types across both players'
discards; Sin repeats randomized exile/copy while the result is a land. These
should share resumable traversal and result bindings. Migrate overlapping
`MillAndSelect`, `ReturnMilledThisWay`, optional loot and search/move composites
when their contracts match. Delete migrated handlers, not their card tests.
Current `ForEachTarget` explicitly forbids suspension; it cannot safely host
these programs without the execution-context work proposed in the
[engine audit](../current-engine-audit.md).

**2. Casting and payment provenance.** Tiered cards require exactly one declared
additional-cost option, while flashback combines an alternate cost with ordinary
additional costs. Shantotto and Tellah read actual mana spent, not mana value;
From Father to Son and Retrieve the Esper read origin zone. Keep this data on the
cast/stack record and pass it through triggers, copies, observations and replay.
Cargo Ship and Freya require restrictions over spell type, activation-source
type and equip-ability identity; extend one typed spending predicate instead of
adding subtype bits. Existing once-per-turn mana support is relevant to Vivi,
but splitting a variable quantity across two colors needs its own witness review.

**3. Attachment and contribution composition.** Job select is token creation
followed by attachment to that exact created object. Most Job Equipment then
grants a subtype and ability to its bearer. Reuse `CreatedThisWay`, attachment
authority and layer grants; extend granted triggered abilities coherently with
existing activated grants. Astrologian's Planisphere and Ninja's Blades expose
bearer identity and retained event/result requirements. Zack needs departed
attachment LKI; Stolen Uniform needs a control-loss watcher. Crew must permit
every legal contribution set, not automatically finish at minimum power.

**4. Permissions and replacements together where required.** Hades, Noctis,
Quistis and Seifer combine zone permission, timing, spending adjustments and
departure/entry terms. Consolidate narrow cast-from-zone composites under these
axes. `PlayWindow` currently offers this turn and through-next-turn; Haste Magic
and Opera Love Song need the next-end-step boundary, and Lightning needs exact
source-control duration. Five land Adventures require alternative spell
characteristics and subsequent land-play permission, not transforming faces.

The Crystals, Quina, Traveling Chocobo and Ancient Adamantoise require real
event-scoped replacement or trigger modification. Ancient Adamantoise also
preserves marked damage across cleanup. A fixed preferred replacement order
cannot support their interactions; use the affected player's pending choice
with applicability re-evaluation. Do not encode these as permanent-specific
damage exceptions.

**5. Complete identities, copies and timing.** Reuse TLA Saga/transform machinery
for Summons and Dominants, while testing Saga creatures, finality, copied
transforming tokens and exile-return identity. Add meld as one permanent backed
by two physical cards; preserve both departures, mana value and copy exceptions.
Gogo requires stack-ability copying and a cannot-copy restriction. Sephiroth
requires a per-ability resolution count and an emblem outside battlefield life.
Y'shtola needs extra end steps; Ultima needs genuine end-turn cleanup. Generalize
the existing turn/combat schedule rather than add separate card-specific queues.

## Closure, acceptance and performance

Created-object closure includes Hero, Knight, Moogle, Bird, Wizard, Robot Warrior,
Horror, Frog, all-color Elemental, Food, Treasure, Angelo and Darkstar; also
Sephiroth's emblem and dynamic copies. Define each actual token once. A Copy
marker is not the copied object's executable definition. Cid's unlimited-copy
text belongs in deck construction policy; catalog eligibility remains unchanged.

Focused acceptance should cover:

- Tiered plus additional costs, paid-mana thresholds, and copied versus cast
  spells; show distinguishable policy choices before spending resources.
- Job token replacement followed by attachment, moving Equipment between
  bearers, and Zack leaving before his ability resolves.
- Alternating selection with reconnect/fork midway; Sin's RNG/result sequence
  must replay exactly without changing the source fork.
- Esper Origins through flashback and transformed entry; meld departure to an
  ordered library; copied meld and transforming permanents.
- Quina with another token replacement, Adamantoise with prevention/redirection,
  and multiple trigger multipliers; expose every required choice.
- Extra end step followed by normal cleanup, and Ultima with pending triggers,
  temporary effects and spells still on the stack.

Keep shared candidates/programs immutable where possible, but profile before
changing storage. Avoid catalog-wide enumeration for local choices, repeated
graveyard scans for every query, and eagerly expanding every subset of
Equipment, sacrifice or search candidates. Structured native decisions must
preserve the full legal space. Reuse revision caches with dependency review:
life total, attachment, damage, cast history and counters now affect more reads.
Add bounded FIN cases to same-runner Search-64 and ordinary step measurements;
retain the TLA comparison and exact-result digests. Jumbo Cactuar/Overkill also
need numeric-range and observation checks, not artificially large load tests.

Validation: read all card/face text, inspected current source and official
release guidance; verified exactly 312 unique ledger IDs and matching names.
No builds, tests, games or performance measurements were run. Detailed rules
adjudication and consumer parity remain implementation acceptance work.
