# HOB — The Hobbit

Research date: 2026-11-04. Priority: survives the next rotation. This is a bounded
full-text planning pass, not implementation or conformance certification. Read
all **193 identities / 210 faces**, including 17 Adventures and eight Sagas, from
`.cache/standard-research/hob.json`; [hob.jsonl](hob.jsonl) assigns every identity
once. Ledger confidence describes confidence in the proposed obligation: `high`
means familiar composition, `medium` requires extension or interaction proof.
It does not mean the card currently works.

## Sources and inventory

The cached Scryfall transcription supplies card-specific text and identities.
[Wizards' release notes](https://magic.wizards.com/en/news/feature/the-hobbit-release-notes)
resolve several material ambiguities: storied requires a storied permanent,
counts distinct permanents and can establish status before a legend-rule loss;
hone works through ability loss; amass chooses one Army even after token
doubling; Bilbo's Adventure permission depends on which exile replacement
applied. Thranduil's acquired linked abilities lose their link when lost and
regained. These are explicit acceptance cases, not inferred shortcuts.

The notes also cover adjacent products. Reconciliation queried Scryfall on this
date with `e:hob`, `include_extras=true`, and both `unique=cards` and
`unique=prints`: **193 identities; 321 printings; identical Oracle-ID set** to
the input. [Chief of the Wilds](https://scryfall.com/card/hoc/103/chief-of-the-wilds),
[Dáin of the Ancient Halls](https://scryfall.com/card/hoc/104/d%C3%A1in-of-the-ancient-halls),
[Thorin, Company's Leader](https://scryfall.com/card/hoc/100/thorin-companys-leader)
and [Bag End Banquet](https://scryfall.com/card/hoc/5/bag-end-banquet) are HOC and
reported not Standard legal; misleading HOB image labels in the notes are not
set membership. No supplemental HOB identities were found.

The [Hobbit update bulletin](https://magic.wizards.com/en/news/announcements/the-hobbit-update-bulletin)
also changes mana-ability classification when an ability moves cards to/from a
library. Cross-set payment work must use the current rule, not extrapolate from
older mana helpers.

## Existing support and ordered shared batches

Source inspection of `managym/cards/src` found exact-name registrations only for
the five basics, owned by `lea.rs`; there is no HOB module. This is a registration
inventory, not a runtime test. Reuse first-printing identities for reprints such
as Wood Elves and Settle the Wreckage. Existing `TargetClause`, delayed watchers,
`DiscardedThisWay`, `OnNthResolutionThisTurn`, crew, scry, copy exceptions,
flashback and layered characteristics are useful foundations. The
[current audit](../current-engine-audit.md) supplies the constraints below.

1. **`existing-composition` — definitions and directed coverage.** Ordinary
   removal, combat grants, counters and tokens need no new card-specific
   executor. Thorin's Last Stand and Warg Tactics exercise modal targeting;
   Troll Negotiations exercises changed power before fight. Zero optional
   targets and an illegal chosen sole target must remain different outcomes.

2. **`scoped-results` — resolution authority.** Recruit mostly composes existing
   draw/discard predicates; retain the discarded object's result across nested
   replacement choices. Amass adds a reusable semantic instruction with Army
   choice, token creation, counters, subtype addition and an explicit result.
   Goblin Plate Mail consumes that Army result, whereas Dwarven Shortsword
   consumes created-token results. Bolg needs sacrifice LKI, a reflexive trigger,
   damage and excess results without overwriting one another. The Eagles Are
   Coming! carries an actual returned count into a next-upkeep watcher.
   Replace non-resumable `ForEachTarget` and migrate duplicated loot/mill-return
   continuations when this shared context becomes authoritative.

3. **`selection-visibility` — inspected sets and destinations.** Troop of Ponies
   partitions searched basics between hand and battlefield; Old Thrush shuffles
   before putting the selected card on top. Riddles needs labeled piles and
   different viewer permissions; Getaway Barrel requires random eligible-card
   selection and random remainder order. Through the Forest Gate selects from
   twenty without enumerating every subset. Generalize existing search and
   `LookAndSelect` ownership, then delete redundant destination-specific paths.
   Uneasy Partings must not reuse the current second-from-top helper unchanged.

4. **`faces-permissions` — casting authority.** Seventeen Adventures need
   alternative characteristics, exact exile permission and correct stack-exit
   provenance. Consolidate with existing linked/impulse/flashback permissions,
   not a second Adventure casting implementation. Bilbo, Thief in the Night;
   Inside Information; Flameshape; and Thranduil's Decree separately require
   resolution-time casting, life costs, private conditional exile access and
   persistent free casting. Bilbo's Gambit adds announced gift provenance and
   a symmetric casting prohibition.

5. **`event-history` — committed events and status.** Enduring story is durable
   player state; Belladonna counts resolutions; Gollum remembers chosen modes
   per incarnation; second-draw and first-cast readers use player history.
   Desert Were-Worm schedules combat from the first qualifying attack.
   Supper for Spiders requires exact graveyard-arrival provenance. Extend the
   event authority and typed bounded summaries rather than replay-scanning logs
   or putting unrelated booleans on individual card handlers.

6. **`replacement-events` — pending proposals.** Bard replaces draws and token
   batches; Head of the Hunt replaces deaths and produces reflexive occurrences.
   They expose the audited fixed-order processor's architectural limit. Replace
   it with resumable affected-player ordering, applicability rechecks and
   committed-event identity; retain the fast no-replacement route.

7. **`payment-provenance` — cost and mana witnesses.** Woodland Weavemaster and
   Desolation require Elf/Dragon restrictions beyond the current three-subtype
   mask. Smaug, Wicked Worm needs actual Treasure-origin mana consumption;
   Uncover the Moon-Letters needs actual amount spent. Kíli, Dwarven Mauler,
   Dáin's attack tax and composite equip costs require shared cost calculation,
   announcement locks and joint resource reservation. Do not add ad hoc subtype
   bits or infer mana spending from sacrifice history.

8. **`layers-attachments` — one characteristics authority.** Hone belongs to
   counters; attachment changes immediately change its contribution. Supper and
   Tom change types while retaining abilities. Thranduil dynamically acquires
   graveyard activated abilities, requiring source rebinding and link lifetime.
   Extend existing grants and revisions, retiring any independent printed-only
   ability reader. Wizard's Staff/Bifur reuse extra-occurrence machinery with
   proper attached-source/conditional scope.

## Objects, observation and bounded acceptance

Inventory real token definitions for Treasure, Human Soldier, Dwarf, flying Bird
Soldier, Axe, Goblin Army, Wolf, flying Dragon, Stone Boulder, Bear and Elf.
Notary copies require nonlegendary exceptions; Supper's Foods are transformed
cards, not substitute Food tokens. Share Adventure work with WOE, gift with BLB,
crew with DFT, and results/replacements/layers across the milestone.

New status, mana provenance, granted abilities, pile permissions and suspended
results must reach semantic capture, native/Python policy inputs, text and
learning tokens together. Preserve frozen schemas, exact Commands, reconnects
and full-clone isolation. Use bounded witnesses: Bard plus amass and Plate Mail;
Bolg with source removal; Bilbo resolving versus countered Adventure; Riddles
with empty/private piles; Thranduil losing/regaining grants; and Treasure versus
ordinary mana paying the same spell. Resume a later choice-producing spell to
expose abandoned frames. Follow debug and package validation requirements.

Measure ordinary step and search replies on these small fixtures, retaining
same-runner digests and cumulative milestone anchor. Watch choice cardinality,
frame cloning, graveyard-dependent cache invalidation and token-batch observer
work. No games, tests or benchmarks were run in this research; no speed or
complete-support claim follows from the ledger.
