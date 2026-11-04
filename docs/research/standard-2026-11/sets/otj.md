# Outlaws of Thunder Junction — independent set analysis

Reviewed 2026-11-04: all 276 cached identities, full Oracle text and face metadata,
including basics, reprints and Jace Reawakened. The input contains no multiface
cards. [The ledger](otj.jsonl) assigns each identity exactly once; dependency
lists intentionally cross batch boundaries. OTJ rotates under the project's
working early-2027 boundary. BIG, OTP, OTC and SPG are not silently included.
The cached Scryfall inventory is transcription evidence, not a legality or
correctness certificate.

A literal definition-name scan found only the five basics already present;
these reuse `lea` definitions. New reprints belong in their first-printing
modules. The other rows describe authoring and engine obligations, not claims
that all 271 need new machinery. No games, benchmarks or production changes
were performed.

## Foundations and actual gaps

The inspected authorities are `state/src/ability.rs`, `card.rs`, `predicate.rs`,
`effect.rs`, and `src/flow/{turn,triggers,decision,play}.rs`. Reuse selectors,
values, `TargetClause`, `WhenYouDo`, `CreateDelayedTrigger`, linked exile,
spell-copy/retargeting, `OnceEachTurn`, activation limits and characteristic
caches. `OnNthResolutionThisTurn` and `MayOnceEachTurn` already distinguish
resolution counts and accepted optional uses; neither substitutes for a
once-per-turn trigger or a second-spell event. Outlaw is one union predicate
across five existing subtype strings, not a new creature type or five handlers.
Desert and Mount similarly use the subtype vocabulary.

Important extensions are narrower than a new card language, but several change
shared execution contracts. `AdditionalCost` currently has no priced mode list;
activations expose at most one target requirement; `EffectDuration` lacks an
attachment-dependent lifetime; copy records lack temporary reversion;
`mana_choices()` recognizes fixed-one output forms. Existing surveil preserves
the order of retained cards, explicitly justified only for surveil one. OTJ's
surveil two/three exposes that shortcut. These are shared repairs/extensions,
not reasons to invent OTJ-specific executors.

## Ordered candidate batches

The order below sequences shared work; high-priority surviving sets should own
it where they already need the same mechanism. Counts are primary assignments,
not independent implementation estimates.

| Batch | Cards | Scope and acceptance anchor |
|---|---:|---|
| OTJ-01 definitions/composition | 88 | Ordinary effects, subtypes, target clauses, token closure and nth-cast triggers. Ruthless Lawbringer composes sacrifice with `WhenYouDo`; This Town Ain't Big Enough reuses target-dependent reduction. Local vocabulary gaps remain explicit in dependencies. |
| OTJ-02 events/values/provenance | 40 | Typed arithmetic/count queries and event receipts: cast origin, paid mana, creature entry/death, modal choices, damage history. Steer Clear snapshots Mount control at casting; Satoru and Freestrider Commando distinguish zero mana spent from a free alternative cost. |
| OTJ-03 plot/permissions | 37 | Special-action payment and face-up exile permission, later-turn gate, origin-sensitive discounts, external plotting and plotted events. Fblthp extends the same operation to private library top; Lilah intercepts only resolution departure. |
| OTJ-04 spree/casting | 21 | Nonempty distinct priced mode subsets, announced before targets and payment; copy retains modes. Lively Dirge additionally needs resolution-time aggregate graveyard selection; One Last Job needs entering attached. |
| OTJ-05 crime/history | 24 | One event authority after cast/activation/trigger placement completes, plus per-player history and indexed listeners. Forsaken Miner registers a graveyard listener; Kaervek copies and casts a card from exile. |
| OTJ-06 saddle/contributors | 18 | Reuse crew's power-based tap payment; add saddle state and exact contributor relations shared with crew. Fortune, Gitrog and Calamity consume those relations; Luxurious Locomotive requires crew history. |
| OTJ-07 copy/grant/duration | 15 | General temporary and entry copy operations, attachment lifetimes, ownership predicates, dynamic activated grants, batch token copies and token CDAs. Assimilation Aegis, Eriette and Vraska expose distinct layer/lifetime obligations. |
| OTJ-08 choices/costs/replacements | 28 | Ordered private selections, optional alternate payments, simultaneous selected moves, generic searches, composite ward/life taxes, counter/damage replacements and executable variable mana output. Another Round, Selvala and Taii Wakeen are boundary witnesses. |
| OTJ-09 combat/scheduling | 5 | Attack/block taxation, cast limit, protection, blocker constraints and extra upkeep steps. Obeka extends the existing schedule; Archangel of Tithes needs legal declaration payment. |

## Rules and closure

Official checks confirm: plotting is a special action, later plotted casting
still pays additional costs, saddle state is not copied or erased by ability
loss, and crime depends on initial targets rather than retargeting. Lively
Dirge's return mode does not target; Satoru's zero-spend condition fails when a
free cast pays a tax. These checks constrain the proposed authorities above.
[Wizards release notes](https://magic.wizards.com/en/news/feature/outlaws-of-thunder-junction-release-notes)

Close token definitions for Mercenary, Zombie Rogue, Sheep, Ox, Angel, Bird,
Dinosaur, Elemental, Elk, Scorpion Dragon, Spirit, Vampire Rogue, Varmint,
Meteorite and Beau; reuse matching Clue/Treasure definitions. “Copy” is a
placeholder for runtime copiable values, not a predefined generic creature.
The current `token.spirit` is a 1/1 with Spirit-only combat restrictions,
whereas OTJ requests a 2/2 white flying Spirit. Same printed token name cannot
establish identity. Beau needs land-count power **and toughness** CDA; Elemental
needs snapshotted copiable P/T; Meteorite owns targeted entry damage and mana
production. Token identity must preserve these distinctions without changing
historical token definitions. Copies can import both faces from other sets even
though OTJ itself has no multiface inventory.

All newly relevant state needs viewer-safe semantic observations, native/Python
tensor parity, LLM text and learning tokens: plot ownership/earliest cast turn,
crime and cast history, saddle state/contributors, paid costs, mode provenance,
copy expiry and scheduled upkeeps. Exact object joins must survive suspended
choices and search clones. Do not expose Fblthp's library top, surveil choices,
or opponents' hidden hands through public history. Frozen versions remain
frozen; any inadequate capacity/semantic admission needs an explicit contract.

## Adversarial witnesses and performance

Use bounded constructed positions through ordinary Commands:

- Final Showdown removes abilities, then grants indestructible before destruction;
  countered/all-illegal spree spells do not partially execute target-free modes.
- Cast a plotted spell with Aven's tax; confirm actual mana provenance, unchanged
  mana value, and no second alternative cost. Plotting itself increments no cast count.
- Retarget a spell onto an opponent without creating crime; place Ruthless
  Lawbringer's reflexive trigger on an opponent permanent and create one crime.
  Kaervek's own-graveyard target is not a crime, but its newly cast copy may be.
- Counter a saddle activation after contributors tap; blink a contributor;
  repeat Calamity's choice; preserve saddle after ability loss and clear it on
  departure. Gitrog uses sacrificed contributor LKI, not a surviving lookup.
- Detach/re-attach Aegis, copy an already copied token, and remove Oko's creature
  target before resolution. Never follow a card into a new incarnation.
- Give Surveil three retained cards needing reordering; choose two Lively Dirge
  creatures within a total cap; distinguish variable Selvala output from an
  ordinary fixed mana source. Snapshot/clone each pending choice and reject stale commands.

Avoid rescanning all cards for crime or turn history: append compact typed
receipts at existing event authorities, index listeners, and cache derived
counts by relevant revisions. Contributor choices must remain identity-complete;
do not enumerate every subset upfront or collapse strategically different
creatures. Share immutable copy payloads while preserving distinct effect edges.
Generalize the scheduler without polling inactive events. Measure future work
on bounded ordinary-step and search-reply fixtures, including clone cost,
legal-action count and allocation growth; retain exact-result digests and
same-runner Engine Budget comparisons, plus cumulative milestone cost. Debug
correctness precedes measurements. No performance improvement is claimed here.
