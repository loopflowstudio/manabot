# MKM: shared execution contracts before card volume

Research dated 2026-11-04. All 279 input identities and 285 face records in
`.cache/standard-research/mkm.json` were read, including Kellan's Adventure,
five split cards, basics and the three supplemental MKM legends. The adjacent
[ledger](mkm.jsonl) assigns each identity once. MKM belongs to the working
rotate-next pool in the [inventory boundary](../README.md); this is reuse and
coverage research, not a legal-deck certificate or implementation commitment.
MKC, CLU and SPG are not silently included. Confidence describes the engineering
assessment, not a passing behavior test.

## Evidence and existing authority

[`cards/src/mkm.rs`](../../../../managym/cards/src/mkm.rs) registers only Proft's
Eidetic Memory; its directed tests cover entry draw, combat-time count, hand size
and legendary behavior. The five basics already belong to `lea`. Other simple
cards are definition candidates, not claims of existing coverage. Reprints
belong in their first-printing modules.

Read authorities include `state/src/{ability,card,effect,predicate}.rs`,
`flow/{decision,resolution,proposed_event}.rs`, and `cards/src/tokens.rs`.
Reuse `ObjectSelector`, `EffectValue`, `TargetClause`, `WhenYouDo`, linked exile,
cast permissions, exact-object delayed triggers, trigger duplication, effect
records, and `Game::characteristics`. Clue and Surveil are already represented.
However, `Decision::Surveil` immediately moves each selected card and simply
resumes on decline: it offers no retained-top ordering. Surveil 2/3 needs a
shared correction, not another implementation. The existing token named Spirit
is colorless with unusual blocking restrictions; it cannot stand in for MKM's
white-black flying Spirit.

The [official release notes](https://magic.wizards.com/en/news/feature/murders-at-karlov-manor-release-notes)
were checked for ambiguous mechanics. Turning face up is a special action;
cloak and disguise differ after ability loss. Solved and suspected are not
copiable values. Cases check their solve condition twice. Conspiracy Unraveler's
alternative evidence payment does not satisfy an optional additional-evidence
bonus; paying both produces separate collection events. Urgent Necropsy's cost
locks after targets. Cryptex's activation is a mana ability. A Yarus-returned
card that cannot turn face up does not acquire cloak's ward. These distinctions
set acceptance boundaries below; the release notes' 2024 legality list does not
establish today's Standard pool.

## Ordered implementation batches

Batch IDs are ledger joins; execution order follows dependencies rather than
numeric ordering. Each count is a primary assignment, not exclusive ownership.

| Batch | Cards | Work and reuse |
|---|---:|---|
| 01 existing composition | 72 | Definitions, tokens, targets and directed coverage; Novice Inspector, ordinary removal, surveil lands. |
| 02 events/history | 28 | Committed event groups, declaration counts, departure LKI and turn summaries; Roots, Homicide Investigator, Teysa. |
| 03 face-down | 49 | Shared face state, special actions, visibility and turn-up events; disguise, cloak, Etrata, Yarus. |
| 04 evidence/costs | 28 | Typed payment composition and provenance across casts, abilities, ward and resolving choices. |
| 05 Cases | 13 | Twelve Cases plus Case File Auditor: solved state, condition gating and solve event. |
| 06 suspect | 18 | Persistent status, prohibition, status predicates and ability-layer interaction. |
| 07 choices/zones | 34 | General choices, relational targeting, retained results, searches and permission lifetimes; includes Surveil repair. |
| 08 copies/faces | 10 | Adventure/split casting, card versus stack copying, temporary copies and linked objects. |
| 09 characteristics/replacements | 20 | Layer/query extensions, damage consequences, poison and replacement authority. |
| 10 mana/setup | 7 | Payment witnesses, compound mana output, spending restrictions and pregame choice. |

Start 01 alongside event/choice repairs. Add 04 with 10, then 06/05 atop event
history. Build 03 once identity, cost and visibility contracts agree. Finish
08/09 through the same authorities. Survivor sets should consume these shared
capabilities first where useful; MKM's rotation priority does not justify a
parallel executor.

Evidence exposes `PaymentCost`'s mana/waterbend limit and field-based activation
costs. Extend one typed payment program with selection constraints, complete
payment receipts and purpose-specific provenance. Reuse it for Axebane Ferox,
Lamplight Phoenix, Urgent Necropsy, Cryptex and Conspiracy Unraveler; Demand
Answers and Rakdos need disjunction and all-or-nothing cardinality too. Preserve
separate cost events and exact exiled references for Voltstrider. Do not encode
all evidence as the existing kicked flag.

Face-down battlefield state is a structural gap: `ZoneEntry::for_zone` currently
clears its exile-only `face_down` flag for battlefield entry. Design a typed
public characteristic profile, private underlying identity and provenance for
turn-up permissions. Keep ordinary target/attachment object identity stable
through turning; zone moves still create new incarnations. Expose the same
viewer-safe decisions to Commands, observations, tensors, tokens and LLM text.
Lumbering Laundry and A Killer Among Us require explicit private look/choice
permissions, not browser filtering. Expose the Culprit also tests hidden
randomization and possible-world knowledge.

Cases and suspect need state distinct from printed abilities and counters.
Their derived abilities join the characteristics authority. Extend existing
turn history with event-time facts, not scans of surviving objects: Burning
Masks distinguishes source incarnations; Gorgon's Kiss distinguishes arriving
creature cards from dying animated noncreatures. Insidious Roots and Chalk
Outline require one occurrence for a simultaneous graveyard-departure group.

The replacement pipeline explicitly documents deterministic source-order
application rather than general affected-player choice. Pilfered Proof's token
replacement and Living Conundrum's draw skip should extend the proposal model;
Massacre Girl needs damage-to-counter consequences and death toughness LKI.
Do not add ad hoc handlers after damage, draw or token creation. Delney already
has a predicate-based duplication primitive; Doorkeeper Thrull adds event
suppression at collection, not a second trigger index.

## Closure, witnesses and cost

Token closure includes Clue; white-blue Detective; white Dog; both 0/0 and 1/1
flying Thopters; black Skeleton and flying Bat; red Goblin; green Plant and
trampling Ooze; white Human; blue Merfolk; black-green reach/menace Spider; red
Imp with death damage; white-black flying Spirit; and legendary green-white
Voja Fenstalker. Keep full definitions, colors and abilities distinct. Split
faces need off-stack combined characteristics; Adventure needs permission after
resolution. Slime Against Humanity also reaches deck-copy admission.

Prioritize adversarial proofs: copy a solved Case; remove abilities from a
suspected creature; pay alternative and additional evidence separately; select
one multi-type Necropsy target in several clauses; interrupt Yarus with a
second graveyard move; turn up a noncreature under Etrata; create copied
face-down tokens; keep two Surveil cards in reverse order; remove a linked
source and return its physical card; and resolve Intrude's opposing pile
choice after a search fork. Use ordinary captured Commands and compare fork
witnesses, public captures and retained private choices.

Runtime risks are evidence subset enumeration, multi-target/distribution
branching, compound mana witnesses, replacement choice suspension and
Doppelgang's quadratic output. Use staged choices and sound feasibility bounds
without deleting legal selections. Maintain event summaries incrementally and
index listeners by event family; avoid scanning full history or all card
definitions per action. Audit tap-revision invalidation for Lost in the Maze.
Measure bounded normal-step and search-fork workloads with the existing paired
Engine Budget and exact digests, tracking cumulative cost. No games, benchmarks
or production edits were performed for this research.
