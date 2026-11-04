# LCI — The Lost Caverns of Ixalan

Research dated 2026-11-04. LCI rotates under the parent research's early-2027
working assumption, so prioritize shared mechanisms needed by surviving sets.
All **291 identities and 327 faces** in `.cache/standard-research/lci.json` were
read: 255 normal cards, 35 transforming cards and one Adventure. The companion
[`lci.jsonl`](lci.jsonl) assigns every identity exactly once, retaining face names.
This includes banned-card inventory entries and basics; it is not a legality
certificate. LCC, REX and SPG are outside this input.

## Existing owners and evidence

There is no `cards/src/lci.rs`. Source-name inspection found only the five basics
among existing definitions; reuse LEA identities. Reprints still belong in their
first-printing modules when added. Definitions remain native `CardDefinition`
values, never a second card language or set-specific executor.

Inspected `state/src/{ability,card,effect,mana}.rs` and
`src/flow/{zones,turn,play,proposed_event,resolution}.rs` already supply transform,
Saga chapters, finality, linked exile, delayed returns, target clauses, optional
payments, reflexive triggers, copy exceptions and layered characteristics.
Finality is explicitly handled by `proposed_event.rs`; it needs combination
proofs, not reinvention. `OnNthResolutionThisTurn` also exists, although its
physical-card/ability key deserves incarnation scrutiny for Vito. No Explore,
Discover, Craft or rebound vocabulary was found in the inspected engine paths.

The [official release notes](https://magic.wizards.com/en/news/feature/the-lost-caverns-of-ixalan-release-notes)
resolve important uncertainties: craft links surviving exiled cards, not vanished
token materials; Locus keeps duplicate granted abilities separately. Explore
still happens after its specified creature leaves, and empty-library exploration
still adds a counter. Discover validates both the found card and selected spell
profile. Market Gnome can trigger during a nested mana activation within craft.
Calendar's untap trigger waits for upkeep priority; its state trigger can recur
after being countered. These are acceptance boundaries, not whole-set certification.

## Shared implementation batches

Counts are primary assignments; secondary dependencies are in the ledger.

| Batch | Cards | Work and authority |
|---|---:|---|
| 01 Composition | 95 | Candidate definitions over existing selectors, costs, triggers and moves; directed proofs remain necessary. |
| 02 Events/history | 45 | Event-time descend, grouped tap/death/damage, crew participation, filtered histories and state triggers. |
| 03 Choices/costs | 28 | Shared sequential selections, typed cost alternatives, multi-object contributions, result receipts and repeated modes. |
| 04 Characteristics | 24 | Predicate/value coverage, layered base-power queries, copy/ability-loss grants and exact durations. |
| 05 Explore | 24 | One resumable action, reveal outcome, counter/keep/mill choice, event and Map closure. |
| 06 Discover/permissions | 26 | Resumable reveal/exile/free-cast-or-hand action; reusable immediate casting and play grants. |
| 07 Craft | 20 | Mixed-zone costs, material links, transformed entry and linked-value/grant consumers. |
| 08 Faces/timing | 11 | Existing transform owner plus death returns, Saga faces, rebound and activation restrictions. |
| 09 Mana/provenance | 12 | Spending-purpose predicates, source-tagged production, payment receipts and dynamic outputs. |
| 10 Replacements | 5 | Entry, life loss, damage, token multiplication and trigger modification coverage. |
| 11 Adventure | 1 | Kellan's alternate spell profile, successful-resolution exile permission and Maps. |

Start with composition and shared predicates; develop events/results and payment
contracts before craft, discover and provenance consumers. These batches are
cross-set work views, not eleven independent engines.

**Coverage extensions:** descend-four/eight use live graveyard permanent counts;
“descended this turn” instead counts permanent-card entries from any zone at the
zone-move authority. Tokens never count. Artifact entries, crew participants and
damage groups likewise need event-time records, not later board reconstruction.
Reuse WOE events/history, DFT contribution costs, FIN paid-mana/face work, TDM
spell profiles and TLA timing/permissions. Historical grants and copied abilities
must retain exact identities, not derive them from present card names.

**Actual contract changes:** craft pays and exiles before resolution, then
returns a new incarnation transformed. A typed cost receipt links each material
incarnation to that returned object. Throne requires four distinct materials;
one multitype object cannot fill four slots. Linked CDAs read current exiled-card
characteristics; Saheeli's Lattice cannot snapshot power at crafting. Locus needs
activated-only grants with independent stable identities, not the existing
single-definition activated-and-triggered grant.

Discover must suspend the enclosing resolution during a genuine cast, preserve
additional costs and spell-profile legality, then randomize the remaining exiled
cards deterministically. A discovered Adventure is not a transforming back face.
Rebound differs from flashback: its exile occurs on successful resolution and
creates a future optional cast. Tomb/Paleontologist permissions modify entry;
Inti expires at the next end step, not cleanup. Reuse `cast_permission` ownership.

`ManaUse` currently encodes three subtype bits, not arbitrary artifact/land
purposes, chosen creature types or production provenance. Extend one typed
spending/production contract used by offers, solver, execution and observation.
Cave mana for Bat Colony, Barracks, Myriad Pools and Tecutlan must remain
attributable through pooling and spending; equal colors are not interchangeable
when their consequences differ. Separate mana amounts from immutable provenance
classes to avoid duplicating whole source objects in forks.

Choices and replacements also cross existing boundaries: crew currently excludes
nonpositive power and stops automatically at its threshold; both hide strategic
taps in LCI. Preserve a legal finish choice and all qualifying contributors.
`proposed_event` currently applies collected replacements in order: Spelunking,
Ojer effects and life-loss changes require affected-player choice and reevaluation.
Braided Net needs an ends-when-untapped duration; Larcenist/Tidebinder need
source-presence duration without treating control change as departure.

## Closure and adversarial witnesses

Register Map; colorless artifact Gnome; white dynamic artifact Gnome Soldier;
black flying Bat; white flying/vigilant Angel; blue hexproof Merfolk; black
nonblocking Fungus; white-black flying Vampire Demon; black Skeleton Pirate;
white lifelink Vampire; red 3/1 and green 3/3 Dinosaurs; green Dinosaur Egg;
red-white Spirit; white-blue artifact Golem; and green dynamic Fungus Dinosaur.
Reuse Treasure. Dynamic copies and every back face need complete characteristics,
abilities and first-print identities; no new emblem is required.

Use bounded Command fixtures for: token versus card descent; simultaneous
Merfolk taps; zero-power and excess crew; departed/empty-library explore;
Twists replacement reentry; countered craft with materials already gone;
copying craft onto a single-faced card; removing one material after crafting;
duplicate Locus grants; illegal discovered Adventure profile; cast prohibition
forcing discover-to-hand; Cave mana split across spells; countered Vito triggers
and blink; Bringer excluding only its own sacrifice receipt; finality versus
God death returns; layered base power under copy/CDA/animation; and Calendar's
pending state-trigger guard. Jade Seedstones must preserve announced allocations
when one target becomes illegal. Preserve LKI for Stalactite Stalker and grouped
Skullspore deaths.

All public histories, cost/material links, chosen types, mana provenance and
suspended decisions must reach native/Python observations, LLM text and learning
tokens together. Private hand/top-library visibility remains viewer-specific.
Keep exact replay/fork identity, stale rejection and trivial-decision skipping;
never silently reinterpret frozen policy inputs.

## Runtime/search acceptance and limits

Update compact histories at event creation and use the trigger index. Cache
live graveyard/type summaries and linked grants on relevant revisions, preserving
duplicates. Share immutable definitions and provenance descriptors across full
clones. Sequential material/mode selections need feasibility pruning rather than
Cartesian action expansion; ordinary cards must not scan LCI-specific handlers.

No games, benchmarks or production changes were run. Future acceptance requires
debug tests and same-runner Engine Budget checks for both ordinary steps and
search, preserving exact-result digests and cumulative milestone cost. Add small
craft, discover, grouped-event and provenance workloads; inspect clone size,
allocation and offer growth alongside time. Static reading establishes a full
inventory plan, not implemented correctness or a measured speed claim.
