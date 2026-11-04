# TLA: finish player control and consolidate the implemented machinery

Research date: 2026-11-04; checkout `5dd11cec4`. All **286 identities and 294
faces** in the downloaded TLA input were read in full. The [ledger](tla.jsonl)
reconciles the retained per-face allocation with that text and assigns every
identity to a present-day shared-work batch. Historical T1–T7 owners remain
provenance, not a proposed delivery schedule. No runtime tests or games ran.

## Present boundary

A literal-name scan finds matching front-face definitions for **285 of 286**
identities. **Secret of Bloodbending** is the only missing name. This measures
source presence, not independently verified execution or complete interactions.
Preserve reprints in their first-printing modules and the eight attached back
faces; a back face is not a second deck entry. The input's bans and format
membership remain separate from definition-based playability. TLE and adjacent
products are not automatically part of this TLA inventory.

TLA is therefore primarily an existing-behavior and consolidation workload for
the Standard milestone. Do not schedule another wholesale set implementation.
Its many composed cards should become regression witnesses when shared owners
change. Known defects should remain visible without withdrawing playable cards.

## Shared batches

Primary counts sum to 286; dependency lists cross these queues. They do not
measure new engine work or prove the nominally simple cards bug-free.

| Batch | Cards | Representative purpose |
|---|---:|---|
| Core composition | 109 | Preserve ordinary verbs, target clauses, triggers and token production |
| Payment and permissions | 60 | Waterbend, restricted mana, optional/additional costs, exile/graveyard casting |
| Characteristics and contextual reads | 31 | LKI power, CDA values, layers, attached/granted characteristics |
| Selection and instruction results | 29 | Ordered private inspection, searches, allocations, opponent choices |
| Events and replacements | 13 | Counters, entry controller, damage modification, expiry and actual results |
| Faces and scheduling | 13 | Sagas, transform, delayed exact subjects, extra turns/combats |
| Copies and ability grants | 10 | Copiable values, copied triggers/spells, inherited activated abilities |
| Event/resolution history | 8 | Reflexive triggers, trigger placement, resolved versus accepted occurrences |
| Exact-object lifetime | 6 | Linked exile, source departure, returns and control changes |
| Contribution costs | 5 | Complete crew choices and recipient animation |
| Catalog naming | 1 | The Rise of Sozin's chapter-II naming/search decision |
| Player control | 1 | Secret of Bloodbending's delegated decisions and visibility |

## Highest-value work and deletion boundaries

**Finish player control with EOE, not as a TLA-only exception.** Secret of
Bloodbending should share the decision-controller and visibility authority
needed by The Dominion Bracelet. Active player, resource owner and decision
maker are distinct. The controller receives permitted private information but
not sideboard visibility; costs use the controlled player's resources. Control
must span nested decisions, expire at the correct actual combat/turn boundary,
and obey later overriding control effects. A skipped phase must not consume the
scheduled control. Concession remains the affected player's decision.
These distinctions follow the [official release notes](https://magic.wizards.com/en/news/feature/avatar-the-last-airbender-release-notes).
Do not implement this by swapping browser seats or globally changing permanent
controllers. Native Commands, capture, reconnect, policies and search require
the same decision authority.

**Repair contextual reads before extending their consumers.** The
[current-engine audit](../current-engine-audit.md) identifies source-power
live-or-zero behavior and owner-based entry replacement queries. TLA supplies
real witnesses: Firebending Student, Fire Lord Zuko and Combustion Man after
source departure; Cruel Administrator and conditional-entry lands entering
under a nonowner. Consolidate these with exact-object LKI and prospective entry
controller/characteristics. Preserve current `Game::characteristics`, contextual
card power and debug cache oracles. These are source-supported concerns here,
not newly reproduced failures.

**Make the existing compositions safe to extend.** `ForEachTarget` currently
requires non-suspending children. Appa, Foggy Swamp Visions and Sozin reanimation
use it for present combinations; a future replacement choice can make previously
automatic movement suspend. Shared traversal must preserve iteration cursor,
target-clause eligibility, result bindings and the parent completion. Migrate
search/discard/mill composites into a common selection/result owner as real
consumers require it, deleting migrated continuations rather than retaining
parallel paths. Accumulate Wisdom, Elemental Teachings and Lost Days provide
ordered, opponent-chosen and private selection witnesses with different rules.

**Preserve complete contribution choices.** The Fire Nation Drill, Phoenix Fleet
Airship, Tundra Tank, War Balloon and Fire Nation Warship inherit the current
crew shortcut: positive-power contributors only and automatic completion at the
threshold. Tapping extra or zero-power creatures can trigger Gran-Gran or
Compassionate Healer. Replace that prefix with complete shared cost selection,
including a deliberate finish when sufficient. Reuse appropriate cost parts
for saddle and station, without giving them crew's timing or semantics.

**Keep the correct existing payment/history distinctions.** `ManaUse` should
generalize from its three subtype bits into typed spending contexts, preserving
Herbalist's Lesson restriction and Jasmine Dragon Tea Shop's Ally-source
activation permission. Barrels of Blasting Jelly already exercises a non-tapping
once-per-turn filter: the live registry permits `tap_source || once_each_turn`.
Do not invent a defect from the README's older tap-only wording.

South Pole Voyager already uses `OnNthResolutionThisTurn`; Earth Kingdom General
uses `MayOnceEachTurn`. They are useful existing owners for later sets' resolved
ability counts and optional-acceptance limits. Retain their difference from
`OnceEachTurn` trigger limiting. TLA also requires distinct bending events;
sharing counters, animation or mana internals must not erase those events.

## Catalog growth and retained semantics

The naming card is **The Rise of Sozin**, not Sozin's Comet. Its implementation
builds owned modal programs for every catalog name, which cannot scale through
the default 512-action policy input. Investigate immutable candidate sharing
and an adequate shared policy decision representation without shortlists or
browser-only filtering. Preserve resolution-time naming, fewer-than-four
search choices, private inspection and the later reflexive reanimation window.

The accepted T6 boundary used real implemented-catalog names, including back
faces; it invented no names for unsupported cards. Printed naming can range
beyond Standard or implemented definitions. Full Standard support therefore
needs an explicit naming-universe decision: possibly separate authoritative
real-name metadata from executable definitions. This is a proposal, not a silent
change to accepted T6 semantics or a reason to add dummy card programs.

## Acceptance, object closure and performance

Keep nine fixed TLA token definitions: Ally, Spirit, Dragon, Monk, Soldier,
Bear, Ballistic Boulder, Clue and Food. Their abilities are rules behavior;
the Copy marker instead represents arbitrary copiable values and exceptions.
Treasure exists in the wider catalog but is not produced by the TLA closure.
Preserve exact transformed identity, lore/chapter progression, remembered
delayed subjects and copied/inherited abilities after the source departs.

Use small cross-mechanism witnesses: crew with a tap trigger and excess
contributors; a replacement interrupting a multi-object move; Sozin name/search
at full catalog scale; controlled-player private selection and reconnect;
countered versus resolved Voyager triggers; declined then accepted General
triggers; and source removal before dynamic firebending resolution. Exercise
native Commands, forks/replay, scalar/vector/Python observations, text, tokens
and Academy on the same decisions.

Retain the existing TLA step/Search-64 workload as a comparison while extending
bounded cases for these new combinations. Profile shared payment, characteristic,
event and continuation work. Do not sacrifice choices, identity or information
to improve timing; do not claim equivalent-work speedup when rules change the
trace. No speed improvement or rules certification is established by this review.

Validation: 286 unique Oracle-ID/name pairs exactly match input; all 294 face
names retained. Obligations reuse the inspected historical allocation where
still accurate and refine the relevant current boundaries. Only the report and
ledger were written; production code and frozen evidence remain unchanged.
