# WOE — Wilds of Eldraine

Research dated 2026-11-04. This rotating set is lower priority than the surviving
pool, but its mechanisms are reusable across that pool. The input is all **281
Oracle identities** in `.cache/standard-research/woe.json`, including Jumpstart
cards, basics and every Adventure face. Every complete rules text was read;
[`woe.jsonl`](woe.jsonl) assigns each identity exactly once. These are inventory
memberships, not a declaration that all are currently legal: legality and bans
remain the parent inventory's separate reconciliation. WOC and WOT are excluded.

## Existing owners and evidence

There is no `managym/cards/src/woe.rs`. Source-name inspection found Prophetic
Prism in `roe.rs` and the five basics in `lea.rs`; reuse their first-printing
identities. This is not a runtime registration or correctness certificate.

The inspected shared vocabulary in `state/src/ability.rs`, `card.rs`,
`predicate.rs` and `effect.rs` already provides selectors, target clauses,
optional payment, reflexive triggers, typed costs, Saga chapters, delayed
watchers, copy exceptions, linked exile and layered characteristics.
`src/flow/cast_permission.rs` owns exact exile permissions;
`src/flow/mana.rs` owns payment witnesses. Reuse those owners rather than
introducing WOE executors. Notably, `CardDefinition.back_face` describes
transforming cards, not Adventure spell profiles. Existing
`OwnerPutsIntoLibrary` chooses second-from-top/bottom, so Misleading Motes
cannot be authored with it unchanged.

The [official release notes](https://magic.wizards.com/en/news/feature/wilds-of-eldraine-release-notes)
independently confirm several acceptance boundaries: Roles conflict per
controller and newest timestamp, with a choice for simultaneous ties; illegal
attachment can prevent token creation. Bargained spell copies retain bargain,
but copies of battlefield permanents do not. Adventures use the selected
spell's characteristics and grant later casting only after resolving into
exile. Ashiok's replacement is mandatory when applicable but does not enlarge
legal life payments. Soul Cauldron grants activated abilities, not triggered
or static abilities. These checks inform the witnesses below; they do not
certify the whole set.

## Shared implementation batches

Primary counts are planning assignments, not mutually exclusive mechanism
counts. The ledger records secondary dependencies and card-specific obligations.

| Batch | Cards | Shared owner and scope |
|---|---:|---|
| 01 Composition | 77 | Existing vocabulary candidates: targeting, ordinary triggers, moves, counters, look/select, Food/Treasure and known mana modes. Add definitions and directed witnesses; do not infer correctness from matching text. |
| 02 Events/history | 31 | `flow/event`, `turn`, `triggers`: event-time nonland entries, filtered cast ordinals, lifetime-of-turn damage, life totals gained/lost, tap causation, combat and grouped occurrence counts. |
| 03 Choice continuations | 27 | `flow/decision`, `resolution`, `zones`: typed number choices, bulk sacrifice outcomes, top/bottom placement, distributed targets, private search and returned/exiled-card receipts. |
| 04 Costs/provenance | 28 | Cast/payment authority: optional sacrifice costs, bargain-conditioned targeting/reductions and carried cast facts; generalized counter/exile/multiple-sacrifice payments. |
| 05 Roles/attachments | 29 | Zone entry, attachment legality, layers and `flow/sba`: ordinary Aura token definitions plus Role conflict handling, attachment creation and granted triggers. |
| 06 Adventure profiles | 51 | Card/spell identity, casting, zones and permissions: alternative spell characteristics, resolution exile, graveyard-only Adventure grants, face-aware predicates and expiry boundaries. |
| 07 Characteristics/grants | 26 | `flow/statics`, copiable values and exact linked objects: relational predicates, dynamic grants, source-relative values, type/name/color changes and spell-level lifelink. |
| 08 Mana/payment | 12 | Payment/proposed-event authority: typed spending predicates, activation discounts/floors, mana multiplication, chosen entry colors and life-payment replacement. |

02 largely extends event vocabulary, but attribution/grouping must be preserved
at event creation, not reconstructed by scanning history on every offer. 03–06
require structural continuation, casting or state-based-action work beyond new
card definitions. 07 includes both predicate coverage and structural changes:
Soul Cauldron's linked set of activated-only grants cannot be represented by
Koh's single-source activated-and-triggered grant. Likeness Looter needs a
persistent copy with retained activation; The Irencrag needs a coordinated
name/type/ability transformation. 08 extends common cost and replacement
contracts. Batches are mergeable cross-set work, not an instruction to build
these eight in isolation or sequentially.

Adventure identity is the largest dependency fan-out: the Virtues, Johann,
Beseech the Mirror, Beluna, Hearth Elemental and Mosswood Dreadknight all
exercise different consumers. A cast profile must drive mana value, colors,
permissions, tokens and offers consistently; do not pretend Adventure is
transforming or simply an alternative cost. Expensive Taste requires permission
for its caster to inspect another owner's face-down exiled cards, beyond
foretell's owner-based privacy. Questing Druid's next-end-step expiry is not
existing through-next-turn cleanup expiry.

## Closure and adversarial witnesses

Define six Roles, plus white vigilant Knight, white flying Bird, white Human,
white Mouse, blue flying Faerie restricted to blocking flyers, black nonblocking
Rat, black Nightmare with exile-history growth, green Beast and white-blue
Elemental. Reuse Food and Treasure. Human Soldier is not the Human token.
Dynamic copy outputs retain copiable exceptions and exact identity; no WOE
emblem is required by this inventory.

Use small constructed Command fixtures: overlapping controllers' Roles;
simultaneous Role creation and Wicked deaths; Curse of the Werefox with failed
attachment; Knight of Doves dying before attached Auras reach the graveyard;
Brave the Wilds losing its bargained target; countered Adventure versus copied
Adventure; Johann casting an Adventure from library top; Beseech choosing a
legal Adventure profile; Soul Cauldron losing its source during payment;
Talion comparing a spell's announced X and characteristic-defining power;
Tangled Colony after damage is removed; Imodane with two slots naming one
creature; Gruff Triplets dying together; and Ashiok with insufficient library
or insufficient life. Also check Troyan mana spent across two spells and
Agatha's one-mana floor with other reductions.

Every new public fact, selected profile, linked grant, lifetime, remembered
object and suspended selection must reach native/Python observations, LLM
text and learning tokens together. Hidden search/exile contents remain
viewer-specific. Preserve stale Command rejection, reconnect decisions, replay
identity and automatic trivial-decision skipping; frozen inputs must not gain
silent reinterpretations.

## Runtime/search acceptance and limits

Keep event counters incrementally at their authority and index trigger sources
by event kind. Cache dynamic grants by source incarnation, linked membership
and relevant revisions, preserving ability identities and duplicate grants.
Reuse immutable card/profile definitions across full-clone forks. Avoid
materializing Cartesian products for bulk sacrifice or counter allocation:
suspend through exact sequential choices with feasibility pruning, retaining
all strategic alternatives. Do not make ordinary cards scan all WOE handlers.

The README's claim that chosen mana modes all tap is narrower than current
`registered_mana_activations_have_finite_source_costs`: it admits
`tap_source || once_each_turn`. Scarecrow Guide therefore needs targeted
verification, not an assumed new executor. Multiple granted abilities on one
source still challenge one-source-per-plan resource accounting.

No games, benchmarks or production changes were run. Future acceptance needs
debug directed tests and the established same-runner Engine Budget for both
ordinary steps and search, with equivalent-workload digests and cumulative
milestone cost retained. Add bounded Role, Adventure and granted-mana fixtures;
measure clone size and offer growth as well as elapsed time. Ledger confidence
is deliberately limited: full inventory reading plus static inspection is not
executed full-card support.
