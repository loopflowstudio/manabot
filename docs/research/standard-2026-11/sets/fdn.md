# Foundations: shared-engine implementation plan

Research snapshot: 2026-11-04, checkout `5dd11cec4`. **Planning, not a rules-conformance certificate.** Every one of the 517 input identities was read in full and receives one row in [fdn.jsonl](fdn.jsonl). No games, tests or benchmarks were run. The ledger's primary batch is its main implementation concern; dependencies are additional concerns, not a complete scheduling graph.

## Inventory and present support

The input is the paginated Scryfall `e:fdn`, `unique=cards` snapshot in `.cache/standard-research/fdn.json`; its page metadata retains URLs, retrieval dates and digests. All 517 rows currently report Standard legality in that secondary source. This includes Starter Collection and Beginner Box cards, not just booster rares and commons. The [official release notes](https://magic.wizards.com/en/news/feature/foundations-release-notes) separately discuss those products, Commander additions and Special Guests. Product adjacency must not import Commander-only cards, Special Guests or Jumpstart 2025 into the Standard target. Reconcile membership against the official format inventory before treating this as a legal-card certificate.

A textual exact-name scan finds **12 existing definitions**: Savannah Lions, Llanowar Elves, the five basics, Giant Cindermaw, Healer's Hawk, Serra Angel, Spectral Sailor and Shivan Dragon. Preserve those first-printing definitions; do not duplicate them in `fdn.rs`. This leaves 505 definition candidates, but existing engine operations can express substantial portions. Definition presence does not prove behavior. Existing `world::set_cards` filters first-printing key prefixes and cannot establish reprint legality.

## Ordered batches

| Batch | Primary rows | Mechanism and representative work |
|---|---:|---|
| Foundation composition | 149 | Existing verbs, target clauses, keywords and ordinary triggers: Helpful Hunter, Stab, Day of Judgment, Burst-style effects without new semantics. |
| Costs and mana | 100 | Shared obligations, alternatives, spending purposes, cardinality and source-use resources: Sire of Seven Deaths, Ovika, Lathril, Quilled Greatwurm, Secluded Courtyard. Includes ordinary dual-land authoring. |
| Choice continuations | 57 | Private inspections, retained ordering, opponent choices, related targets and allocations: Cephalid Inkmage, Curator of Destinies, Blasphemous Edict, Biogenic Upgrade. |
| Characteristics | 50 | Layered predicates, subtype changes, ability grants and protection scopes: Witness Protection, Imprisoned in the Moon, Kellan, Elenda. |
| Event history | 47 | Event counts, actual amounts, turn scopes and exact source instances: Cat Collector, Venom Connoisseur, Predator Ooze, Steel Hellkite. |
| Context queries | 34 | Typed quantities and predicates over source, target, event and completed instruction: Lunar Insight, Fumigate, Heartfire Immolator, Fiendish Panda. |
| Replacement events | 25 | Entry, damage, counters, life and zone movement: Doubling Season, Twinflame Tyrant, Progenitus, Angel of Vitality. |
| Object lifetime | 23 | New incarnations, simultaneous movement, linked returns and copy exceptions: Nine-Lives Familiar, Abyssal Harvester, Valkyrie's Call. |
| Play permissions | 13 | Zone, timing, payment and per-turn usage are independent: Muldrotha, Tinybones, Omniscience, Vizier. |
| Game procedure | 11 | Loyalty/emblems, poison, opening-hand entry, turn termination and win/loss prevention: Kaito, Fynn, Leyline Axe, Time Stop, Herald. |
| Combat constraints | 5 | Mandatory attack/block and subtype restrictions: Juggernaut, Joraga Invocation. |
| Stack copies | 3 | Copy multiplicity, retargeting and delayed watchers: Thousand-Year Storm, Teach by Example, Bolt Bend. Other batches depend on this machinery. |

These are dependency-oriented authoring queues, not independent executor projects. Begin with a small composition slice and query/history foundations; then shared payment, resumable choice and replacement authorities; finally layer/lifetime/permission combinations and procedure changes. Easy definitions can proceed as their dependencies become available. Do not finish all 149 nominally simple cards before testing difficult combinations: a cheap early correctness witness for each authority provides more useful leverage.

## Highest-value consolidation

**One contextual query authority.** `EffectValue`, `Condition`, `CardPredicate` and source/target helpers currently contain several one-purpose questions. Extend their common subjects and operations instead of adding `FumigateCount`, `LunarInsightCount` or another card-shaped enum. Heartfire Immolator needs sacrificed-source power; Fiendish Panda needs death LKI for targeting; Tribute to Hunger needs the sacrificed object's toughness. `EffectValue::SourcePower` currently documents zero after source departure, whereas `TriggeringPower` has LKI. Consolidate exact-object characteristic queries before multiplying these paths. Retain explicit distinctions between target announcement and resolution values.

**Resumable traversal and instruction results.** Generalize the existing frame queue, not a second interpreter. The current `ForEachTarget` branch in [`resolution.rs`](../../../../managym/src/flow/resolution.rs) requires non-suspending children; that is insufficient for sequences containing opponent choices or searches. `Surveil` in [`decision.rs`](../../../../managym/src/flow/decision.rs) can move selected cards to the graveyard but declining resumes without ordering multiple retained cards. Cephalid Inkmage requires that missing order. Use a retained cursor, typed result bindings and explicit choice ownership. Preserve simultaneous sacrifice/movement; iterating player choices does not mean applying each sacrifice immediately.

**One replacement authority.** Doubling Season must distinguish counters added as effects from loyalty costs. Fumigate counts actual destruction; Exsanguinate gains actual lost life. Replacement results, not requested amounts, should supply later instructions. Generalize event predicates and replacement selection/re-evaluation in the existing proposed-event pipeline, then remove equivalent special cases. Independently test prevention against doubling and source/controller changes.

**Costs, permissions and characteristics compose.** Life and composite ward require a richer obligation vocabulary than current `PaymentCost::{Mana, Waterbend}`. Muldrotha's type quota is neither general flashback nor a global cast flag. Secluded Courtyard's mana can pay an activation of a matching creature source, unlike a casting-only restriction. Prefer typed spend contexts and permission records with ownership and expiry. Three Tree Mascot is a useful boundary test, **not a demonstrated guard violation**: the actual mana registry guard permits `tap_source || once_each_turn`. Verify once-per-turn reservation/payment behavior before changing it; the README's stronger tap-only description is stale.

## Acceptance and performance

Use small real-card scenarios crossing native Commands, forks/replay, tensor/text observations and tokens:

- Remove Heartfire Immolator as its activation cost; preserve its last power. Contrast a blinked source incarnation and source power changing while the ability waits.
- Surveil three, mill one, explicitly reverse the other two; keep opponent visibility redacted through reconnect/fork.
- Resolve Doubling Season with a planeswalker entry and then a loyalty activation: entry replacement and cost payment must differ. Combine token creation and copy exceptions.
- Reanimate several creatures simultaneously under a different controller; entry replacements and death/entry trigger sources must observe the appropriate batch state.
- Pay colored costs through Three Tree Mascot, Secluded Courtyard and tagged Pyromancer's Goggles mana; preserve alternatives and reject repeated use where the ability forbids it.
- Inspect an opponent's hand with Spyglass, choose a name outside both decks, and disable that name's nonmana abilities in every relevant zone. Avoid one policy action per catalog card.

Keep whole-game `Env::step` and search workloads. Profile trigger/event indexing, repeated characteristic queries, copy/fork allocation, private-choice expansion and mana feasibility. Predicates should advertise the state revisions they read so caching remains correct. Add bounded mixed-mechanic positions rather than combinatorial pairwise card testing. A new batch's acceptable cost is measured on the same runner; it does not justify cumulative slowdown or removal of existing legal choices.

## Created objects and procedure closure

The card ledger is not the complete runtime-object inventory. Derive required Cat, Food, Knight, Faerie, Drake, Spirit, Insect, Rat, Goblin, Dragon, Elf Warrior, Raccoon, Fish, Zombie, Human, Soldier, Treasure, Beast, Serpent and other printed tokens from producer cards. Distinguish characteristics, colors, intrinsic abilities and named legendary objects such as Scion of the Deep and Koma's Coil. Token copies retain arbitrary copiable payloads; haste/death-trigger exceptions cannot be flattened into an ordinary printed token. Kaito and Vivien emblems need persistent rule sources, observation and learning representation independently of their departed planeswalkers.

Leyline Axe makes pregame procedure a **card dependency**: selection occurs from the actual opening hand after mulligan handling, not by moving it into play after normal setup. Thus setup cannot be categorically deferred while claiming all FDN cards work. Time Stop needs cleanup semantics; Herald and poison need terminal-state arbitration; Hare Apparent needs construction exceptions. These boundaries should be explicit shared engine work, while between-game tournament sideboarding remains a separate product decision.

Validation: exact 517-row Oracle-ID/name coverage verified against the fetched input, with no missing or duplicate rows; face names retained. This pass supplies reviewed planning obligations, not proofs that all proposed primitives already work.
