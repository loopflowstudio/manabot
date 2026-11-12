# Aetherdrift: contribution costs, activation history and player-owned rules

Research: 2026-11-04; inspected checkout `5dd11cec4`. This is a planning review,
not an implementation or conformance claim. All 276 identities and their full
rules text in the downloaded DFT input were read. The [ledger](dft.jsonl) assigns
every identity once, with card-specific obligations and additional dependencies.
Its coverage check passes: 276 rows, 276 distinct Oracle IDs, exact input match.
No games, benchmarks or runtime tests were run.

The input is Scryfall's unique-card DFT inventory, pinned in the parent
[source inventory](../source-inventory.json). DFT survives the next rotation.
DRC Commander and SPG Special Guests are separate products: appearing in an
Aetherdrift booster does not confer DFT Standard legality. Use actual format
membership and current bans, including later reprints. Existing source-name
inspection found six matching definitions: Lightning Strike in `ths.rs` and
five basic lands in `lea.rs`; there is no DFT module. This is a source inventory,
not proof those definitions meet every DFT interaction.

## Shared batches

Counts identify the primary planning owner; dependency edges cross the rows.
They are not independent engine implementations or estimates of effort.

| Batch | Cards | Main shared work |
|---|---:|---|
| A — core composition | 43 | Reuse verbs, predicates, values, target clauses and results |
| B — mobility contribution costs | 45 | Complete crew/saddle choices, contribution modifiers and participation events |
| C — activation costs/history | 45 | Exhaust identity, cycling events, composite costs, activation triggers |
| D — mana/pricing | 36 | General spending-purpose predicates and shared cost evaluation |
| E — private-zone programs | 29 | Resumable selection, ordering, permissions and result binding |
| F — player speed | 39 | Player-owned state, SBA initialization and conditional abilities |
| G — event history | 3 | Aggregate events, per-turn modes and contextual counts |
| H — characteristics | 13 | Contextual values, conditional layers and copy-aware readers |
| I — replacement/copy/control | 17 | Choice and re-evaluation, exact incarnation and copiable exceptions |
| J — scheduled effects | 4 | Ordered combats and captured delayed subjects |
| K — planeswalker authority | 2 | Loyalty, attackability, attachments and emblems |

Begin with A alongside the shared B–E foundations; event-history work enables F
and G. H and I consolidate existing authorities before their complex cards are
registered. J extends the TLA scheduler. K should share a cross-set planeswalker
implementation. A's cards still need their individual predicate/target audit;
“core” does not certify zero engine changes.

## The strongest multiplier: contribution costs

[Current crew selection](../../../../../../managym/src/flow/play.rs) has two concrete
shortcuts: `crew_candidates` admits only positive-power creatures (around line
1651), and `crew_tap_action` completes immediately at the threshold (around
1680). DFT makes omitted choices consequential: Interface Ace untaps after
being tapped; Canyon Vaulter grants an ability to the recipient; extra or
zero-power crew members can deliberately cause tap/participation triggers.

Replace that crew-specific prefix with a shared selection/payment context:
eligible contributors, amount evaluator, optional completion once sufficient,
and the exact chosen set. Crew and saddle share selection but retain distinct
rules events, source exclusions and timing. Pilot bonuses and Interface Ace's
toughness substitution modify contribution, **not actual power**. Alacrian
Armory/Guidelight Matrix can saddle without paying saddle; Kolodin can animate
without crew. Neither should fabricate participation events.

Acceptance: preserve zero/excess contributions, negative/modified power,
summoning-sick contributors, repeated participation after untap, and a creature
that is both Mount and Vehicle. Rover Blades must detach when it becomes a
creature. Use incremental choices rather than enumerating every contributor
subset into the policy action table.

## Activation and mana are one larger boundary

[`ActivatedAbilityDefinition`](../../../../../../managym/state/src/card.rs) already
has exhaust, once-per-turn, hand activation, discard and crew fields. Extend
shared cost and activation semantics rather than adding one field for each
DFT card. Draconautics Engineer and Loot need independent per-ability use;
Elvish Refueler changes permission without resetting history. Afterburner Expert
needs a graveyard trigger for another exhaust activation. Cycling must retain
announced X for Valor's Flagship/Webstrike Elite and emit discard and cycling
events with the correct ordering.

[`ManaUse`](../../../../../../managym/state/src/mana.rs) currently names three subtype
bits. Guidelight Optimizer, Boommobile, Redshift and Pit Automaton require
artifact-spell and arbitrary-activation restrictions; adding more special bits
would repeat the same scaling problem. Use a typed purpose predicate evaluated
by both pricing and execution. Loot's upfront-cost mana ability must retain a
payment witness. Radiant Lotus targets a player, so its mana-producing ability
uses the stack; it must not enter the automatic mana path. The current registry's
finite-use guard is broader than tap-only; investigate each new source's actual
planner transition rather than relying on an older README generalization.

Acceptance: artifact affinity locks before sacrificing an artifact for mana;
restricted mana cannot pay unrelated costs; copied exhaust is not another
activation; multiple ward costs each produce their own obligation.

## Speed and event results

Speed is genuinely player-owned. It starts through an SBA, persists after the
source leaves, and later increases through a once-per-turn player trigger.
Max-speed abilities can function outside the battlefield. These details are
specified in the [official release notes](https://magic.wizards.com/en/news/feature/aetherdrift-release-notes).
Store speed and relevant turn history once; expose it through facts and all
observation/token paths. Avoid per-card speed flags and permanent scans on every
query. Test countering the speed trigger, multiple life-loss events, control
changes, Spikeshell Harrier's comparison, and graveyard max-speed abilities.

Discard, mill, sacrifice and movement results need reusable typed context.
Greasewrench Goblin draws its actual discarded count; Dredger's Insight selects
only newly milled cards; Repurposing Bay reads sacrificed mana value; Monument
to Endurance chooses unused modes at trigger placement. These cannot be
approximated by global counts or a last-touched-card variable.

## Higher-risk reuse and acceptance

E should extend ordered private selection once: Stock Up, surveil two, multiple
search zones, exact library insertion and Gonti's face-down permissions.
[Current Surveil](../../../../../../managym/state/src/ability.rs) explicitly preserves
kept order for the registered count-one subset; Grim Bauble and Wreckage
Wickerfolk require more. Ancient Vendetta also exposes catalog-wide naming's
fixed action-capacity problem. No card-specific UI or silent choice is acceptable.

I is a useful independent test suite for the replacement redesign: Caradora
modifies counters, Vnwxt replaces draws, Far Fortune modifies damage, and
Pyrewood Gearhulk prohibits prevention. Mimeoplasm needs as-entry selection,
linked exiles and repeatable copy changes; Cursecloth Wrappings grants a
zone-specific ability whose copy exceptions persist. Trade the Helm is atomic;
Coalstoke Gearhulk combines entering controller, finality and delayed identity.
Exercise competing replacements and departed sources instead of testing only
isolated final boards.

Full Throttle must insert two combats without inventing a main phase between
them; Push the Limit captures returned objects for sacrifice. The Aetherspark
is a planeswalker/Equipment integration witness, not permission to add an
Aetherspark handler. Chandra requires a command-zone emblem, independent of its
source's continued existence.

Required created-object closure includes Cat, Servo, Thopter, Pilot, Zombie,
Dinosaur Dragon, Goblin, Elephant, Insect, Vehicle and Treasure definitions,
plus Chandra's emblem and arbitrary copied/embalmed objects. Reuse existing
matching token definitions; special Pilot and Vehicle abilities are real rules.

## Performance and evidence limits

Keep indexed event dispatch, exact-object joins and cached characteristics.
Profile payment branching, simultaneous entry/counter replacement, large private
selections, generated tokens and copied grants with bounded real positions.
Compare ordinary step and fixed-budget search separately on one runner, retaining
exact digests for equivalent semantics. No speed claim follows from this review.
The ledger is exhaustive planning triage, not a line-by-line CR proof; each batch
still needs directed debug tests and native/policy/text/Academy parity before
claiming implemented support.
