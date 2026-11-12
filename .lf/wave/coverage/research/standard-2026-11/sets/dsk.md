# DSK — Duskmourn: House of Horror

Reviewed 2026-11-04 at `5dd11cec4`. DSK rotates at the next rotation under the
research's working boundary; its shared mechanisms remain useful to surviving
sets. The input contains **276 Oracle identities and 299 faces**, including 23
Room cards. Every full rules text and face was read; [dsk.jsonl](dsk.jsonl)
assigns each identity exactly once. This is a mechanism/authoring inventory,
not Standard legality certification or a passing gameplay claim.

The secondary transcription is `.cache/standard-research/dsk.json`, with dated
source digests. [Wizards' release notes](https://magic.wizards.com/en/news/feature/duskmourn-house-of-horror-release-notes)
were consulted for Rooms, impending, manifest dread and survival. The source's
Central Elevator explanatory bullet appears to omit “doesn't” and contradicts
its own displayed Oracle instruction; implement the negative name comparison
in the inventory, with an independently specified witness, rather than copying
that apparent editorial error.

## Existing support and shared ownership

There is no `dsk.rs` module. Static catalog inspection finds six existing input
names: Pyroclasm in `ice.rs`, plus the five basics in `lea.rs`; these retain their
first-printing identities. The remaining 270 are not existing definitions.
The 63 composition-batch rows include those six and 57 definition candidates;
“candidate” means reusable primitives appear sufficient, not proven coverage.

The strongest reuse is in [ability.rs](../../../../../../managym/state/src/ability.rs):
`TargetClause`, `Move`, `Moved`, `PutCountersFrom`, `CreateDelayedTrigger`,
`WhenYouDo`, `OnNthResolutionThisTurn`, retargeting, private selection and
copy-with-exceptions. Finality and stun already exist. Existing costs cover
sacrifice-or-mana, hand activations and once-per-object activation; existing
cast permissions cover specific exile windows and flashback. These avoid
bespoke handlers for Rite of the Moth, Clockwork Percussionist, and similar cards.

Important boundaries remain concrete. `ZoneEntry::for_zone` clears face-down
state for battlefield entry; foretell's face-down exile is not manifest support.
`CardDefinition::back_face` describes transforming cards, not independently
unlocked split permanents. `ManaUse` represents subtype spell/activation
purposes, not unlocking or turning face up. Surveil currently keeps remaining
cards in their existing order: DSK's surveil-two cards need a real order choice.
`SourceTapped` reads only a live source in `flow/statics.rs`; survival needs
source LKI when the source has departed. Dynamic static P/T is also narrower
than the general `EffectValue` vocabulary.

MTG241 execution/selection/replacement/context work, MTG242 mana, MTG243 naming,
and MTG244 layers are dependencies, not four new DSK implementations. This
review creates no Tasks and does not infer those Tasks have already delivered.

## Candidate batches and greatest challenges

| Batch | Cards | Scope |
|---|---:|---|
| DSK-01 | 63 | Existing definitions and composition candidates |
| DSK-02 | 23 | Events, history, counters and survival |
| DSK-03 | 36 | Costs, selections, contextual values and continuations |
| DSK-04 | 12 | Characteristics, copies, names and granted abilities |
| DSK-05 | 13 | Mana purposes, permissions and alternative costs |
| DSK-06 | 44 | Rooms and dependent eerie/unlock behavior |
| DSK-07 | 29 | Manifest dread and dependent hidden-object behavior |
| DSK-08 | 10 | Five impending Overlords and five Enduring creatures |
| DSK-09 | 20 | Replacements, entry snapshots and game outcomes |
| DSK-10 | 5 | Optional opening-hand Leyline placement |
| DSK-11 | 21 | Delirium predicates and distinct-type values |

Counts are primary assignments, not disjoint mechanic populations; dependencies
in the ledger deliberately cross batches.

**Rooms are the largest public-state extension.** One card needs two cast
profiles, per-door locked state, contextual names/mana value, unlock payments,
and lock/unlock instructions. Casting chooses one door; entry from elsewhere
starts locked. Unlocking is a special action, and fully-unlocked events differ
from enchantment entry. Test Bottomless Pool's cast, reanimation, unlock, lock,
and unlock-again paths; then Central Elevator/Promising Stairs name queries,
Marina's targeted choice and Inquisitive Glimmer's cost reduction. This should
join general profile/name authorities, not masquerade as transform.

**Manifest dread is the largest hidden-information extension.** Preserve an
underlying card and public unnamed 2/2 object, native private selection, exact
identity through face changes, inspection permissions, and face-up actions.
Test zero/one/two-card libraries, Paranormal Analyst's remembered graveyard
result, and Equipment attachment after suspended choice. Found Footage changes
who may inspect; Unable to Scream prohibits turning up. Hauntwoods Shrieker
requires reveal without necessarily turning up. Valgavoth's Onslaught retains
all created objects across repeated choices. Share this foundation with MKM's
face-down work while keeping each mechanic's payment and ward rules distinct.

**Provenance and characteristics connect the premium cycles.** Impending remains
a creature spell, but its paid-cost provenance controls entry counters and
battlefield type suppression; copying that permanent does not inherit its
payment. Enduring returns must remove creature status without erasing printed
abilities or following a later graveyard incarnation. Use Overlord of the
Hauntwoods, Enduring Vitality and Niko as witnesses for type changes, grants,
copy exceptions and delayed identity. Marvin requires dynamic activated-ability
grants with source binding, not definition-level ability copying.

**History, choices and replacements contain the strategic traps.** Survival
means the second main phase, not every postcombat main, and its tapped check
uses departure LKI. Delirium counts distinct card types, not cards or subtypes;
read it at the instruction's proper time. Victor already has a useful
resolution-count primitive, whereas Irreverent Gremlin's optional use is not a
once-trigger limit. Miasma Demon needs discard before reflexive targeting;
Monstrous Emergence retains a cast-cost selection and reads its power in the
proper context. Meathook Massacre II needs payer-specific life choices and
simultaneous sacrifice handling. Mindskinner, Rollercrusher Ride and Screaming
Nemesis exercise damage/prevention/outcome ordering. These are strong MTG241
acceptance witnesses.

## Closure, observability and performance

Close both Room faces and token dependencies before claiming set coverage:
Toy, Glimmer, white flying Spirit, flying Insect, restricted Beast, Horror,
Demon, Gremlin, green Spider, multicolor Insect, blue variable-size Spirit,
Everywhere, Primo and Shard. Treasure exists. Existing token names are not proof
of equivalent definitions: the current Spirit is a different token. Reflection
and Balloon are copy exceptions, not substitute generic tokens; Kaito's emblem
is another independent rules object. Basic land types on Everywhere must
supply real mana abilities.

Implement observable state with each mechanic: public door statuses, exact
face-down object joins, owner/permission-limited underlying identity, announced
cost provenance, counters, delayed subjects and live payment purposes. Carry it
through native/Python tensors, semantic capture, text, learning tokens, replay
and search forks. Preserve frozen input layouts; any genuinely new public
state needs a declared input-version boundary rather than silently absent data.

Start with composition and shared foundations, then bounded Room, manifest and
cycle slices. Pay particular attention to repeated characteristic reads,
granted-mana enumeration, trigger multiplication, multi-object selection and
hidden-card branching. Add small representative search-reply and ordinary-step
workloads with exact outcome digests; use existing same-runner Engine Budget
checks and track cumulative cost. No games, benchmarks, stress or production
changes were run here. Identity/name/face equality and unique row coverage were
validated offline; card-specific correctness remains proposed acceptance work.
