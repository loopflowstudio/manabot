# Proposed events and replacement/prevention v1

This document defines Etude's first curated replacement-effects boundary.
It is intentionally smaller than the general Magic rules in CR 614-616.

## Lifecycle

Gameplay code submits a typed `ProposedEvent` before mutating `GameState`:

1. Validate the proposal's player and exact `ObjectRef` identities.
2. Collect matching replacement/prevention definitions from immutable card
   definitions.
3. Sort candidates by `(source ObjectRef, definition index)` and apply each
   collected candidate once.
4. Commit the surviving event to `GameState`.
5. Emit the existing factual `GameEvent`s; triggers, observations, and
   presentation continue to consume only committed facts.

The supported proposals are damage, signed life change, destruction, +1/+1
counter change, and zone movement. A destruction commit submits its resulting
battlefield-to-graveyard move through the zone proposal, preserving the exact
incarnation/LKI boundary from CR 400.7.

Player damage commits the life loss derived from that damage inside the damage
event. It is not proposed a second time as an independent life change, so
damage prevention is applied exactly once. Direct life gain/loss and lifelink
use the life-change proposal.

## Curated replacement definitions

Version 1 supports only:

- prevent N damage to a replacement source's controller or their permanent;
- double that damage (an ordering driver);
- this object enters the battlefield tapped;
- this object enters with N +1/+1 counters.

Entry properties are installed before the permanent is published and before
the committed `CardMoved` event is processed. Entering tapped does not emit a
`PermanentTapped` event.

No production card is added by this slice. Trace fixtures attach these
declarative meanings to scenario-local driver definitions through the existing
copy-on-write test seam.

## Determinism and absent states

Candidate order depends only on exact source identity and definition order,
never hash-map iteration, RNG, a state clone, or presentation order. A stale or
wrong-zone object proposal commits nothing. A fully prevented damage proposal
emits neither `DamageDealt` nor its derived `LifeChanged` event. An empty
replacement set follows the same committed event path and external event
shapes as before this pipeline.

## Unsupported CR 616 breadth

CR 616.1 ordinarily lets the affected player or affected object's controller
choose among multiple applicable replacement/prevention effects. Version 1
does not yield a choice. It uses the deterministic order above so replay,
training, and search remain exact while the choice ABI is designed separately.

Version 1 also does not support:

- recomputing applicability after each transformation;
- a replacement applying to one event more than once;
- dependencies between replacement effects;
- replacement effects that create another replacement decision;
- simultaneous-event ordering;
- redirects, consumable "next N" shields, draw/token/cost replacements, or
  broad replacement-generated event trees.

Cards needing any excluded behavior must remain rejected or unsupported at
content-build time rather than approximated through this pipeline.

