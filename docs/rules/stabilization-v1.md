# Post-commit stabilization v1

This contract defines the deterministic rules boundary used immediately before
the engine publishes priority.

## Fixed point

`Game::stabilize_before_priority` is the sole pre-priority authority for the
supported CR 117.5 / 704.3 sequence:

1. collect triggers only from committed `GameEvent` facts;
2. discover and commit one simultaneous batch of supported state-based actions;
3. if any SBA occurred, return to step 1 and check the full SBA set again;
4. put waiting triggers on the stack in deterministic APNAP/enqueue order;
5. return to step 1, then publish priority only when no event, SBA, or waiting
   trigger remains.

A required trigger target choice may suspend the loop. Completing the choice
re-enters stabilization through the ordinary atomic action boundary before any
priority offer is published.

## Identity and simultaneous commits

SBA discovery converts live permanent storage slots to exact `ObjectRef`
values. Later commits reject stale incarnations instead of following the same
physical card after a zone change. One-shot delayed triggers likewise watch an
exact object, not a bare card storage ID.

All supported SBAs discovered in one check share a sorted pre-batch trigger
source snapshot. Their factual `GameEvent` projections are committed in
`ObjectRef` order, while leave-the-battlefield trigger matching sees the same
pre-batch abilities for every simultaneous departure (CR 603.6c, 704.3).

## Compatibility boundary

The fixed action table, observation tensor/data objects, presentation protocol,
legacy `GameEvent` shapes, and registered card definitions are unchanged.
`CardId` and `PermanentId` remain compatibility projections at those external
surfaces; exact refs are used inside the touched rules continuations.

## Deliberate exclusions

This slice adds no layer machinery, card registrations, replacement families,
legend rule, attachment rule, counter-cancellation rule, or general
simultaneous-event choice ABI. It stabilizes only the SBA families already
implemented by the engine.
