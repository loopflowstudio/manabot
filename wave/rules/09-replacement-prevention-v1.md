# 09: Replacement/Prevention v1

## Finish line

Damage replacement/prevention entry points exist, hooking into the event
system from stage 02, and are validated in focused interaction tests.

## Changes

### Replacement Effect Hooks

Replacement effects intercept `GameEvent`s before they execute. The event
system gains a pre-event check: "does any replacement effect apply to this
event?"

Focus on damage replacement/prevention first (CR 614, 615):
- "Prevent the next N damage" shields
- "If ~ would be dealt damage, prevent that damage" static prevention
- "Instead" replacement effects for damage

### ETB Replacement Effects

Common patterns that modify how permanents enter:
- "Enters tapped" (CR 614.1c)
- "Enters with N +1/+1 counters" (CR 614.1c)

These hook into `GameEvent::CardMoved` with `to: Zone::Battlefield`.

### Choice/Ordering

When multiple replacement effects apply to the same event, the affected
object's controller chooses order (CR 616). Support basic single-replacement
first, then multi-replacement ordering.

### Trace Tests

- Damage prevention shield absorbs damage
- "Enters tapped" modifies ETB
- Positive and negative: replacement applies only to matching events
- At least one multi-effect interaction test

## Done when

- Damage replacement/prevention examples run correctly.
- ETB replacement effects ("enters tapped") work.
- Trace tests cover at least one multi-effect interaction.
- Training smoke test passes.
