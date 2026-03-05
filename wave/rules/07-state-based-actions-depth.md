# 07: State-Based Actions Depth

## Finish line

SBA handling is expanded beyond the current subset and reliably integrated
with trigger/priority loops, including simultaneous-loss edge cases.

## Changes

### Expand SBA checks

Add missing SBAs needed by current and near-future card pool:

| SBA | Rule | Priority |
|-----|------|----------|
| Legend rule (2+ legendaries same name) | 704.5j | High — needed for any legendary cards |
| 0 toughness to graveyard | 704.5f | High — needed for -X/-X effects |
| Token cleanup (non-battlefield) | 704.5d | High — needed once tokens exist |
| +1/+1 and -1/-1 counter cancellation | 704.5q | Medium — needed when both counter types coexist |
| Aura falls off (illegal/unattached) | 704.5m | Medium — needed for Aura support |
| Equipment unattach (illegal) | 704.5n | Medium — needed for Equipment |

### SBA loop correctness (CR 704.3)

Verify the iterative checking process: SBAs checked repeatedly until none
are performed, then triggered abilities placed, then checked again, before
priority is granted.

### Edge cases

- Simultaneous terminal conditions (both players at 0 life = draw)
- SBA triggers chaining (creature dies → death trigger → SBA check again)
- Multiple SBAs in same check (creature with lethal damage AND 0 toughness)

### Trace tests

- Each new SBA: positive and negative path
- Simultaneous loss → draw
- SBA → trigger → SBA chain
- Legend rule with multiple copies

## Done when

- SBA loop behavior matches documented supported CR subset.
- Edge-case SBA trace tests pass and are rule-cited.
