# 08: Illegal Actions + Rollback Boundaries

## Finish line

Illegal plays are handled more robustly beyond index bounds, with explicit
rollback/documented limits.

## Changes

- Extend illegal-action detection for target/timing/cost invalidity.
- Add rollback for feasible partial actions.
- Document non-rollback boundaries to avoid hidden behavior.
- Add adversarial tests for illegal sequencing.

## Done when

- Illegal action handling is predictable and tested.
- Deviation/limit notes are visible in coverage docs.
