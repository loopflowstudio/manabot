# 11: Rules-Driver Cardset Expansion

## Finish line

A curated card pack exists to exercise each implemented rule family with both
happy-path and adversarial interactions.

## Changes

- Add cards specifically to stress priority/stack, triggers, SBA, layers,
  replacement.
- For each new family, include at least one pairwise interaction test.
- Link cards to rule families in coverage notes.
- Validate DSL expressiveness — can all driver cards be represented
  declaratively? Document any that require special handling.

## Done when

- Each implemented family has at least one driver card + interaction trace test.
- Test suite demonstrates cross-family behavior, not only isolated mechanics.
- DSL coverage gaps (if any) documented.
