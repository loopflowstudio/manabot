---
requires: a working card batch and its behavior checks
produces: the decomposition-phase plan in `scratch/plan.md`, cut into slices
---
Find the reusable shape that lets the next cards in this area compose without new engine code.

1. Map the batch's selectors, rules types, observation and token projections
   and their owners. Pick real consumers or interactions that expose
   duplicated rules or special paths; don't invent future mechanics.
2. Write the **Delete — do not maintain** list, deepest obsolete owner first,
   and group it into slices: one shared mechanism and every caller it moves.
   Each slice moves its real consumers in the same cut and keeps the directed
   behavior tests on the surviving owner.
3. Set the review criterion: named representative cards compose from the
   shared mechanism with no card-specific engine logic or duplicate state.
   This phase is about growing the card file, not runtime speed. If the
   structure already meets it, say why and plan no slices.
4. A semantic counterexample sends the work back to the working contract
   before any refactoring on top of it; don't hide it behind a cleaner API.
