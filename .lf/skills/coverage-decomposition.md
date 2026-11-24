---
requires: a working card batch and its behavior checks
produces: a decomposition plan in `scratch/plan.md`, cut into slices
action_style: exploratory
---
Find the reusable shape that lets further cards in this area compose without new engine code.

1. Map the batch's selectors, rules types, observation and token projections
   and their owners. Pick real consumers or interactions that expose
   duplicated rules or special paths; don't invent future mechanics.
2. Write a **Delete — do not maintain** list, deepest obsolete owner first,
   and group the work into slices: one shared mechanism and every caller it
   moves. Each slice cuts real consumers over together and keeps directed
   behavior tests on the surviving owner.
3. State the review criterion: named representative cards compose from the
   shared mechanism with no card-specific engine logic or duplicate state.
   Focus on growing the card file, not runtime speed. If the current
   structure already meets that bar, say why and plan no slices.
4. A semantic counterexample stops dependent refactoring: fix the rules
   contract first rather than hiding it behind a cleaner API.
