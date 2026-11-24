---
requires: a built card batch, its target interactions and focused check results
produces: behavior findings in `scratch/plan.md`
action_style: procedural
---
Check that the whole card batch plays by the rules for every consumer that reads it.

1. Review the batch as built, not only the last slice. Trace representative
   interactions from each slice through definitions, rules decisions,
   complete offers, tensor and text observations and learning tokens. Check
   outcomes against the rules, not against what the code does. When payment
   changed, exercise competing legal spends: auto spends the most restricted
   applicable pooled mana before ordinary mana, manual keeps meaningful pip
   and spend choices, and the receipt separates committed contributions from
   floating mana.
2. Check debug invariants, Commands, stale rejection, replay, forks and
   viewer privacy. Behavior a deletion removed must live on the surviving
   path, with no card-specific parallel authority and no silent choice
   truncation.
3. Reuse the current head's focused test results; run a constructed fixture
   only to settle an open question. Don't run full games or training checks.
4. Write each defect into `scratch/plan.md` with its narrowest reproduction
   and required fix, replacing earlier findings rather than adding a second
   report. Don't edit code here.

When deciding whether to continue:

- **Iterate** with the defects and unbuilt slices while semantics are missing
  or broken; compiling code with missing rules is not done.
- **Advance** when every planned slice is built and its interactions hold.
