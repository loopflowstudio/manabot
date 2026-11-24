---
requires: the built card batch, its accepted interactions and focused checks
produces: behavior findings in `scratch/plan.md` and the working phase's loop decision
---
Check that the whole batch plays by the rules for every consumer that reads it, then decide whether the working phase is done.

1. Review the batch as built, not the last slice. Take representative
   interactions from each slice through definitions, rules decisions,
   complete offers, tensor and text observations and tokens, and check the
   outcomes against the rules, not against what the code does. When payment
   changed, exercise competing legal spends: auto spends the most restricted
   applicable pooled mana before ordinary mana, manual keeps meaningful pip and
   spend choices, and the receipt tells committed contributions from floating
   mana.
2. Check debug invariants, Commands, stale rejection, replay, forks and viewer
   privacy. Behavior a deletion removed lives on the surviving path; no
   card-specific parallel authority, no silent choice truncation.
3. Reuse implement's focused results for the current head; run only a check
   that settles an open question. Constructed fixtures, not games or learning
   runs.
4. Write each defect into the plan with its narrowest reproduction and the fix
   it needs, replacing earlier findings rather than adding a second report.

Iterate, naming the defects and unbuilt slices, while the phase isn't met;
code that compiles with missing semantics isn't met. Advance when every slice
is built and its interactions hold, and say what the decomposition and
performance phases inherit.
