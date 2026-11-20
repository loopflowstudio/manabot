---
requires: implemented cohort, accepted interactions and focused debug evidence
produces: behavior findings in the existing scratch design
---
Review whether the batch implements its intended rules for every shared consumer.

1. Trace representative card interactions through definitions, rules decisions,
   complete offers, tensor/text observations and tokens. Check the tested
   outcomes against the rules. When payment changes, exercise competing legal
   spends. Verify that auto spends the most restricted applicable pooled mana
   before ordinary mana, that manual retains meaningful pip and spend choices,
   and that the receipt distinguishes committed contributions from floating mana.
2. Check debug invariants, Commands, stale rejection, replay/forks and privacy.
   Ensure required behavior survived deletions and no card-specific parallel
   authority or silent choice truncation replaced it.
3. Record concrete defects with the narrowest reproduction and required fix.
   Distinguish focused implementation evidence from suites owned by gate.
   State whether the working phase is satisfied and what remains for later phases.
   Update existing findings; do not create a second report.

The following navigation decision uses these findings. Missing semantics cannot
advance just because code compiles. No games or learning claims follow from this
review; use constructed fixtures for rules checks.
