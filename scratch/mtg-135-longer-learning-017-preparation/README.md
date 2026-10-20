# MTG-135 protocol017 handoff preparation

Prepared 2026-10-18T19:31:49.623265+00:00. **No architecture has been selected.** The full 22-trajectory,
18,304-game panel is still running on the Mini. This package delivers exact candidate
recipes and static compatibility findings; it is not a scored recommendation,
launch request, new execution hold, or change to MTG-122's budget/cohort.

The eventual final recommendation will report the best supported 64-stream option
explicitly. A 256-stream outcome is an alternative, not an automatic replacement.
MTG-122's own latest accepted direction still governs: GPU training, Mini assessment,
all 36,000 games and frozen controls, separate $35 inclusive cap, no laptop science,
and no displacement of the existing Mini cohort.

## Exact tested clocks

Every 64-stream candidate uses 64 learner transitions per stream per collection/
learning iteration: 4,096 transitions per iteration. The tested full horizon is
524,288 transitions = 128 iterations. These iterations are not Adam optimizer
steps. Raw checkpoints are retained at iterations 0,32,64,128, corresponding to
0,131,072,262,144,524,288 transitions. The learner can skip empty timestep groups;
actual Adam steps and selected exposures must remain measured, separate counters.

The learning schedule maps iteration i to r(i)=1+(i-1)*2047/127. Learning rate
stays 1e-4 through the tested horizon; tau falls from 0.05 to
0.005076577477226473. EMA is 0.999 per iteration, not per transition. Preserve the
whole tested recipe; an earlier competence crossing does not authorize truncating
it or keeping 128 iterations with a different schedule.

## Receiver compatibility findings

The receiver was inspected at commit `8e7042517450e9a547fb353d95f77facc864290d` with the exact working-file
hashes in [compatibility.json](compatibility.json).

1. `AtaraxosMoveLearning` has no `iteration_schedule` field in this checkout.
   Importing the tested JSON therefore requires the tested schedule implementation
   and its source integration. Do not strip the field to make a recipe load.
   At iteration 128 the unmapped tau would be 0.011662912394210, rather than
   the tested 0.005076577477226473.
2. `ComplexityPlan` requires at least 1,000 iterations in multiples of 100. It
   rejects the tested 128-iteration horizon. Keeping 1,000 would collect 4,096,000
   transitions, 7.8125 times the tested experience, with another schedule. A fresh,
   versioned plan must represent explicit transition/iteration coordinates;
   preserve historical plans instead of changing their meaning.
3. `environment_complexity.regime` reconstructs the old baseline and overrides
   streams/updates, so setting model width alone is not an exact recipe transfer.
   Carry the full selected architecture, observation and learner fields into one
   common recipe across all three pools. Rebind the shared research catalog and
   authored pool rosters, and initialize freshly. Product checkpoints and their
   identity buffers are not valid research-catalog initializations.
4. The current campaign executes evaluation inline. The accepted GPU-training/
   Mini-evaluation split and removal of inline diagnostics remain implementation
   and calibration work. Preserve initialization, midpoint and endpoint paired
   controls and all 36,000 games. A quarter-point export may be retained without
   adding a new scored cohort. Do not confuse removal of expensive inline probes
   with removal of cheap progress counters or scientific assessments.

[Tested source excerpts](tested-source/) retain the full exact source files for
the schedule models, grid recipe construction and original baseline at
`3c3681d2b26a7fc8167928fa63fe24a6cd3d058a`. They are context for integration, not instructions to overwrite
receiver modules or bypass normal content/runtime identity admission.

## What still has to happen

- MTG-135 must finish the full panel, verify every trace/replay/manifest, review
  the generated figures and apply the frozen selection rule. No partial ranking
  or unqualified competence claim belongs in this package.
- The final handoff must include all scores, both seeds, per-role results, exact
  chosen recipe, checkpoint/time/resource evidence, operational deviations and
  the separate best supported 64-stream choice.
- MTG-122 must integrate that recipe into its current source and calibrate all
  three pools with its actual GPU, shared research catalog and detached evaluation.
  RTX5090 inline-diagnostic timings are not L4 split-pipeline cost guarantees.
  A small fixed artifact reserve is headroom, not measured storage/transfer cost.
- A longer horizon, approximately 20k-update study, different optimizer grouping,
  or model-ceiling claim requires additional evidence. None is supplied here.

All eleven candidate JSON recipes are copied unchanged into [recipes](recipes/),
including their original index; the five 64-stream candidates share the clocks
above. [The frozen protocol](protocol017.md) remains unchanged. Its historical
placement/pause language is superseded only by the separately recorded user
placement/scope direction, not by silently rewriting scientific evidence.

The manifest binds every copied byte. This preparation does not itself prove
receiver runtime compatibility or any playing-strength outcome.
