# Manabot Linear identifiers — 2026-10-16

Jack Heart requested Linear issue identifiers for every Task using a local hexadecimal short code. Restored Intelligence’s existing Initiative binding from `wave/intelligence/GOAL.md` into the current `.lf/wave/intelligence/GOAL.md` configuration and synchronized the existing Tasks through `lf task sync`. No Task identity, checkout, execution, or completion state was replaced.

All 14 affected Tasks now have Linear identifiers and issue URLs; a fresh repository read found no remaining hexadecimal Task short codes. Updated the corresponding cmux labels and Session title prefixes.

| Previous code | Linear issue | Task |
| --- | --- | --- |
| fc765e | [MTG-137](https://linear.app/loopflow/issue/MTG-137/run-corrected-world-mini-and-gpu-baseline-studies) | Run corrected-world Mini and GPU baseline studies |
| 85fb3b | [MTG-138](https://linear.app/loopflow/issue/MTG-138/distinguish-strategic-regularization-from-update-control-in-manabot) | Distinguish strategic regularization from update control in Manabot |
| 399539 | [MTG-139](https://linear.app/loopflow/issue/MTG-139/test-whether-coordinated-schedules-or-exploration-cycles-improve) | Test whether coordinated schedules or exploration cycles improve durable learning |
| 82b6a9 | [MTG-140](https://linear.app/loopflow/issue/MTG-140/distinguish-strategic-cycling-from-durable-checkpoint-improvement) | Distinguish strategic cycling from durable checkpoint improvement |
| e0e4bd | [MTG-141](https://linear.app/loopflow/issue/MTG-141/evaluate-one-step-kl-and-magnetic-updates-for-policy-guided-search) | Evaluate one-step KL and magnetic updates for policy-guided search |
| c3be2f | [MTG-142](https://linear.app/loopflow/issue/MTG-142/call-the-play-app-academy-so-the-etude-experiment-code-can-be-imported) | Call the play app Academy so the Etude experiment code can be imported |
| 59dec9 | [MTG-143](https://linear.app/loopflow/issue/MTG-143/measure-how-advantage-filtering-changes-the-evidence-the-policy-learns) | Measure how advantage filtering changes the evidence the policy learns from |
| 197344 | [MTG-144](https://linear.app/loopflow/issue/MTG-144/stop-wall-clock-budgeted-training-tests-from-failing-on-slow-ci) | Stop wall-clock-budgeted training tests from failing on slow CI runners |
| 49cd27 | [MTG-145](https://linear.app/loopflow/issue/MTG-145/find-the-useful-balance-between-continuation-quality-and-search-depth) | Find the useful balance between continuation quality and search depth |
| 3d37a1 | [MTG-146](https://linear.app/loopflow/issue/MTG-146/separate-belief-calibration-from-downstream-search-strength) | Separate belief calibration from downstream search strength |
| 542346 | [MTG-147](https://linear.app/loopflow/issue/MTG-147/preserve-ordinary-action-and-object-identity) | Preserve ordinary action and object identity |
| a89e76 | [MTG-148](https://linear.app/loopflow/issue/MTG-148/establish-manabot-discord-for-nightly-learning-reports) | Establish manabot Discord for nightly learning reports |
| 6700bb | [MTG-149](https://linear.app/loopflow/issue/MTG-149/run-nightly-learnability-regressions-with-shared-models) | Run nightly learnability regressions with shared models |
| b2b530 | [MTG-150](https://linear.app/loopflow/issue/MTG-150/remove-wandb-entirely-now-that-etude-tracks-training) | Remove wandb entirely now that Etude tracks training |

The restored binding is a local, uncommitted repository configuration change. The broad `lf repo refresh intelligence` command still requires the Wave file on the default branch; individual Task synchronization succeeded.
