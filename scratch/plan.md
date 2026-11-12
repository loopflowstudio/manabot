# MTG-303 plan: regime functions over a Run handle

## Steers (Jack Heart, 2026-11-12, verbatim where quoted)

- Stacked Task MTG-304 (`/Users/jack/src/manabot.training-library`) owns
  `training/learning/` and `rollout/`; this Task keeps `Run`, stage functions
  and execution control flow. "Call the existing learning functions through
  their current names where you can, and commit in small steps."
- Rename rehearse → smoke everywhere: "maybe `smoke` instead of rehearse?"
  `smoke(regime)`, `manabot train --smoke`, `TrainingJob.smoke()`. Etude
  EXP-49 makes `etude launch` call a job file's top-level `smoke()` locally
  and refuse on failure; job files define `smoke()` calling the regime's smoke.
- Drop `RegimeCall`: "not sure I love \"RegimeCall\" that feels a little
  frameworky to me". A regime is `fn(run, **arguments)`; bound arguments are
  `functools.partial(fn, **arguments)`. `execute(regime, out=, seed=,
  resume_from=)` / `smoke(regime)` take any callable accepting a run. The run
  records the callable's module:qualname, the partial's JSON arguments
  (hashed), the source revision, and each stage function records its own
  resolved arguments in its stage record. Remote jobs carry function path +
  JSON arguments (`TrainingJob(regime=...)` accepts the callable or partial).
  Docs: arms as a dict of partials.
- Positioning: "we write (multiple, but not too many) loops that are our SOTA
  - those loops are made out of composable pieces - you can write your own
  learner by taking bits and pieces and putting them into your own outer loop
  or using etude like we do". Docs present regime functions as how our
  maintained loops run and are recorded; writing your own is secondary.

## Built

- `manabot/training/run.py`: `Run` (record, persist/export, clocks, seeds,
  monitoring, failure settlement), `StageScope`, `Model`, `open_run`,
  `attempt` (lease + resume admission shared by both recipe kinds).
- `manabot/training/stages.py`: stage bodies extracted from the old
  `_execute_regime`; `run_saved_regime`; public `self_play`, `import_policy`,
  `collect_belief`, `train_belief`.
- `execution.py`: `validate_regime`, `_runtime_identities`, `execute_regime`
  (attempt → open_run → run_saved_regime). Old 1,360-line body deleted.
- Records: `RunRecord.regime` optional + `regime_call` (exactly one);
  `StageRecord.operation` for function-run stages; `recovery_policy`,
  `stage_operations()`.
- Saved-regime suites pass after retargeting monkeypatches to `run`/`stages`
  (17 training modules; one abrupt-exit failure was a mid-run commit moving
  `source_commit`, not code).

## Remaining

1. Function layer per steers: `execute(regime, ...)`, `smoke(regime)`,
   identity from callable/partial (draft in /tmp/fn_layer.py used RegimeCall —
   rework); stage records carry resolved stage arguments.
2. `tests/training/test_regime_functions.py` (written against RegimeCall API;
   rework to partials): saved≡function equivalence, two models, smoke,
   resume at an update boundary.
3. CLI `--function/--arguments/--smoke`; `TrainingJob(regime=callable)`,
   `TrainingJob.smoke()`; one experiment file as a function (outcome 6).
4. Docs: rename, partials, positioning; `RunPaused` → `TrainingPaused`.
5. Gate: training suites, package boundaries, regime serialization tests.

## Not in scope / deferred

- Per-game result rows (MTG-282): docs do not promise them.
- Function runs as producers for `import_policy`/learning-state admission
  (admission reads `source.regime`); game-error recovery in function runs.
