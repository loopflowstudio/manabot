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

## Built (commits on this branch)

- `manabot/training/run.py`: `Run` (record, persist/export, clocks, seeds,
  monitoring, failure settlement), `StageScope`, `Model`, `open_run`,
  `attempt` (lease + resume admission shared by both recipe kinds);
  `execute(regime, out=, seed=, resume_from=, recovery=, ...)`,
  `smoke(regime, out=)`, `regime_record`, `load_regime`, `RunResult`.
  A regime is any module-level function or `functools.partial` of one.
- `manabot/training/stages.py`: stage bodies extracted from the old
  `_execute_regime`; `run_saved_regime`, `smoke_form`; public `self_play`,
  `import_policy`, `collect_belief`, `train_belief`. Function stages build a
  one-stage `TrainingRegime` context validated by `validate_regime`.
- `execution.py`: `validate_regime`, `_runtime_identities`, `execute_regime`.
- Records: `RunRecord.regime` optional + `regime_function`
  (`RegimeFunctionRecord`: function, arguments, sha, `SourceRevision`,
  wall_seconds, recovery); `StageRecord.operation` and `.arguments`.
- CLI `--function/--arguments/--smoke` (smoke also for saved regimes);
  `TrainingJob.smoke(out)`; `experiments/two_models.py`.
- `tests/training/test_regime_functions.py`: saved≡function bit-exact
  (initial_raw/raw/ema), two models end to end, argument checks, smoke,
  resume at an update boundary bit-exact, saved-regime smoke.
- Docs rewritten for partials, smoke and the positioning steer.

## Remaining

1. Gate (module by module; whole-suite run stalled on a busy host): every
   module in tests/training, test_package_boundaries, test_experiment_files,
   test_attack_choice and test_shared_effects passes except failures outside
   this diff: test_etude_evaluation (4) and test_head_to_head (2)
   `No module named 'duckdb'`; test_game_errors (3) fixture replay
   `binding_mismatch` (diff touches no env/, game_errors, managym, sim or
   fixtures); test_training_throughput::test_a_packed_seed_digests_as_it_does_solo
   (checkpoint `source/etude_run`, added in #464, is not in the packing
   digest's volatile list). Re-run these on a quiet host before landing.
2. `TrainingJob(regime=callable)`: jobs are built around one saved
   self-play stage (Etude `until=updates=N`, continuations, stop reserve);
   a function job needs a declared update target. Docs say not yet.
3. Outcome 6 (an existing Etude experiment file as a function) waits on 2.
4. Function runs as producers for `import_policy` / learning-state
   admission; game-error tolerance in function runs.

## Not in scope / deferred

- Per-game result rows (MTG-282): docs do not promise them.
