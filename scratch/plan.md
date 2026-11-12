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
  A regime is any module-level function or `functools.partial` of one;
  `execute` calls it with the annotation-validated arguments it records.
  Saved and function stages shrink through one `stages.smoke_stage`.
- `manabot/training/stages.py`: stage bodies extracted from the old
  `_execute_regime`; public `self_play`, `import_policy`,
  `collect_selection`, `collect_belief`, `train_belief`; `run_saved_regime`
  calls them (see the concept-review follow-up below).
- `execution.py`: `validate_regime`, `_runtime_identities`, `execute_regime`.
- Records: `RunRecord.regime` optional + `regime_function`
  (`RegimeFunctionRecord`: function, arguments, sha, `SourceRevision`,
  wall_seconds, recovery); `StageRecord.operation` and `.arguments`.
- CLI `--function/--arguments/--smoke` (saved smoke via
  `execute_regime(smoke_execution=)`); `TrainingJob.smoke(out)`; `experiments/two_models.py`.
- `tests/training/test_regime_functions.py`: saved≡function bit-exact
  (initial_raw/raw/ema), two models end to end, argument checks, smoke,
  resume at an update boundary bit-exact, saved-regime smoke.
- Docs rewritten for partials, smoke and the positioning steer.
- Main merged at `e5897c7ab`: MTG-312 `Execution.bucketed_shapes` and MTG-311
  per-update `experiment.log_table("self_play_outcomes", ...)` live in
  `stages.py`, so function `self_play` takes both through `execution=` and
  the shared stage body; `test_bucketed_execution` now patches the stage
  bodies (`8093e1bc3`).

## Remaining

1. Gate (module by module; whole-suite run stalled on a busy host): every
   module in tests/training, test_package_boundaries, test_experiment_files,
   test_attack_choice and test_shared_effects passes except failures outside
   this diff: test_etude_evaluation (4) and test_head_to_head (2)
   `No module named 'duckdb'`; test_game_errors (3) fixture replay
   `binding_mismatch` (diff touches no env/, game_errors, managym, sim or
   fixtures); test_training_throughput::test_a_packed_seed_digests_as_it_does_solo
   (checkpoint `source/etude_run`, added in #464, is not in the packing
   digest's volatile list). Re-run these on a quiet host before landing, plus
   test_bucketed_execution and the regime suites after the main merge.
2. `TrainingJob(regime=callable)`: jobs are built around one saved
   self-play stage (Etude `until=updates=N`, continuations, stop reserve);
   a function job needs a declared update target. Docs say not yet.
3. Outcome 6 (an existing Etude experiment file as a function) waits on 2.
4. Function runs as producers for `import_policy` / learning-state
   admission; game-error tolerance in function runs.

## Not in scope / deferred

- Per-game result rows: #466 (MTG-282) merged without them ("Take the
  training-side result rows out of deck-building foundations"), so nothing
  exists to route through `Run`. Per-update outcome rows go through Etude
  `log_table`; docs do not promise per-game rows.

## Concept review (2026-11-12, against `6700caf3a`) and its follow-up

The review found two callers per stage body: `run_saved_regime` dispatched
to the private bodies while public functions built a one-stage
`TrainingRegime` (`_stage_regime`). Done in the next implement pass:

1. `run_saved_regime` is an ordinary regime function: one public call per
   saved stage (`stage=<id>`, its settings, the regime's `schedule_clock`).
   Fresh self-play stages get `run.model(<id>, ..., seed=seeds["initialization"])`;
   `initial` reuses that stage's `Model` (validation admits only the latest
   self-play stage, so this always continues live, also across intervening
   belief stages). A belief collection naming a self-play stage whose model a
   later stage continued reads that stage's own export (`_published_model`).
   Public `collect_selection` added. `RunRecord.regime` and digest unchanged.
2. Bodies take decks, model config, observation, settings and schedule clock;
   `_stage_regime` and `smoke_form` are gone. `admit_self_play` (stages.py,
   rosters/adaptive/deck source) and `check_self_play` (regime.py, the
   per-stage rules `TrainingRegime` used to inline) are called by both
   `validate_regime` and `self_play`. `admit_policy` / `admit_learning_state`
   take `decks, agent, observation[, schedule_clock]`. `Run` carries
   `recovery_max_microsteps` / `recovery_every_updates`.
3. Input digest checks live in the stage functions (`_record_inputs` over
   `Model.source`, the stage that published the model's weights, or the
   `BeliefCorpus` stage). `collect_belief` returns `BeliefCorpus`;
   `train_belief` reads the verified dataset back from disk.
4. `execute_regime(..., smoke_execution=)`; `--smoke` on a saved regime and
   `TrainingJob.smoke` (now returns `TrainingRun`) record the saved regime.
5. `Model.session: SelfPlaySession | None`.

Checks: test_regime_functions, test_regimes, test_belief_stages (+ a
continued-policy test), test_policy_import, test_learning_state,
test_bucketed_execution, test_frozen_opponent, test_training_decks,
test_recovery, test_active_recovery, test_active_training, test_etude_job,
test_game_error_recovery, test_mps_training, test_baseline_recipe,
test_phase_timing, test_step_cadence, test_numerical_health,
test_ataraxos_regime, test_automatic_game_capture, test_action_cutoff,
test_process_tree_rss, test_time_cohort, test_monitoring,
test_package_boundaries, test_experiment_files: all pass except the known
test_game_errors `binding_mismatch` (3).

### Open product choice (does not block 1-5)

- **How a function job declares its Etude budget.** Remote jobs are built on
  one saved self-play stage with `until=updates=N`, continuations and a stop
  reserve. A function regime has no single update count. Options: the job
  passes an `updates` argument the function must accept (a naming
  convention), Etude stops function jobs on wall/USD only and they resume at
  update boundaries, or the function returns its planned updates from a
  smoke run. This decides what `TrainingJob(regime=callable)` and Outcome 6
  look like; it is an Etude contract question, so it likely belongs in an
  Etude PR.
- Known limit, not a choice yet: `run.stage(name)` custom stages refuse
  resume past them, so a regime that uses one cannot recover after it.
