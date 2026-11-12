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
- `manabot/training/stages.py`: public `self_play`, `import_policy`,
  `collect_selection`, `collect_belief`, `train_belief`, each doing its own
  work after `_open_stage(run, declaration, ...)`; only self-play keeps a
  private loop (`_train_self_play`). `run_saved_regime` calls them (see the
  concept-review follow-up below).
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

Outcomes 1-5 and 7 are met (Outcome 4 since `eeca3f374`/`dcd13b7c9`: one
caller per stage body; Outcome 7 on resume since `32dd45e43`/`242df4246`). Outcome 6 waits on the open Etude choice below.

1. Gate (module by module; a whole-suite run stalled on a busy host). Known
   failures outside this diff, to re-run on a quiet host before landing:
   test_etude_evaluation (4) and test_head_to_head (2) `No module named
   'duckdb'`; test_game_errors (3) fixture replay `binding_mismatch` (diff
   touches no env/, game_errors, managym, sim or fixtures);
   test_training_throughput::test_a_packed_seed_digests_as_it_does_solo
   (checkpoint `source/etude_run`, added in #464, is not in the packing
   digest's volatile list).
2. `TrainingJob(regime=callable)`: jobs are built around one saved
   self-play stage (Etude `until=updates=N`, continuations, stop reserve);
   a function job needs a declared update target. Docs say not yet. Waits
   on the open Etude choice below.
3. Outcome 6 (an existing Etude experiment file as a function) waits on 2.
4. Function runs as producers: `admit_policy` and `admit_learning_state`
   read `source.regime` and refuse a function-run producer by name
   ("regime-function run", `test_a_regime_function_producer_is_refused_by_name`).
   Admitting them needs a producer contract over stage records instead of
   the saved regime. Game-error tolerance in function runs is unbuilt.

## Not in scope / deferred

- Per-game result rows: #466 (MTG-282) merged without them ("Take the
  training-side result rows out of deck-building foundations"), so nothing
  exists to route through `Run`. Per-update outcome rows go through Etude
  `log_table`; docs do not promise per-game rows.

## Concept review (2026-11-12, against `6700caf3a`) and its follow-up

The review found two callers per stage body: `run_saved_regime` dispatched
to the private bodies while public functions built a one-stage
`TrainingRegime` (`_stage_regime`). Done since (`eeca3f374`, `dcd13b7c9`):

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
5. A model's self-play lineage is `Model.trained_by`; the open learner is
   `Run.live_session` (see concept review 2).

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

## Concept review 2 (2026-11-12, against `d4271e1fc`)

One implementation per stage holds; the remaining mismatch is that a
model's continuation lineage is derived from live state. `self_play` sets
`TrainSelfPlay.initial` from `model.session`, and `_restore_model` clears
the session on resume, so a resumed function run records a continued stage
as fresh. Probe (two consecutive `self_play` calls on one model, interrupted
after update 1 of the second, `RecoveryPolicy(checkpoint_updates=1)`, seed 5):
raw weights bit-equal to the uninterrupted run, but stage 2
`arguments.settings.initial` was `"self-play"` uninterrupted and `None`
resumed. That breaks Outcome 7's comparable resolved arguments.

Done since: `Model.trained_by` holds the last self-play stage that trained
the model; `self_play` and `_restore_model` set it and `initial` reads it.
`Model.session` is gone: a stage continues live when `Run.live_session` is
the stage named by `trained_by`, else from this stage's recovery snapshot,
else it is refused by name (also covers a model trained in non-consecutive
stages). The parent attempt checked the continuation, so the snapshot path
does not check it again. Tests:
`test_a_resumed_continuation_records_and_trains_as_uninterrupted` (the
probe: equal stage arguments, bit-equal raw weights) and
`test_a_continuation_without_learner_or_snapshot_is_refused` (orphaned and
interleaved). Check: test_regime_functions, test_regimes, test_recovery,
test_active_recovery, test_belief_stages, test_package_boundaries: 61 passed.

Unchanged: the open Etude budget choice and the `run.stage` resume limit.

## Concept review 3 (2026-11-12, against `7fcacc3e5`)

Lineage and one-caller-per-stage hold. Remaining mismatch: a `Model` is
where a run's weights live, but public `self_play` also takes three
starting states (`initial_weights`, `pretrained`, `learning_state`) that
the body loads over `initial=model.network` without looking at the model.
In a function regime, `m = import_policy(run, "m", ...)` then
`self_play(run, m, pretrained=...)` trains the pretrained weights while the
stage records `import-m`'s artifacts as its input (`producer=model.source`),
and `initial_weights`/`learning_state` behave the same way. Saved regimes
never hit this: each fresh stage gets a new `run.model`, and
`TrainingRegime` validation already refuses `initial` with another starting
state.

Done: `self_play` refuses the three starting states on a model with
`source` or `trained_by` set, naming model, argument and stage, before
`_open_stage`; docs say so. Test
`test_a_starting_state_over_a_model_with_weights_is_refused` uses real
paths (exported fresh model + `pretrained`; trained model +
`initial_weights`) rather than a faked import. Check: regime_functions,
regimes, policy_import, learning_state, recovery, package_boundaries — 87
passed.

No saved-regime record, digest or stage id changes. Unchanged: the open
Etude budget choice and the `run.stage` resume limit.
