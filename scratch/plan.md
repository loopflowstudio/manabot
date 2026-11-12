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

## Concept review (2026-11-12, headless, against `6700caf3a`)

The interaction holds: a regime is `fn(run, **kwargs)` or a partial; `execute`
/ `smoke` take it; models are `run.model(...)`; stage functions return typed
results and replay their records on resume. The biggest mismatch is inside:

**Saved regimes do not call the stage functions.** `run_saved_regime`
dispatches on stage kind straight to the private bodies (`_train_self_play`,
`_import_policy`, ...), and public `self_play` / `import_policy` go the other
way: they build a `TrainSelfPlay` and a one-stage `TrainingRegime`
(`_stage_regime`) so the same bodies can read schema. So there are two
callers of each body and the configuration is still the stage interface.
Outcome 4 asks for the opposite direction ("calls the same stage functions";
"the old one calls the new one").

### Next implement direction (implied by the design, no product choice)

1. `run_saved_regime(run, regime)` becomes an ordinary regime function: one
   `run.model("agent", ...)` (or `import_policy` for an `initial` import),
   then one public stage-function call per saved stage, passing the saved
   stage's `id` as `stage=`, `seed=run.seeds["initialization"]`, its
   execution and settings. A saved stage whose `initial` names the previous
   self-play stage is the same model trained again (live continuation). Add
   a public `collect_selection` so every saved kind has a function.
   Records stay as they are: `RunRecord.regime` and its digest still name
   the saved regime, so receipts and evidence are unchanged.
2. Stage bodies take what they read (decks, agent, observation, the stage
   settings record, schedule clock) instead of a `TrainingRegime`; delete
   `_stage_regime`. Keep `validate_regime` for saved files, and run the
   per-stage deck/world/curriculum admission it does from the stage
   function itself. The schedule clock is a stage argument (saved:
   `run_elapsed_budget` default; function: `iteration_fraction`).
3. Input-artifact digest checks (`initial`, `policy`, `dataset`) move into
   the stage functions that read those inputs, so function regimes get them
   too.
4. `TrainingJob.smoke` goes through `execute_regime`'s smoke path, so a smoke
   of a saved job records the saved regime's identity rather than a function
   record whose argument is the whole regime JSON.
5. Small: `Model.session: SelfPlaySession | None` (TYPE_CHECKING import)
   instead of `Any`.

Proof stays the existing bit-exact saved≡function test plus the saved-regime
recovery and smoke suites; every saved fixture must keep its record digest.

Counterexample to check while doing 1: a saved regime whose second self-play
stage has `initial` naming an *earlier, non-adjacent* stage (fresh learner
from those weights, not a continuation). It maps to a new model adopted from
that stage's export, not to training the same `Model` again.

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
