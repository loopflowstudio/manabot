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

## Concept reviews (2026-11-12): all three mismatches resolved

1. Two callers per stage body (against `6700caf3a`): `run_saved_regime` is
   now an ordinary regime function, one public call per saved stage
   (`eeca3f374`, `dcd13b7c9`). `initial` reuses that stage's `Model`; a
   belief collection naming a continued self-play stage reads that stage's
   own export (`_published_model`). Per-stage rules live in `check_self_play`
   (regime.py) and `admit_self_play` (stages.py), called by both
   `validate_regime` and `self_play`; input digests are checked in the stage
   functions (`_record_inputs` over `Model.source`). `RunRecord.regime` and
   digest unchanged.
2. Lineage from live state (against `d4271e1fc`): a resumed run recorded a
   continued stage's `initial` as `None` (uninterrupted: `"self-play"`), raw
   weights still bit-equal. Now `Model.trained_by` names the last self-play
   stage; a stage continues live when `Run.live_session` is that stage, else
   from this stage's recovery snapshot, else is refused by name
   (`32dd45e43`, `242df4246`).
3. Starting states over held weights (against `7fcacc3e5`):
   `import_policy` then `self_play(..., pretrained=...)` trained weights the
   record never named. `self_play` refuses `initial_weights`, `pretrained`
   and `learning_state` on a model with `source` set (every `trained_by`
   write also sets `source`), before `_open_stage` (`0822bddc0`).

Last check (after 3): regime_functions, regimes, policy_import,
learning_state, recovery, package_boundaries: 87 passed. Broader module list
passed after 1, except the known test_game_errors `binding_mismatch` (3).

4. Fourth review (against `40d0e373a`): no new mismatch. Checked
   two_models on resume (the skipped opponent export takes the same
   `export-reference` name later, so numbering holds), an imported model
   entering self-play (fresh learner, producer recorded) and a stage export
   returned for a later `run.export` without decks (documented: published
   weights are returned as they are). This slice is Outcomes 1-5 and 7;
   the brief allows it: "Land in reviewable slices if the whole does not
   fit". Next is the gate, not more design.

## Open product choice (follow-up; does not block this slice)

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
