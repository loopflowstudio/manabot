# MTG-303 plan: regime functions over a Run handle

## Docs written (intended behavior; nothing implemented yet)

- `docs/training.md`: new "Regime functions" section (Run handle, stage
  functions, RegimeCall identity, rehearsal, pause/resume, saved regimes);
  CLI `--function/--arguments/--rehearse`; remote-training `TrainingJob(call=)`
  and `TrainingJob.rehearse()`; "Regimes and stages" reframed as saved regimes.
- `docs/experiments.md`: "Declaring a comparison" opens with arms as a dict of
  `RegimeCall`s.
- `README.md`: "Training is code over a run" paragraph; recipes paragraph.
- `manabot/README.md`: `training/` layout lists `run.py` and `stages.py`.

## API the docs commit to

- `manabot.training.run`: `Run`, `Model`, `RegimeCall.of(fn, **args)`,
  `execute(call, out=, seed=, resume_from=)`, `rehearse(call, device=)`,
  `RunPaused`; `run.model(name, agent, observation)`, `run.export(model,
  weights=)`, `run.metrics(values, update=)`, `run.stage(name)`.
- `manabot.training.stages`: `self_play`, `import_policy`, `collect_belief`,
  `train_belief`, `run_saved_regime`. `self_play` kwargs mirror
  `TrainSelfPlay` field names/defaults; `opponent=Model` means frozen opponent;
  repeat call on a model = `initial` continuation.
- `TrainingJob(call=...)` alongside `regime=`; `TrainingJob.rehearse()`.
- `run.json` gets a `regime_call` block (function path, resolved arguments,
  `arguments_sha256`, revision = commit + tracked-source dirty digest).
- Example regime: `experiments/two_models.py` (two models, train one against the
  other frozen, export both) is the outcome-5 proof.

## Not in scope / deferred

- Per-game result rows (MTG-282 `5bf811ced`): the docs do not promise them.
  If the Run handle reaches result rows, write them through it reusing that
  code; otherwise they go to the deck-builder Task.

## Proposed Etude PR (not worked around in manabot)

Jack Heart (steer on outcome 7): "Agreed on dry run or like \"do a single tiny
update\" as a paradigm possibly to put into etude instead of config
validation". Proposal: an experiment-file hook (e.g. `rehearse(run)` or a
`Job.rehearse` command) that `etude launch` runs locally before renting, and
refuses the launch on failure. manabot side: `TrainingJob.rehearse()` /
`manabot train --rehearse`. Until that PR lands, the docs say `etude launch`
does not rehearse on its own.
