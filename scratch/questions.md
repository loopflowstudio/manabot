# Open questions for review (MTG-329)

1. Recovery guarantees given up. The docs (`docs/training.md#what-resume-does-not-promise`)
   state as intended: no exact resume mid-collection (current-game recovery and
   exact CPU recovery removed), no proof a resumed run equals an uninterrupted
   one, no in-run source repair. Frozen regimes that declare `recovery`:
   elemental-long-183-v1, elemental-two-seat-181-v1, elemental-uniform-174-v1,
   elemental-ur-gw-186-v1, tla-frontier-c5fba4-w64. The task says this "is a
   decision for review, not something to settle silently"; it is not settled.
2. Saved regimes. Docs say they stay readable but no longer execute, and
   `--regime`/`--regime-name` name the last revision that runs them (the
   constraint allows "stay readable or name the revision"). Alternative:
   translate saved self-play regimes into `SelfPlay` so frozen experiments keep
   running on main. Chosen the simpler option; reversible.
3. Checkpoint evaluation during training moves to `etude eval` passes; the
   local `monitor_worker` follower is not documented any more. Is a local
   follow-while-training mode still wanted?
4. `--recipe` now also selects experiment recipes (`batched_baseline`,
   `<experiment>:<variant>`), replacing `--regime-name`. Assumed; reversible.
5. Belief recipe module. `manabot/recipes/belief.py` already holds the
   population-supervision demo (`run_demo`, `belief-learn-demo`), so the recipe
   is `manabot/recipes/belief_sampler.py`; docs updated. Reversible.
6. `collect_belief` takes the checkpoint path and `match`, not `weights=`: a
   path already names raw or EMA. Docs updated.
7. PPO schedules read `(iteration - 1) / updates`, as the executor did, so a
   PPO run's `updates` sets its rates; only damped-policy schedules are
   absolute. Docs now say so. Make PPO absolute (a reference horizon) instead?
8. Settings that a resumed loop may change are `device`, `workers`, `threads`
   (excluded from the `state.pt` settings digest); everything else refuses.
9. A finished recipe run again with a larger `--steps` trains on and replaces
   `final.pt` (the loop does this today; docs now say so). `etude continue`
   makes a child run for the same thing remotely. Should the local form also
   be a new run directory, so a finished run's `final.pt` is never replaced?
10. Stale later steps (found in the 2026-11-12 smoke). `--recipe belief
    --steps 1`, then `--steps 2` on the same `--out`: `policy/final.pt` is
    replaced (sha 7ab8b51c → b063a85d) but `histories.json` and `belief.pt`
    exist, so both steps return at once and the sampler still describes the
    update-1 policy. "A step whose output exists returns at once" and "a later
    target replaces `final.pt`" contradict each other for multi-step recipes.
    Options: a later target needs a new `--out` (question 9), or each step
    records its input's digest and reruns when it differs. Not changed.
11. Experiment name (found writing docs, 2026-11-12). Etude's `track` raises
    `experiment_differs_from_launch` when a recipe names an experiment other
    than the launching file's stem (`etude/train/run.py`), so
    `track(experiment="elemental")` in the recipe would refuse every launch
    from another experiment file. Docs now show `track` with no experiment:
    the launch names it, and `manabot train --recipe X` records it under `X`
    (the CLI would set `ETUDE_EXPERIMENT`, as it sets `ETUDE_OUT`). The
    prototype now does this (`ETUDE_EXPERIMENT`; a recipe called with neither
    refuses). Assumed; the alternative is an Etude
    change that lets a recipe's name be a default the launch overrides.
12. Default `--out`. Docs: `.runs/<recipe>-<seed>` (rerun resumes), except the
    `local` smoke, which keeps today's fresh timestamped directory; otherwise
    a second `uv run manabot train` would return at once without checking
    anything. Baseline takes the resumable default. Assumed; reversible.
13. `docs/experiments.md` "Declaring a comparison" and "Execute locally" still
    describe `Experiment.resolve()` → `TrainingRegime`, `run_experiment`,
    `active_runtime` and `pause.request`, all of which run through the
    executor (`experiment/runner.py` imports it). Does a local comparison
    runner survive the deletion, or is a comparison several `training(...)`
    jobs in an experiment file and nothing else? The section is not rewritten
    until that is decided.
