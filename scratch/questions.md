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

1a. (2026-11-12) Question 1 is partly settled by direction. Jack Heart, on the
    list that included deleting the collector's snapshot/restore: "do em all in
    this diff". The collector (`manabot/training/rollout/collection.py`) can no
    longer be saved or restored: `snapshot`, `restore`, `CollectorSnapshot`,
    `AdaptiveSnapshot`, `CurrentGames`, the step journal and the
    `recovery_max_microsteps` / `recovery_max_game_microsteps` constructor
    arguments are gone. The regime executor still exists and still accepts
    `resume_from`, but a resumed attempt now opens fresh native streams seeded
    `collection seed + snapshot iteration` and carries over only the learner,
    optimizer, sampling RNGs and cumulative collector counts
    (`UpdateSnapshot` format 5). It does not reproduce the uninterrupted run:
    games in progress at the snapshot are abandoned. A snapshot written at or
    before 439379869 (format 4, pickled collector state) is refused with that
    revision named; an adaptive stage refuses to resume because its sampler
    state is no longer saved. `TrainingRegime.recovery` and
    `recovery_max_microsteps` still validate and serialize (saved regimes stay
    readable) and still select snapshot cadence, but no longer bound a journal.
    Still open for review: whether the five frozen regimes listed above need
    anything beyond "run from 439379869".
14. (2026-11-13) Recipe function name. Docs now call a recipe module's entry
    point `train(config)` (`manabot.recipes.elemental.train`), which experiment
    files import or define themselves, beside their existing `runs()`,
    `eval(state)` and `report()`. Basis: "i liekd it better when there were
    explicit eval(), train() and launch() functions". `runs()` is not renamed
    to `launch()`: cutover.md says not to rename existing `runs()`
    declarations, and Etude discovers `runs`. Assumed; the alternative keeps
    `elemental(config)` / `belief(config)` as the names.
15. (2026-11-13) Questions 9 and 10 are answered in the docs by making outputs
    immutable rather than by requiring a new `--out`: there is no `final.pt`;
    the endpoint is the last update's checkpoint, and a later target writes a
    later file. Belief outputs carry the policy file's stem, so a policy
    trained further gets its own histories and sampler. Assumed from
    cutover.md ("Published checkpoint paths must not be overwritten", "cannot
    reuse histories or a sampler from the old policy"); not confirmed by a person.
16. (2026-11-13) `manabot train` exit status after a planned stop is 75 in the
    docs, matching Etude's training exit; it was 1. Assumed.
