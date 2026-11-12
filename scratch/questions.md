# Assumptions taken (reversible)

- Smoke size: one update of one transition per stream (2 streams unless a
  curriculum stratifies them), CPU unless `device=`; written to the given
  `out` with `training.sqlite` beside it (Etude EXP-49 shape).
- Stage names default to the function name numbered by call order
  (`self-play`, `self-play-2`); `stage=` overrides. Resume relies on this order.
- Resume re-enters the function from the top; completed stages replay their
  recorded results. Code between stages must be deterministic.
- Model names lowercase and unique; models are seeded from `run.seed` +
  crc32(name) unless `seed=`; the equivalence proof passes `seed=run.seed`.
- Function self-play uses the `iteration_fraction` schedule clock (saved
  regimes default to `run_elapsed_budget`).
- `run.export` binds `decks` setups when given a MatchConfig, else the world
  (needs a world-complete model). Stage exports are the model's checkpoints.
- Pause raises `TrainingPaused` and the attempt ends `interrupted` (existing
  saved-regime behavior); docs no longer promise `RunPaused`/`paused`.
- Fixed a latent bug while moving code: a second fresh self-play stage passed
  the previous stage's RunContext as its Etude experiment name.
- Kept the `#regimes-and-stages` heading so existing links stay valid.
- Self-play always keeps a restored completed snapshot live
  (`keep_restored=True`), so a saved regime whose next stage is fresh now
  rebuilds that stage's collector on recovery before discarding it (extra
  `recovery_seconds`, no change to trained results). A function cannot look
  ahead; one rule for both kinds was the simpler choice.
- Function self-play's Etude resolved config now records the run's
  `regime_function` plus the stage settings (`stage`), and its regime tag is
  the function's qualname; a saved run's carry the saved regime (unsmoked).
- `TrainingRegime` per-stage rules moved into `check_self_play`; a regime
  violating several rules may now report a different one first, and the
  accelerator message names the stage's own device.
