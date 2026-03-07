- `run_first_light eval` is implemented as an evaluation-only run with `total_timesteps=0`.
  It records a completed run and final eval, but it does not resume from a saved checkpoint.
