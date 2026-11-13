"""Updates: each learns from one collected batch.

`damped_policy.py` is the default update and `ppo.py` the control it is
measured against. Both take the backend module first
(`manabot.backends.torch`) and do array math through it; the damped update
also runs on `manabot.backends.mlx`. The rest are the pieces they share and
that a custom update may call: numerical health and selection diagnostics,
over host rows. The terms and losses live in the backend's `rl.py`, the
optimizer step in its `learner.py` and rollout admission in its `rollout.py`.
"""
