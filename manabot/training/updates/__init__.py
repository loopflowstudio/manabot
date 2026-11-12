"""Updates: each learns from one collected batch.

`damped_policy.py` is the default update and `ppo.py` the control it is
measured against. Both take the backend module first
(`manabot.backends.torch`) and do array math through it. The rest are the
pieces they share and that a custom update may call: rollout admission,
numerical health and selection diagnostics. These read the player's
observations, legal actions and `ActorCritic` heads; the terms and losses
live in the backend's `rl.py`.
"""
