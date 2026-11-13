"""Updates: each learns from one collected batch.

`damped_policy.py` is the default update and `ppo.py` the control it is
measured against. The rest are the pieces they share and that a custom update
may call: learner state and rollout admission, the reference magnet, numerical
health and selection diagnostics. These read the player's observations, legal
actions and `ActorCritic` heads; model-agnostic pieces live in `manabot.rl`.
"""
