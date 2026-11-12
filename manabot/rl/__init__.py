"""Reinforcement-learning pieces for a model of any kind.

This is the PyTorch implementation of the maths our updates use: returns and
advantages (`returns`), per-row loss terms (`losses`), advantage selection
(`selection`) and the optimizer step and EMA (`step`), with the numerical
checks they share (`numerics`). An update's per-minibatch step (forward, loss,
backward, clip, optimizer step) is where another framework plugs in.

Every module here reads tensors, an `nn.Module`, an optimizer and small typed
records, and imports nothing from the game, its observations, the player
model or collection (`tests/test_package_boundaries.py` enforces it). The
player's updates and their pieces live in `manabot.training.updates`.
Import names from their modules; this package
re-exports nothing.
"""
