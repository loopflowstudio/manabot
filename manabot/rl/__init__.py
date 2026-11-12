"""Reinforcement-learning pieces for a model of any kind.

Every module here reads tensors, an `nn.Module`, an optimizer and small typed
records, and imports nothing from the game, its observations, the player
model or collection (`tests/test_package_boundaries.py` enforces it). The
player's updates and their pieces live in `manabot.training.updates`.
Import names from their modules; this package
re-exports nothing.
"""
