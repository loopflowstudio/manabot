"""Permute a live viewer projection while the native encoder resolves its joins."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np

from manabot.env.observation import ObservationSpace
from manabot.policy.spans import ObservationSelection
import managym


def encode(raw: managym.Observation, space: ObservationSpace) -> ObservationSelection:
    batch = managym.encode_observation_batch(
        [raw], space.layout, space.config.action_overflow
    )
    return ObservationSelection(space, ((batch, 0),))


@contextmanager
def reorder_offers(
    raw: managym.Observation, order: Sequence[int] | np.ndarray
) -> Iterator[None]:
    """Present the same offers in another order for the duration of the read."""
    space = raw.action_space
    original = space.actions
    if sorted(order) != list(range(len(original))):
        raise ValueError("expected a permutation of all offers")
    space.actions = [original[i] for i in order]
    raw.action_space = space
    try:
        yield
    finally:
        space.actions = original
        raw.action_space = space


@contextmanager
def reorder_objects(raw: managym.Observation) -> Iterator[None]:
    """Reverse the occupied card and permanent slots, preserving native identities."""
    keys = ("agent_cards", "opponent_cards", "agent_permanents", "opponent_permanents")
    original = {key: getattr(raw, key) for key in keys}
    try:
        for key, rows in original.items():
            setattr(raw, key, rows[::-1])
        yield
    finally:
        for key, rows in original.items():
            setattr(raw, key, rows)
