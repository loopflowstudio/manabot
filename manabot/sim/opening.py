"""Signed units for a frozen player's first-own-decision critic proxy."""

import math

import numpy as np

from manabot.env.observation import ObservationBuffers
from manabot.policy.protocol import BackendPolicy


def signed_value(value: float, semantic: str | None) -> float:
    if not math.isfinite(value):
        raise ValueError("opening critic is nonfinite")
    if semantic == "win_logit":
        return math.tanh(value / 2)
    if semantic == "signed_outcome":
        return value
    raise ValueError("opening critic requires signed_outcome or win_logit units")


def opening_value(agent: BackendPolicy, obs: ObservationBuffers) -> float:
    batch = agent.prepare(
        {key: np.asarray(value, np.float32)[None] for key, value in obs.items()},
        inference=True,
    )
    _, values = agent.evaluate(batch)
    # Backend.evaluate already converts categorical WDL logits to expectation.
    return signed_value(float(values[0]), agent.value_semantic)
