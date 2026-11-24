"""Install the writable observation transport for dense parity and scripted games."""

import numpy as np

from manabot.policy.spans import ObservationSelection
from manabot.training.rollout.collection import (
    Collector,
    ObservationRows,
    live_observations,
)


def dense_collector(collector: Collector) -> dict[str, np.ndarray]:
    """Preserve the current native roots and give the fixture writable rows."""
    live = live_observations(collector)
    buffers = (
        live.export() if isinstance(live, ObservationSelection) else dict(live)
    ) | collector._buffers
    space = collector.observation_space
    collector._env.set_buffers(
        buffers, space.config.action_overflow, layout=space.layout
    )
    collector._buffers = buffers
    collector._observations = ObservationRows(buffers, collector._observation_keys, 0)
    return buffers
