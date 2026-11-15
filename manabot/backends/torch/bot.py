"""One viewer observation through a loaded baseline, retaining all legal scores."""

import numpy as np
import torch

from manabot.backends.torch.arrays import device_of, on_device
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
import managym


def evaluate_observation(
    agent: ActorCritic, space: ObservationSpace, observation: managym.Observation
) -> tuple[tuple[float, ...], float]:
    count = len(observation.action_space.actions)
    if count > space.encoder.max_actions:
        space = ObservationSpace(
            space.encoder.config.model_copy(update={"max_actions": count})
        )
    encoded = space.encode(observation)
    with torch.no_grad():
        inputs = on_device(
            agent.prepare(
                {
                    key: np.asarray(value, dtype=np.float32)[None]
                    for key, value in encoded.items()
                },
                inference=True,
            ),
            device_of(agent),
        )
        logits, value = agent(inputs)
        probabilities = tuple(torch.softmax(logits, dim=-1)[0, :count].tolist())
    return probabilities, float(value.item())
