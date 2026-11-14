"""One viewer observation through a loaded baseline, retaining all legal scores."""

import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.model.inference import admit_on_device
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
    inputs = admit_on_device(
        agent,
        {key: torch.as_tensor(value).unsqueeze(0) for key, value in encoded.items()},
    )
    with torch.no_grad():
        logits, value = agent(inputs)
        probabilities = tuple(torch.softmax(logits, dim=-1)[0, :count].tolist())
    return probabilities, float(value.item())
