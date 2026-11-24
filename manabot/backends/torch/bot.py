"""One viewer observation through a loaded baseline, retaining all legal scores."""

import numpy as np
import torch

from manabot.backends.torch.policy import PolicyInference
from manabot.env.observation import ObservationSpace, encode_observation
import managym


def evaluate_observation(
    scope: PolicyInference, space: ObservationSpace, observation: managym.Observation
) -> tuple[tuple[float, ...], float]:
    """Return legal-action probabilities and the acting viewer's value.

    The open scope caches the full card-program table once for its unchanged
    policy, so repeated evaluations (Search calls this per rollout ply)
    run only the observation forward instead of re-encoding the catalog.
    """
    count = len(observation.action_space.actions)
    if count > space.config.max_actions:
        space = ObservationSpace(space.config.model_copy(update={"max_actions": count}))
    if space.one_token:
        obs_input: object = managym.encode_observation_batch(
            [observation], space.layout, space.config.action_overflow
        )
    else:
        # Deprecated v9/v10 dense path; removed in 0.3.0.
        encoded = encode_observation(space, observation)
        obs_input = {
            key: np.asarray(value, dtype=np.float32)[None]
            for key, value in encoded.items()
        }
    with torch.no_grad():
        logits, value = scope.agent(scope.prepare(obs_input))  # type: ignore[arg-type]
        probabilities = tuple(torch.softmax(logits, dim=-1)[0, :count].tolist())
    return probabilities, float(value.item())
