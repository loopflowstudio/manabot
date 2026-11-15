"""One viewer observation through a loaded baseline, retaining all legal scores."""

import numpy as np
import torch

from manabot.backends.torch.policy import PolicyInference
from manabot.env.observation import ObservationSpace
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
    if count > space.encoder.max_actions:
        space = ObservationSpace(
            space.encoder.config.model_copy(update={"max_actions": count})
        )
    encoded = space.encode(observation)
    with torch.no_grad():
        logits, value = scope.agent(
            scope.prepare(
                {
                    key: np.asarray(value, dtype=np.float32)[None]
                    for key, value in encoded.items()
                }
            )
        )
        probabilities = tuple(torch.softmax(logits, dim=-1)[0, :count].tolist())
    return probabilities, float(value.item())
