"""One viewer observation through a loaded baseline, retaining all legal scores."""

import torch

from manabot.backends.torch.policy import PolicyInference
from manabot.env.observation import ObservationSpace, encode_policy_batch
import managym

__all__ = [
    "evaluate_observation",
]


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
    obs_input = encode_policy_batch(space, [observation])
    with torch.no_grad():
        logits, value = scope.agent(scope.prepare(obs_input))
        probabilities = tuple(torch.softmax(logits, dim=-1)[0, :count].tolist())
    return probabilities, float(value.item())
