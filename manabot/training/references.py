"""Full-support reference policies over viewer-visible legal offers."""

import torch


def reference_distribution(obs: dict[str, torch.Tensor], kind: str) -> torch.Tensor:
    if kind not in {"uniform", "action_type_uniform"}:
        raise ValueError(f"unknown reference: {kind}")
    valid = obs["actions_valid"] > 0
    if not valid.any(-1).all():
        raise ValueError("reference requires at least one legal offer per row")
    if kind == "uniform":
        return valid / valid.sum(-1, keepdim=True)
    action_types = obs["actions"][..., :-1]
    types = action_types.argmax(-1)
    # Count each type once per row without an offers-by-offers matrix.
    counts = action_types.new_zeros(*types.shape[:-1], action_types.shape[-1])
    counts.scatter_add_(-1, types, valid.to(counts.dtype))
    weights = valid / counts.gather(-1, types).clamp_min(1)
    return weights / weights.sum(-1, keepdim=True)
