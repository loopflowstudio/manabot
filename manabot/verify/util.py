"""Shared helpers for bounded verification diagnostics."""

from __future__ import annotations

import logging

import numpy as np
from pydantic import Field
import torch

from manabot.config.load import deep_merge
from manabot.env.env import Env
from manabot.env.observation import ActionEnum
from manabot.infra.config import (
    BaseConfig,
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)


class VerificationConfig(BaseConfig):
    """Environment and untrained-policy setup for bounded diagnostics."""

    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    match: MatchConfig = Field(default_factory=MatchConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    agent: ModelConfig = Field(default_factory=lambda: ModelConfig(attention_on=False))


def build_verification_config(**overrides: object) -> VerificationConfig:
    base = VerificationConfig().model_dump()
    match_overrides = overrides.pop("match", None)
    merged = deep_merge(base, overrides)
    if isinstance(match_overrides, dict):
        merged["match"] = {**base["match"], **match_overrides}
    elif match_overrides is not None:
        merged["match"] = match_overrides
    return VerificationConfig.model_validate(merged)


def _agent_device(agent) -> torch.device:
    try:
        return next(agent.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def _tensor_observation(
    obs: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        for key, value in obs.items()
    }


def _select_agent_action(agent, obs: dict[str, np.ndarray], deterministic: bool) -> int:
    tensor_obs = _tensor_observation(obs, device=_agent_device(agent))
    action, _, _, _ = agent.get_action_and_value(
        tensor_obs,
        deterministic=deterministic,
    )
    return int(action.item())


def _pass_priority_fallback(env: Env) -> int | None:
    """Return a pass-priority action index for the current raw action space."""

    raw_obs = getattr(env, "last_raw_obs", None)
    action_space = getattr(raw_obs, "action_space", None)
    actions = getattr(action_space, "actions", None)
    if actions is None:
        return None

    pass_priority_type = int(ActionEnum.PRIORITY_PASS_PRIORITY)
    for index, option in enumerate(actions):
        if int(option.action_type) == pass_priority_type:
            return index
    return None


def step_with_fallback(env: Env, action: int, fallback_action: int = 0):
    """Step env, retrying with safer fallbacks when policy action errors."""

    try:
        return env.step(action)
    except Exception as original_error:
        candidates: list[int] = []
        if fallback_action >= 0:
            candidates.append(int(fallback_action))

        pass_idx = _pass_priority_fallback(env)
        if pass_idx is not None and pass_idx not in candidates:
            candidates.append(pass_idx)

        if 0 not in candidates:
            candidates.append(0)

        last_error = original_error
        for candidate in candidates:
            try:
                return env.step(candidate)
            except Exception as error:
                last_error = error

        raise last_error


def suppress_truncation_logs() -> None:
    """Reduce noisy truncation warnings; scripts read truncation counters directly."""

    logging.getLogger("manabot.env.observation.encode_actions").setLevel(logging.ERROR)
    logging.getLogger("manabot.env.observation.encode_cards").setLevel(logging.ERROR)
    logging.getLogger("manabot.env.observation.encode_permanents").setLevel(
        logging.ERROR
    )
