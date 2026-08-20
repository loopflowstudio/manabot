"""Ordinary engine runtime for the belief-forming manabot."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import torch

from manabot.belief.agent import AgentMemory, AgentStep, Manabot, ViewerDecision
from manabot.belief.encoding import BeliefEncodingSchema, belief_schema_from_engine
from manabot.belief.state import (
    BeliefError,
    BeliefModel,
    CompatibleDealBeliefModel,
    ViewerHistory,
)
from manabot.model.agent import Agent
from managym.decision import Command, Observation, SemanticTransition
from managym.possible_worlds import PossibleWorldSpace


def _engine(runtime_env: Any) -> Any:
    return getattr(runtime_env, "_engine", runtime_env)


def _batched_observation(
    observation: Mapping[str, np.ndarray | torch.Tensor],
) -> dict[str, torch.Tensor]:
    if "actions_valid" not in observation:
        raise BeliefError("runtime observation is missing its legal-action mask")
    action_mask = torch.as_tensor(observation["actions_valid"])
    if action_mask.ndim not in {1, 2}:
        raise BeliefError(
            "runtime legal-action mask must be unbatched or singly batched"
        )
    already_batched = action_mask.ndim == 2
    if already_batched and action_mask.shape[0] != 1:
        raise BeliefError("ManabotPlayer consumes one engine decision at a time")
    tensors: dict[str, torch.Tensor] = {}
    for key, value in observation.items():
        tensor = torch.as_tensor(value)
        tensors[key] = tensor if already_batched else tensor.unsqueeze(0)
    return tensors


def viewer_decision_from_engine(
    engine: Any,
    observation: Mapping[str, np.ndarray | torch.Tensor],
    history: ViewerHistory,
) -> ViewerDecision:
    """Bind one encoded observation to native semantic and world authority."""

    semantic = Observation.from_json(engine.semantic_observation_json(history.viewer))
    if semantic.revision != history.current_revision:
        raise BeliefError("runtime history is stale for the current engine decision")
    if semantic.viewer_state_hash != history.current_viewer_state_hash:
        raise BeliefError("runtime history does not match the current viewer state")
    frame = semantic.decision
    if frame is None or frame.actor != history.viewer:
        raise BeliefError("ManabotPlayer can act only at its viewer's decision")
    if frame.revision != semantic.revision:
        raise BeliefError("semantic decision revision differs from its Observation")

    tensors = _batched_observation(observation)
    legal_indexes = tuple(
        int(index)
        for index in torch.nonzero(tensors["actions_valid"][0] > 0, as_tuple=False)
        .flatten()
        .tolist()
    )
    if legal_indexes != tuple(range(len(frame.offers))):
        raise BeliefError(
            "native semantic offers are not aligned with the encoded action surface"
        )
    commands = tuple(
        Command(
            command_id=f"manabot-{frame.revision}-{int(offer['id'])}",
            expected_revision=frame.revision,
            offer_id=int(offer["id"]),
        )
        for offer in frame.offers
    )
    space = PossibleWorldSpace.from_engine(engine, history.viewer)
    return ViewerDecision(
        observation=tensors,
        world_space=space,
        viewer_history=history,
        legal_commands=commands,
    )


class ManabotPlayer:
    """Stateful ordinary player using generated beliefs on every decision."""

    def __init__(
        self,
        policy_value: Agent,
        *,
        belief_model: BeliefModel | None = None,
        belief_schema: BeliefEncodingSchema | None = None,
    ) -> None:
        self.policy_value = policy_value
        self.belief_model = belief_model or CompatibleDealBeliefModel()
        self.belief_schema = belief_schema
        self.manabot: Manabot | None = None
        self.viewer: int | None = None
        self.history: ViewerHistory | None = None
        self.memory = AgentMemory()
        self.last_step: AgentStep | None = None
        self.last_action: int | None = None

    def start_game(self, env: Any, seat: int) -> None:
        engine = _engine(env)
        observation = Observation.from_json(engine.semantic_observation_json(seat))
        history = ViewerHistory.from_observation(observation)
        schema = self.belief_schema
        if schema is None:
            schema = belief_schema_from_engine(
                engine,
                PossibleWorldSpace.from_engine(engine, seat),
                count_buckets=self.policy_value.belief_count_buckets,
            )
        self.manabot = Manabot(
            policy_value=self.policy_value,
            belief_model=self.belief_model,
            belief_schema=schema,
        )
        self.belief_schema = schema
        self.viewer = seat
        self.history = history
        self.memory = AgentMemory()
        self.last_step = None
        self.last_action = None

    def act(
        self,
        env: Any,
        observation: Mapping[str, np.ndarray | torch.Tensor],
    ) -> int:
        if self.manabot is None or self.viewer is None or self.history is None:
            raise RuntimeError("start_game must run before ManabotPlayer.act")
        decision = viewer_decision_from_engine(_engine(env), observation, self.history)
        step = self.manabot.decide(decision, self.memory)
        action_by_offer = {
            command.offer_id: action_index
            for command, action_index in zip(
                decision.legal_commands,
                step.result.legal_action_indexes,
                strict=True,
            )
        }
        try:
            action = action_by_offer[step.command.offer_id]
        except KeyError as error:
            raise BeliefError(
                "Manabot selected a Command outside the action surface"
            ) from error
        self.memory = step.next_memory
        self.last_step = step
        self.last_action = action
        return action

    def command_for_action(
        self, engine: Any, action: int, *, command_id: str
    ) -> Command:
        del engine
        if self.last_step is None or action != self.last_action:
            raise RuntimeError("action does not match the latest Manabot decision")
        return replace(self.last_step.command, command_id=command_id)

    def observe_step(
        self, env: Any, acting: int, transition: SemanticTransition
    ) -> None:
        del acting
        if self.viewer is None or self.history is None:
            raise RuntimeError("start_game must run before ManabotPlayer.observe_step")
        engine = _engine(env)
        observation = Observation.from_json(
            engine.semantic_observation_json(self.viewer)
        )
        self.history = self.history.advance(transition.receipt, observation)


__all__ = ["ManabotPlayer", "viewer_decision_from_engine"]
