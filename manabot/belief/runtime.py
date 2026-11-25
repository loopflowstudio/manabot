"""Ordinary engine runtime for the belief-forming manabot."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from manabot.belief.agent import AgentMemory, AgentStep, Manabot, ViewerDecision
from manabot.belief.encoding import (
    BeliefCheckpointBinding,
    BeliefEncodingSchema,
    belief_schema_from_engine,
)
from manabot.belief.protocol import BeliefPolicy
from manabot.belief.state import (
    BeliefError,
    BeliefModel,
    CompatibleDealBeliefModel,
    ViewerHistory,
)
import managym
from managym.decision import Command, Observation, SemanticTransition
from managym.possible_worlds import PossibleWorldSpace


def _engine(runtime_env: Any) -> Any:
    return getattr(runtime_env, "_engine", runtime_env)


def viewer_decision_from_engine(
    engine: Any,
    observation: managym.Observation,
    history: ViewerHistory,
) -> ViewerDecision:
    """Bind one viewer observation to native semantic and world authority."""

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
        observation=observation,
        world_space=space,
        viewer_history=history,
        legal_commands=commands,
    )


class ManabotPlayer:
    """Stateful ordinary player using generated beliefs on every decision."""

    def __init__(
        self,
        policy_value: BeliefPolicy,
        *,
        belief_model: BeliefModel | None = None,
        belief_schema: BeliefEncodingSchema | None = None,
        checkpoint_binding: BeliefCheckpointBinding | None = None,
    ) -> None:
        self.policy_value = policy_value
        self.belief_model = belief_model or CompatibleDealBeliefModel()
        self.belief_schema = belief_schema
        self.checkpoint_binding = checkpoint_binding
        self.manabot: Manabot | None = None
        self.viewer: int | None = None
        self.history: ViewerHistory | None = None
        self.memory = AgentMemory()
        self.last_step: AgentStep | None = None
        self.last_action: int | None = None

    def start_game(self, env: Any, seat: int) -> None:
        self.manabot = None
        self.viewer = None
        self.history = None
        self.memory = AgentMemory()
        self.last_step = None
        self.last_action = None

        engine = _engine(env)
        observation = Observation.from_json(engine.semantic_observation_json(seat))
        history = ViewerHistory.from_observation(observation)
        runtime_schema = belief_schema_from_engine(
            engine,
            PossibleWorldSpace.from_engine(engine, seat),
            count_buckets=self.policy_value.belief_count_buckets,
        )
        if self.checkpoint_binding is not None:
            self.checkpoint_binding.validate_schema(runtime_schema)
        schema = self.belief_schema or runtime_schema
        if schema.identity != runtime_schema.identity:
            raise BeliefError("configured belief schema does not match the runtime")
        self.manabot = Manabot(
            policy_value=self.policy_value,
            belief_model=self.belief_model,
            belief_schema=schema,
        )
        self.belief_schema = schema
        self.viewer = seat
        self.history = history

    def act(
        self,
        env: Any,
        observation: managym.Observation,
    ) -> int:
        if self.manabot is None or self.viewer is None or self.history is None:
            raise RuntimeError("start_game must run before ManabotPlayer.act")
        decision = viewer_decision_from_engine(_engine(env), observation, self.history)
        step = self.manabot.decide(decision, self.memory)
        action = int(np.argmax(step.result.policy))
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
        if self.viewer is None or self.history is None:
            raise RuntimeError("start_game must run before ManabotPlayer.observe_step")
        engine = _engine(env)
        observation = Observation.from_json(
            engine.semantic_observation_json(self.viewer)
        )
        self.history = self.history.advance(
            transition.receipt,
            observation,
            acting=acting,
        )


__all__ = ["ManabotPlayer", "viewer_decision_from_engine"]
