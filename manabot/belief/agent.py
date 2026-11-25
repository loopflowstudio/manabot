"""Belief-forming lifecycle around manabot's shared policy/value core."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from manabot.belief.encoding import BeliefEncodingSchema, encode_belief
from manabot.belief.protocol import BeliefPolicy
from manabot.belief.range import BeliefError, BeliefState
from manabot.belief.state import (
    BeliefModel,
    BeliefUpdate,
    ViewerHistory,
)
from managym.decision import Command
from managym.possible_worlds import PossibleWorldSpace


def _output_bytes(policy: NDArray[np.float32], value: np.float32) -> bytes:
    return (
        policy.astype("<f4", copy=False).tobytes()
        + np.asarray([value], dtype="<f4").tobytes()
    )


@dataclass(frozen=True, slots=True)
class ViewerDecision:
    """One viewer-safe policy decision and its authoritative belief domain."""

    observation: Mapping[str, np.ndarray]
    world_space: PossibleWorldSpace
    viewer_history: ViewerHistory
    legal_commands: tuple[Command, ...]

    def __post_init__(self) -> None:
        if self.viewer_history.viewer != self.world_space.viewer:
            raise BeliefError("decision history changed viewer")
        if self.viewer_history.current_revision != self.world_space.source_revision:
            raise BeliefError(
                "decision history revision does not match its world space"
            )
        if (
            self.viewer_history.current_viewer_state_hash
            != self.world_space.source_viewer_state_hash
        ):
            raise BeliefError("decision history does not match its world space")
        valid = self.observation.get("actions_valid")
        if valid is None or valid.ndim != 2 or valid.shape[0] != 1:
            raise BeliefError("ViewerDecision requires one batched action mask")
        if int((valid[0] > 0).sum().item()) != len(self.legal_commands):
            raise BeliefError("legal Commands do not match the observation action mask")


@dataclass(frozen=True, slots=True)
class AgentMemory:
    """Autonomous belief memory; explicit evaluation never mutates it."""

    belief: BeliefState | None = None


@dataclass(frozen=True, slots=True)
class PolicyValueResult:
    policy: NDArray[np.float32]
    value: np.float32
    legal_action_indexes: tuple[int, ...]
    belief_identity: str
    belief_encoding_receipt: str
    inference_receipt: str
    encoding_seconds: float
    inference_seconds: float

    @property
    def output_bytes(self) -> bytes:
        return _output_bytes(self.policy, self.value)


@dataclass(frozen=True, slots=True)
class AgentStep:
    command: Command
    result: PolicyValueResult
    belief_update: BeliefUpdate
    next_memory: AgentMemory
    belief_update_seconds: float


class Manabot:
    """Ordinary belief-forming agent with one explicit decision core."""

    def __init__(
        self,
        *,
        policy_value: BeliefPolicy,
        belief_model: BeliefModel,
        belief_schema: BeliefEncodingSchema,
    ) -> None:
        policy_value.admit_belief(belief_schema)
        self.policy_value = policy_value
        self.belief_model = belief_model
        self.belief_schema = belief_schema

    def decide(self, decision: ViewerDecision, memory: AgentMemory) -> AgentStep:
        """Form a belief, run policy/value, choose a Command, and update memory."""

        started = time.perf_counter()
        update = self.belief_model.update(
            previous=memory.belief,
            world_space=decision.world_space,
            viewer_history=decision.viewer_history,
        )
        update_seconds = time.perf_counter() - started
        result = self.evaluate_under_belief(decision, update.belief)
        action_index = int(np.argmax(result.policy))
        try:
            command_position = result.legal_action_indexes.index(action_index)
        except ValueError as error:
            raise BeliefError(
                "policy selected an action without a legal Command"
            ) from error
        return AgentStep(
            command=decision.legal_commands[command_position],
            result=result,
            belief_update=update,
            next_memory=AgentMemory(belief=update.belief),
            belief_update_seconds=update_seconds,
        )

    def evaluate_under_belief(
        self, decision: ViewerDecision, belief: BeliefState
    ) -> PolicyValueResult:
        """Evaluate one explicit intervention without mutating agent memory."""

        if belief.space.identity != decision.world_space.identity:
            raise BeliefError("belief world-space identity does not match the decision")
        encoded_started = time.perf_counter()
        view = encode_belief(belief, self.belief_schema)
        encoding_seconds = time.perf_counter() - encoded_started

        observation = dict(decision.observation)
        observation.update(view.observation(batch_size=1))
        prepared = self.policy_value.prepare(observation, inference=True)
        inference_started = time.perf_counter()
        probabilities, value = self.policy_value.evaluate_probabilities(prepared)
        policy = np.asarray(probabilities[0], dtype=np.float32).copy()
        inference_seconds = time.perf_counter() - inference_started
        policy.setflags(write=False)
        scalar_value = np.float32(value[0])
        legal_action_indexes = tuple(
            int(index) for index in np.flatnonzero(prepared.legal[0])
        )
        output_bytes = _output_bytes(policy, scalar_value)
        receipt = hashlib.sha256()
        receipt.update(
            decision.viewer_history.current_viewer_state_hash.encode("ascii")
        )
        receipt.update(decision.world_space.identity.encode("ascii"))
        receipt.update(view.encoding_receipt.encode("ascii"))
        receipt.update(output_bytes)
        return PolicyValueResult(
            policy=policy,
            value=scalar_value,
            legal_action_indexes=legal_action_indexes,
            belief_identity=belief.digest,
            belief_encoding_receipt=view.encoding_receipt,
            inference_receipt=receipt.hexdigest(),
            encoding_seconds=encoding_seconds,
            inference_seconds=inference_seconds,
        )


__all__ = [
    "AgentMemory",
    "AgentStep",
    "Manabot",
    "PolicyValueResult",
    "ViewerDecision",
]
