"""Belief-forming lifecycle around manabot's shared policy/value core."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.belief.encoding import BeliefEncodingSchema, encode_belief
from manabot.belief.state import (
    BeliefError,
    BeliefModel,
    BeliefState,
    BeliefUpdate,
    ViewerHistory,
)
from manabot.model.agent import Agent
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

    observation_identity: str
    observation: Mapping[str, torch.Tensor]
    world_space: PossibleWorldSpace
    viewer_history: ViewerHistory
    legal_commands: tuple[Command, ...]

    def __post_init__(self) -> None:
        if self.observation_identity != self.world_space.source_viewer_state_hash:
            raise BeliefError("decision observation does not match its world space")
        if self.viewer_history.identity != self.world_space.source_history_identity:
            raise BeliefError("decision history does not match its world space")
        if self.viewer_history.viewer != self.world_space.viewer:
            raise BeliefError("decision history changed viewer")
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
        policy_value: Agent,
        belief_model: BeliefModel,
        belief_schema: BeliefEncodingSchema,
    ) -> None:
        if policy_value.max_conditions > 0:
            raise BeliefError(
                "Manabot cannot use the legacy positional condition channel"
            )
        if policy_value.belief_count_buckets != belief_schema.count_buckets:
            raise BeliefError("Agent and belief encoding count buckets differ")
        maximum_card_id = max(row.card_def_id for row in belief_schema.rows)
        if maximum_card_id >= policy_value.belief_card_embedding.num_embeddings:
            raise BeliefError("belief schema exceeds the Agent card vocabulary")
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
        result = self._evaluate(decision, update.belief)
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

        return self._evaluate(decision, belief)

    def _evaluate(
        self, decision: ViewerDecision, belief: BeliefState
    ) -> PolicyValueResult:
        if belief.space.identity != decision.world_space.identity:
            raise BeliefError("belief world-space identity does not match the decision")
        encoded_started = time.perf_counter()
        view = encode_belief(belief, self.belief_schema)
        encoding_seconds = time.perf_counter() - encoded_started

        device = next(self.policy_value.parameters()).device
        observation = {
            key: value.to(device) for key, value in decision.observation.items()
        }
        observation.update(view.to_torch_observation(batch_size=1, device=device))
        was_training = self.policy_value.training
        self.policy_value.eval()
        inference_started = time.perf_counter()
        try:
            with torch.inference_mode():
                logits, value = self.policy_value(observation)
                policy_tensor = torch.softmax(logits[0], dim=-1)
        finally:
            self.policy_value.train(was_training)
        inference_seconds = time.perf_counter() - inference_started
        policy = np.asarray(
            policy_tensor.detach().cpu().numpy(), dtype=np.float32
        ).copy()
        policy.setflags(write=False)
        scalar_value = np.float32(value[0].detach().cpu().item())
        legal_action_indexes = tuple(
            int(index)
            for index in torch.nonzero(
                observation["actions_valid"][0] > 0, as_tuple=False
            )
            .flatten()
            .cpu()
            .tolist()
        )
        output_bytes = _output_bytes(policy, scalar_value)
        receipt = hashlib.sha256()
        receipt.update(decision.observation_identity.encode("ascii"))
        receipt.update(decision.world_space.identity.encode("ascii"))
        receipt.update(view.encoding_receipt.encode("ascii"))
        receipt.update(output_bytes)
        return PolicyValueResult(
            policy=policy,
            value=scalar_value,
            legal_action_indexes=legal_action_indexes,
            belief_identity=belief.identity,
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
