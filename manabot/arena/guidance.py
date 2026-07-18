"""Checkpoint-policy priors for the existing INT-6 arena authority."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from manabot.env import ObservationSpace
from manabot.model.agent import Agent
from manabot.sim.flat_mc import MatchupPlayer, load_checkpoint_agent
from manabot.sim.mcts import (
    DeterminizedPuctPlayer,
    LeafEvaluation,
    UniformRandomLeafEvaluator,
)

from .models import file_sha256
from .players import build_player


class PolicyPriorRandomLeafEvaluator:
    """Use checkpoint logits only for priors and random terminal leaf values."""

    def __init__(self, agent: Agent, observation_space: ObservationSpace):
        try:
            device = next(agent.parameters()).device
        except StopIteration as exc:  # pragma: no cover - Agent has parameters
            raise ValueError("policy-prior evaluator requires a model") from exc
        if device.type != "cpu":
            raise ValueError("policy-prior evaluator requires a CPU model")
        self.agent = agent.eval()
        self.observation_space = observation_space
        self.random_leaf = UniformRandomLeafEvaluator()
        self.last_root_priors: np.ndarray | None = None

    def _priors(
        self, observation: Any | Mapping[str, np.ndarray], *, action_count: int
    ) -> np.ndarray:
        if action_count < 1:
            raise RuntimeError("policy-prior node has no legal actions")
        encoded = (
            observation
            if isinstance(observation, Mapping)
            else self.observation_space.encode(observation)
        )
        tensor_obs = {
            key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            logits, _discarded_value = self.agent(tensor_obs)
            legal_logits = logits[0, :action_count]
            if not torch.isfinite(legal_logits).all():
                raise RuntimeError("checkpoint returned nonfinite legal logits")
            priors = torch.softmax(legal_logits, dim=-1)
        result = priors.detach().cpu().numpy().astype(np.float64)
        if result.shape != (action_count,) or not np.isclose(result.sum(), 1.0):
            raise RuntimeError("checkpoint returned invalid legal-action priors")
        return result

    def root_priors(self, observation: Any | None, *, action_count: int) -> np.ndarray:
        if observation is None:
            raise ValueError("policy-prior evaluator requires the root observation")
        priors = self._priors(observation, action_count=action_count)
        self.last_root_priors = priors.copy()
        return priors

    def evaluate(
        self,
        env: Any,
        observation: Any,
        *,
        root_player: int,
        node_player: int,
        seed: int,
        max_steps: int,
        branch_session: Any | None = None,
    ) -> LeafEvaluation:
        priors = self._priors(observation, action_count=int(env.action_count()))
        random_evaluation = self.random_leaf.evaluate(
            env,
            observation,
            root_player=root_player,
            node_player=node_player,
            seed=seed,
            max_steps=max_steps,
            branch_session=branch_session,
        )
        return LeafEvaluation(
            priors=priors,
            root_value=random_evaluation.root_value,
            cap_hit=random_evaluation.cap_hit,
        )


def build_arena_player(
    registration: Any, *, seed: int, checkpoint_path: str | None = None
) -> tuple[MatchupPlayer, ObservationSpace | None]:
    """Resolve the additive prior-only candidate or delegate unchanged kinds."""

    spec = dict(registration.player_spec)
    if spec.get("kind") != "policy_prior_puct":
        return build_player(registration, seed=seed, checkpoint_path=checkpoint_path)
    if checkpoint_path is None:
        raise FileNotFoundError("checkpoint candidate bytes are unavailable")
    if spec["implementation_source_sha256"] != file_sha256(Path(__file__)):
        raise RuntimeError("policy-prior PUCT implementation source drift")
    agent, observation_space = load_checkpoint_agent(checkpoint_path)
    evaluator = PolicyPriorRandomLeafEvaluator(agent, observation_space)
    player = DeterminizedPuctPlayer(
        int(spec["sims"]),
        worlds=int(spec["worlds"]),
        c_puct=float(spec["c_puct"]),
        max_steps=int(spec["max_steps"]),
        seed=seed,
        evaluator=evaluator,
        branch_driver_id=str(spec["branch_driver_id"]),
        branch_audit=False,
    )
    semantics = registration.search_semantics
    if semantics is None:
        raise RuntimeError("policy-prior PUCT has no registered search semantics")
    if semantics.model_dump() != {
        "branch_audit": False,
        "root_prior": "checkpoint-policy-softmax-v1",
        "leaf_evaluator": "uniform-random-terminal-v1",
    }:
        raise RuntimeError("policy-prior PUCT search semantics drift")
    if spec["root_noise"] != "none" or not spec["deterministic"]:
        raise RuntimeError("policy-prior PUCT stochastic policy semantics drift")
    return player, observation_space
