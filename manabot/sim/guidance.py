"""Build registered simulation players, including checkpoint-guided search.

`build_registered_player` is the one entry point. Code players come from the
Torch-free `players`; search and checkpoint players load their policy here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from manabot.env.observation import ObservationSpace
from manabot.infra.identity import file_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint
from manabot.sim.players import CODE_PLAYER_KINDS, build_code_player
from manabot.sim.registration import PlayerRegistration
from manabot.sim.search import local_update
from manabot.sim.search.flat_mc import MatchupPlayer, make_player
from manabot.sim.search.puct import (
    DeterminizedPuctPlayer,
    LeafEvaluation,
    UniformRandomLeafEvaluator,
    puct_source_sha256,
)


class PolicyPriorRandomLeafEvaluator:
    """Use checkpoint logits only for priors and random terminal leaf values."""

    def __init__(self, agent: ActorCritic, observation_space: ObservationSpace) -> None:
        try:
            device = next(agent.parameters()).device
        except StopIteration as exc:  # pragma: no cover - ActorCritic has parameters
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


def build_registered_player(
    registration: PlayerRegistration, *, seed: int, checkpoint_path: str | None = None
) -> tuple[MatchupPlayer, ObservationSpace | None]:
    """Resolve the additive prior-only candidate or delegate unchanged kinds."""

    spec = dict(registration.player_spec)
    if spec.get("kind") in CODE_PLAYER_KINDS:
        return build_code_player(registration, seed=seed), None
    if spec.get("kind") != "policy_prior_puct":
        return _build_policy_player(
            registration, seed=seed, checkpoint_path=checkpoint_path
        )
    if checkpoint_path is None:
        raise FileNotFoundError("checkpoint candidate bytes are unavailable")
    if spec["implementation_source_sha256"] != file_sha256(Path(__file__)):
        raise RuntimeError("policy-prior PUCT implementation source drift")
    agent, observation_space = load_checkpoint(checkpoint_path)
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


def _build_policy_player(
    registration: PlayerRegistration, *, seed: int, checkpoint_path: str | None = None
) -> tuple[MatchupPlayer, ObservationSpace | None]:
    spec = dict(registration.player_spec)
    if registration.runner_kind == "checkpoint":
        if checkpoint_path is None:
            raise FileNotFoundError("checkpoint candidate bytes are unavailable")
        spec["path"] = checkpoint_path
        if spec["kind"] == "agent_puct":
            if puct_source_sha256() != spec["implementation_source_sha256"]:
                raise ValueError("agent PUCT implementation source drift")
            if (
                file_sha256(checkpoint_path) != registration.checkpoint_sha256
                or Path(checkpoint_path).stat().st_size != registration.checkpoint_bytes
            ):
                raise ValueError("agent PUCT checkpoint byte identity drift")
            spec["checkpoint"] = checkpoint_path
        if spec["kind"] == "local_update":
            if (
                local_update.local_search_source_sha256()
                != spec["implementation_source_sha256"]
            ):
                raise ValueError("local-update implementation source drift")
            spec["checkpoint"] = checkpoint_path
            spec["checkpoint_sha256"] = registration.checkpoint_sha256
    player, obs_space = make_player(spec, seed=seed)
    attribute_names = {
        "sims": "simulations"
        if spec["kind"] in {"determinized_puct", "agent_puct"}
        else "sims",
        "rollouts_per_world": "rollouts",
        "worlds": "worlds",
        "c_puct": "c_puct",
        "max_steps": "max_steps",
        "branch_driver_id": "branch_driver_id",
        "branch_audit": "branch_audit",
    }
    for field, attribute in attribute_names.items():
        if field in spec and getattr(player, attribute, None) != spec[field]:
            raise RuntimeError(f"constructed player did not echo {field}")
    if spec["kind"] == "determinized_puct":
        semantics = registration.search_semantics
        if semantics is None:
            raise RuntimeError("determinized PUCT has no registered search semantics")
        if player.branch_audit != semantics.branch_audit:
            raise RuntimeError("constructed player did not echo branch_audit")
        if type(player.evaluator) is not UniformRandomLeafEvaluator:
            raise RuntimeError(
                "constructed player did not use the registered leaf evaluator"
            )
    if spec["kind"] == "agent_puct":
        if player.evaluator.value_mode != spec["value_mode"]:
            raise RuntimeError("constructed player did not echo value_mode")
        if (
            sum(p.numel() for p in player.evaluator.agent.parameters())
            != registration.parameter_count
        ):
            raise ValueError("agent PUCT checkpoint parameter count drift")
    return player, obs_space
