"""A frozen flat policy opponent for the native text-game proof.

Checkpoint loading owns world/input admission. This adapter preserves the saved
encoder and samples the trained policy on CPU, with an isolated RNG per decision.
It never upgrades legacy observations or silently substitutes random actions.
"""

import hashlib
from pathlib import Path

from pydantic import BaseModel
import torch

from manabot.model.checkpoint import load_checkpoint
from manabot.model.world import validate_agent_setup
from manabot.semantic.text_observation import TextDecision
from manabot.verify.util import _select_agent_action
import managym


class CheckpointIdentity(BaseModel):
    path: str
    sha256: str
    size_bytes: int
    parameters: int
    policy_input_version: int
    policy_history_version: int
    seed: int
    deterministic: bool = False
    device: str = "cpu"


class CheckpointOpponent:
    def __init__(
        self, path: Path, players: list[managym.PlayerConfig], seed: int
    ) -> None:
        self.agent, self.space = load_checkpoint(str(path))
        if self.agent.config.compound_decisions or self.agent.belief_count_buckets:
            raise ValueError("text-game opponent requires an ordinary flat policy")
        validate_agent_setup(self.agent, players)
        self.identity = CheckpointIdentity(
            path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
            parameters=sum(p.numel() for p in self.agent.parameters()),
            policy_input_version=self.space.encoder.config.policy_input_version,
            policy_history_version=self.space.encoder.config.policy_history_version,
            seed=seed,
        )
        self.decisions = 0

    def select(self, env: managym.Env, decision: TextDecision) -> str:
        raw = env.observation_for_player(env.current_agent_index())
        if len(raw.action_space.actions) != len(decision.actions):
            raise ValueError("policy action space differs from the captured decision")
        obs = self.space.encode(raw)
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(self.identity.seed + self.decisions)
            index = _select_agent_action(self.agent, obs, deterministic=False)
        if not 0 <= index < len(decision.actions):
            raise ValueError("policy selected an unavailable action")
        self.decisions += 1
        return decision.actions[index].reference
