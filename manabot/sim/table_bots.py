"""Training capabilities discovered by Academy through distribution metadata.

This module never imports Academy; the engine owns the callable and retained
output types. Heavy model imports happen only when a capability is requested.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import managym
from managym.bots import (
    BeliefSummary,
    BotArtifactError,
    BotDecision,
    BotPolicy,
    VerifiedArena,
)

if TYPE_CHECKING:
    from manabot.model.catalog import CheckpointSummary, UnavailableEntry
    from manabot.sim.search.runtime import RetainedCheckpointRegistration


class CheckpointVillain:
    """A trained ActorCritic loaded from a training checkpoint (.pt).

    Torch and the model stack are imported lazily so that search/random games
    never pay the import cost.
    """

    def __init__(
        self,
        path: str,
        *,
        deterministic: bool = False,
        player_configs: Sequence[managym.PlayerConfig] | None = None,
    ) -> None:
        from manabot.backends.torch.checkpoint import load_checkpoint
        from manabot.model.world import validate_agent_setup

        self.agent, self.obs_space = load_checkpoint(path)
        if player_configs is not None:
            validate_agent_setup(self.agent, player_configs)
        self.deterministic = deterministic
        self.last_decision: BotDecision | None = None

    def __call__(self, env: managym.Env, obs: managym.Observation) -> int:
        del env
        self.last_decision = None
        # Policy weights score each action independently of the padded action
        # count. At inference preserve the entire rules offer list, including
        # decisions wider than the training batch's storage capacity.
        space = self.obs_space
        if len(obs.action_space.actions) > space.encoder.max_actions:
            from manabot.env.observation import ObservationSpace

            space = ObservationSpace(
                space.encoder.config.model_copy(
                    update={
                        "max_actions": len(obs.action_space.actions),
                    }
                )
            )
        encoded = space.encode(obs)
        import torch

        from manabot.verify.util import _tensor_observation

        with torch.no_grad():
            logits, value = self.agent(_tensor_observation(self.agent, encoded))
            distribution = torch.distributions.Categorical(logits=logits)
            action = int(
                (
                    logits.argmax(dim=-1)
                    if self.deterministic
                    else distribution.sample()
                ).item()
            )
            self.last_decision = BotDecision(
                action=action,
                probabilities=distribution.probs[
                    0, : len(obs.action_space.actions)
                ].tolist(),
                value=float(value.item()),
                value_semantic=self.agent.value_semantic or "undeclared critic units",
            )
        return action


class TrainingBots:
    name = "manabot"
    watch_players = ("random", "greedy", "heuristic", "search")

    def discover(
        self, paths: list[Path], runs_dir: Path
    ) -> tuple[list[CheckpointSummary], list[UnavailableEntry]]:
        from manabot.model.catalog import discover_checkpoints

        return discover_checkpoints(paths, runs_dir=runs_dir)

    def checkpoint_policy(
        self,
        path: str,
        *,
        deterministic: bool,
        player_configs: Sequence[managym.PlayerConfig],
    ) -> BotPolicy:
        return CheckpointVillain(
            path, deterministic=deterministic, player_configs=player_configs
        )

    def watch_policy(self, kind: str) -> BotPolicy:
        from manabot.env.env import Env
        from manabot.sim.anchors import ScriptedGreedyPlayer
        from manabot.sim.heuristic import HeuristicPlayer

        player = HeuristicPlayer() if kind == "heuristic" else ScriptedGreedyPlayer()

        def anchor(env: managym.Env, obs: managym.Observation) -> int:
            return player.act(cast(Env, SimpleNamespace(last_raw_obs=obs)), {})

        return anchor

    def checkpoint_registration(self, seed: int) -> RetainedCheckpointRegistration:
        from manabot.sim.search.runtime import (
            RetainedCheckpointMismatchError,
            RetainedCheckpointUnavailableError,
            retained_int7_policy_only_checkpoint,
        )

        try:
            return retained_int7_policy_only_checkpoint(seed)
        except RetainedCheckpointUnavailableError as exc:
            raise BotArtifactError("advisor_artifact_unavailable") from exc
        except RetainedCheckpointMismatchError as exc:
            raise BotArtifactError("advisor_artifact_mismatch") from exc

    def verify_arena(self, record: dict[str, Any]) -> VerifiedArena:
        from manabot.sim.replay import replay_environment, replay_games
        from manabot.sim.search.local_update import LocalUpdateReceipt

        receipt = replay_games([record])
        if not receipt.passed or record.get("failure") is not None:
            raise ValueError("Arena replay verification failed; import rejected")
        arena, _ = replay_environment(record)
        decisions = []
        for expected in record["decisions"]:
            index = int(expected["command"]["offer_id"])
            decision = BotDecision(
                index,
                unavailable="Policy, value and search outputs were not retained in this arena recording.",
            )
            if expected.get("local_update_receipt") is not None:
                retained = LocalUpdateReceipt.from_json(
                    expected["local_update_receipt"]
                )
                if retained.viewer != int(expected["actor"]):
                    raise ValueError(
                        "Arena search evidence differs from the acting offers"
                    )
                decision = BotDecision(
                    action=index,
                    probabilities=list(retained.target),
                    search_scores=list(retained.values),
                    belief=BeliefSummary(
                        retained.belief_model,
                        retained.belief_digest,
                        retained.condition_mass,
                        len(
                            {
                                (rollout.world_index, rollout.world_seed)
                                for rollout in retained.rollouts
                            }
                        ),
                    ),
                    unavailable="Search values are signed frozen-policy rollout estimates; no separate root critic was retained.",
                )
            decisions.append(decision)
        return VerifiedArena(
            arena._engine, arena.last_raw_obs, arena.match.to_rust(), decisions
        )


extension = TrainingBots()
