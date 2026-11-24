"""Training capabilities discovered by Academy through distribution metadata.

This module never imports Academy; the engine owns the callable and retained
output types. Heavy model imports happen only when a capability is requested.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
import warnings

import managym
from managym.bots import (
    BeliefSummary,
    BotArtifactError,
    BotDecision,
    VerifiedArena,
)
from managym.policy import PolicyRunner, PolicySpec

if TYPE_CHECKING:
    from manabot.sim.search.runtime import RetainedCheckpointRegistration


class CheckpointVillain:
    """Retained checkpoint loader, delegating execution to an independent provider session."""

    def __init__(
        self,
        path: str,
        *,
        deterministic: bool = False,
        player_configs: Sequence[managym.PlayerConfig] | None = None,
    ) -> None:
        from manabot.model.world import validate_agent_setup
        from manabot.sim.policy_provider import BaselineSession, provider

        warnings.warn(
            "CheckpointVillain is deprecated; use PolicySpec and PolicyRegistry (removed in 0.3.0)",
            DeprecationWarning,
            stacklevel=2,
        )
        spec = PolicySpec("manabot", path, {"deterministic": deterministic})
        # The retained loader permits omitted setup; Academy itself admits every
        # explicit setup through PolicyRegistry before constructing a runner.
        session = BaselineSession(spec, seed=0)
        try:
            descriptor = provider.policy(spec).describe()
        except BaseException:
            session.close()
            raise
        self._runner = PolicyRunner(session, descriptor, seat=1)
        assert session.agent is not None and session.space is not None
        self.agent = session.agent
        self.obs_space = session.space
        self.deterministic = deterministic
        try:
            if player_configs is not None:
                validate_agent_setup(self.agent, player_configs)
        except BaseException:
            self._runner.close()
            raise

    @property
    def last_decision(self) -> BotDecision | None:
        return self._runner.last_decision

    def __call__(self, env: managym.Env, obs: managym.Observation) -> int:
        return self._runner(env, obs)

    def close(self) -> None:
        self._runner.close()


class TrainingBots:
    """Optional historical replay/advice capabilities, independent of players."""

    name = "manabot"

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
        from manabot.sim.search.local_update_receipt import LocalUpdateReceipt

        receipt = replay_games([record])
        if receipt.trace_mismatches:
            raise ValueError(
                "Game trace digest mismatches: the file was edited after recording."
            )
        if record.get("failure") is not None:
            raise ValueError(
                "This evaluation game retained a failure; import rejected."
            )
        if not receipt.passed:
            mismatches = ", ".join(
                f"{name}={value}"
                for name, value in receipt.to_dict().items()
                if name not in {"games", "decisions", "passed"} and value
            )
            raise ValueError(
                f"Replay verification failed: {mismatches}. "
                "This revision may deal or build a different game; open the trace with the revision that wrote it."
            )
        arena, _ = replay_environment(record)
        decisions = []
        for expected in record["decisions"]:
            index = int(expected["command"]["offer_id"])
            decision = BotDecision(
                index,
                unavailable="Policy, value and search outputs were not retained in this recorded game.",
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
