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
from managym.policy import PolicyRegistry, PolicyRunner, PolicySpec

if TYPE_CHECKING:
    from manabot.sim.search.runtime import RetainedCheckpointRegistration


def CheckpointVillain(
    path: str,
    *,
    deterministic: bool = False,
    player_configs: Sequence[managym.PlayerConfig] | None = None,
) -> PolicyRunner:
    """Deprecated checkpoint loader; use a PolicySpec (removed in 0.4.0)."""
    from manabot.sim.policy_provider import provider

    warnings.warn(
        "CheckpointVillain is deprecated; use PolicySpec and PolicyRegistry (removed in 0.4.0)",
        DeprecationWarning,
        stacklevel=2,
    )
    registry = PolicyRegistry(providers={"manabot": provider})
    key = registry.register(
        PolicySpec("manabot", path, {"deterministic": deterministic})
    )
    return registry.load(key, player_configs or (), seat=1, seed=0)


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
