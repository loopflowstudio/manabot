"""Canonical evaluation records and their offline audit view.

Only the capture owns setup and commitments. Evaluation adds private runner
metadata in context and per-decision audit evidence in annotations. The
Command-shaped view is derived for existing reports, never written for a new
ordinary game. Constructed scenarios and retained traces keep their reader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

from manabot.infra.identity import canonical_sha256

if TYPE_CHECKING:
    from manabot.sim.registration import PlayerRegistration
    from managym.recording import GameRecord


def set_execution_limits(
    game: dict[str, Any],
    registrations: Sequence[PlayerRegistration],
    max_commands: int,
    game_seconds: float,
) -> None:
    """Record a leg's execution budget and players on the game being settled."""
    game["execution_limits"] = {"commands": max_commands, "seconds": game_seconds}
    game["registrations"] = [r.model_dump(mode="json") for r in registrations]


def is_record(data: dict[str, Any]) -> bool:
    return data.get("schema") == "managym-game-v1"


def stored_game(game: dict[str, Any]) -> dict[str, Any]:
    """Return the sole persisted trajectory, or an unchanged retained trace."""
    return game.get("record", game)


def bind_record(game: dict[str, Any], record: GameRecord) -> None:
    """Freeze settled runner metadata alongside the capture's native inputs."""
    record.context["evaluation"] = deepcopy(evaluation_metadata(game))
    game["record"] = record.model_dump(mode="json", by_alias=True)
    game["game_trace_sha256"] = record.sha256


def evaluation_metadata(game: dict[str, Any]) -> dict[str, Any]:
    """Select the runner fields copied into capture context."""
    return {
        k: v
        for k, v in game.items()
        if k
        not in {
            "record",
            "decisions",
            "pending_decision",
            "game_trace_sha256",
            "capture_path",
            "winner",
            "terminated",
            "initial_state_digest",
        }
    }


def finalize_game(game: dict[str, Any]) -> None:
    from managym.recording import GameRecord

    if "record" in game:
        bind_record(game, GameRecord.model_validate(game["record"]))
    else:
        game.pop("game_trace_sha256", None)
        game["game_trace_sha256"] = canonical_sha256(game)


@dataclass(frozen=True)
class DecisionIdentity:
    """The runner's identity and sampling coordinates for one decision."""

    actor: int
    revision: int
    player_id: str
    player_seed: int
    ordinal: int

    @classmethod
    def for_actor(
        cls, game: dict[str, Any], actor: int, revision: int, counts: Mapping[str, int]
    ) -> DecisionIdentity:
        player = game["seat_players"][actor]
        return cls(
            actor, revision, player, game["player_seeds"][player], counts[player]
        )

    @property
    def policy_seed(self) -> int:
        return (self.player_seed + self.ordinal) % (2**63)

    def evidence(self, elapsed: float) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "revision": self.revision,
            "player_id": self.player_id,
            "player_seed": self.player_seed,
            "player_decision_ordinal": self.ordinal,
            "policy_rng_seed": self.policy_seed,
            "latency_seconds": elapsed,
        }


def evaluation_inputs(data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    # Offline reports project retained JSON without probing the native runtime.
    if len(data["games"]) != 1:
        raise ValueError("evaluation audit requires exactly one game")
    inputs = data["games"][0]
    count = inputs.get("committed")
    if count is None:
        count = len(inputs["actions"])
    if not isinstance(count, int) or not 0 <= count <= len(inputs["actions"]):
        raise ValueError("invalid evaluation commitment count")
    if len(inputs["annotations"]) != len(inputs["actions"]) or len(
        inputs["digests"]
    ) != len(inputs["actions"]):
        raise ValueError("evaluation inputs lack aligned digests or annotations")
    return inputs, count


def evaluation_outcome(
    data: dict[str, Any], inputs: dict[str, Any], committed: int
) -> dict[str, Any]:
    metadata = data["context"]["evaluation"]
    end = inputs.get("end")
    # The record keeps the engine's ending; a runner failure (context) voids the
    # result even after a terminal step, as retained failure rows do.
    terminated = (
        metadata.get("failure") is None
        and end is not None
        and end["status"] == "complete"
    )
    game = {
        **metadata,
        "terminated": terminated,
        "winner": end.get("winner") if terminated else None,
    }
    if committed < len(inputs["actions"]) and game.get("failure") is None:
        game["failure"] = (
            f"{data['error_class']}: {data['error']}"
            if data.get("error_class") is not None
            else "execution_outcome_unknown"
        )
    return game


def recorded_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Read a retained game's setup and outcome without projecting its actions."""
    return (
        evaluation_outcome(data, *evaluation_inputs(data)) if is_record(data) else data
    )


def audit_view(data: dict[str, Any], *, digest: str | None = None) -> dict[str, Any]:
    """Derive a retained-trace-shaped view carrying its canonical ``record``.

    Projection invokes no engine and never rewrites persisted evidence.
    """
    if not is_record(data):
        return data
    inputs, count = evaluation_inputs(data)
    game = deepcopy(evaluation_outcome(data, inputs, count))
    end = inputs.get("end")
    game["initial_state_digest"] = (
        inputs["digests"][0]
        if inputs["digests"]
        else end.get("digest")
        if end
        else None
    )
    decisions = []
    for index, annotation in enumerate(inputs["annotations"]):
        if annotation is None or "evaluation" not in annotation["extra"]:
            raise ValueError("evaluation record lacks decision audit metadata")
        decision = deepcopy(annotation["extra"]["evaluation"])
        action = inputs["actions"][index]
        offer_id = (
            decision["chosen_offer"]["id"]
            if "chosen_offer" in decision
            else action
            if isinstance(action, int)
            else json.loads(action)["offer_id"]
        )
        command = {
            "command_id": f"{game['match_id']}:{index}",
            "match_id": game["match_id"],
            "expected_revision": index,
            "prompt_id": index,
            "offer_id": offer_id,
            "answers": [],
        }
        decision["command"] = command
        decision["command_sha256"] = canonical_sha256(command)
        decision["pre_state_digest"] = inputs["digests"][index]
        if index < count:
            decision["post_state_digest"] = (
                inputs["digests"][index + 1]
                if index + 1 < len(inputs["digests"])
                else inputs["end"]["digest"]
                if inputs.get("end") is not None
                else None
            )
            decisions.append(decision)
        else:
            game["pending_decision"] = decision
    game["decisions"] = decisions
    game["record"] = data
    if digest is not None:
        game["game_trace_sha256"] = digest
    return game
