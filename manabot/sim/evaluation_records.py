"""Canonical evaluation records and their offline audit view.

Only the capture owns setup and commitments. Evaluation adds private runner
metadata in context and per-decision audit evidence in annotations. The
Command-shaped view is derived for existing reports, never written for a new
ordinary game. Constructed scenarios and retained traces keep their reader.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from manabot.infra.identity import canonical_sha256

if TYPE_CHECKING:
    from managym.recording import GameRecord


def is_record(data: dict[str, Any]) -> bool:
    return data.get("schema") == "managym-game-v1"


def stored_game(game: dict[str, Any]) -> dict[str, Any]:
    """Return the sole persisted trajectory, or an unchanged retained trace."""
    return game.get("record", game)


def bind_record(game: dict[str, Any], record: GameRecord) -> None:
    """Freeze settled runner metadata alongside the capture's native inputs."""
    metadata = evaluation_metadata(game)
    record.context["evaluation"] = metadata
    game["record"] = record.model_dump(mode="json", by_alias=True)
    game["game_trace_sha256"] = record.sha256


def evaluation_metadata(game: dict[str, Any]) -> dict[str, Any]:
    return {
        k: deepcopy(v)
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


def audit_view(data: dict[str, Any], *, digest: str | None = None) -> dict[str, Any]:
    """Derive a retained-trace-shaped view carrying its canonical ``record``.

    Projection invokes no engine and never rewrites persisted evidence.
    """
    if not is_record(data):
        return data
    # Offline reports project retained JSON without probing the native runtime.
    # Identity-sensitive consumers pass the validated GameRecord's digest.
    if len(data["games"]) != 1:
        raise ValueError("evaluation audit requires exactly one game")
    game = deepcopy(data["context"]["evaluation"])
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
    end = inputs.get("end")
    game["terminated"] = end is not None and end["status"] == "complete"
    game["winner"] = end.get("winner") if game["terminated"] else None
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
        command = {
            "command_id": f"{game['match_id']}:{index}",
            "match_id": game["match_id"],
            "expected_revision": index,
            "prompt_id": index,
            "offer_id": decision["chosen_offer"]["id"],
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


def decision_annotation(decision: dict[str, Any]) -> dict[str, object]:
    """Retain runner evidence; native inputs and digests have their own owner."""
    return {
        "extra": {
            "evaluation": {
                k: v
                for k, v in decision.items()
                if k
                not in {
                    "command",
                    "command_sha256",
                    "pre_state_digest",
                    "post_state_digest",
                }
            }
        }
    }
