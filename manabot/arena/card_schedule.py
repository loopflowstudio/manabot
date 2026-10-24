"""Bind retained card games to Etude's scheduled deal and arena matrix.

CardConditions is shared by production and read-time admission. Runtime identity
comes from the retained Etude environment, never the reader's installed runtime.
"""

from typing import Literal

from etude.eval.records import Environment, Match

from manabot.arena.card_evidence import CAPTURE_VERSION, METRIC_VERSION, GameEvidence
from manabot.arena.match import selected_decks
from manabot.arena.matchup import AuthoredMatchup
from manabot.arena.models import (
    ArenaKey,
    PlayerRegistration,
    StrictModel,
    canonical_sha256,
)
from manabot.infra.config import MatchConfig

RUNTIME_FIELDS = (
    "world",
    "engine_source_sha256",
    "content_manifest_sha256",
    "content_digest",
)


class CardConditions(StrictModel):
    deal: int
    key: ArenaKey
    authored_matchup: AuthoredMatchup | None
    game_seconds: float
    max_commands: int
    capture_version: Literal[1] = CAPTURE_VERSION
    metric_version: Literal["gih-copies-v1"] = METRIC_VERSION


def card_match_players(match: Match) -> tuple[PlayerRegistration, PlayerRegistration]:
    """Resolve the same registered participant pair for execution and admission."""
    if len(match.agents) != 2 or match.games != 8:
        raise ValueError("card evaluation requires a complete eight-leg deal")
    first, second = (
        PlayerRegistration.model_validate(agent.config["registration"])
        for agent in match.agents
    )
    if (first.player_id, second.player_id) != tuple(a.name for a in match.agents):
        raise ValueError("scheduled participant differs from registration")
    return first, second


def admit_card_match(
    games: list[GameEvidence], match: Match, environment: Environment
) -> tuple[str, str]:
    """Validate the full matrix and return participant names in Etude row order."""
    conditions = CardConditions.model_validate(match.conditions)
    players = card_match_players(match)
    if (
        len(games) != match.games
        or {game.leg for game in games} != set(range(8))
        or len({game.game for game in games}) != 8
    ):
        raise ValueError(
            "completed match requires unique evidence for every scheduled leg"
        )
    identity = {field: environment.config.get(field) for field in RUNTIME_FIELDS}
    if (
        any(not isinstance(value, str) or not value for value in identity.values())
        or environment.files != canonical_sha256(identity)
        or identity["world"] != conditions.key.world
        or environment.config.get("capture_version") != conditions.capture_version
        or environment.config.get("metric_version") != conditions.metric_version
    ):
        raise ValueError("incompatible scheduled card environment")
    names = (players[0].player_id, players[1].player_id)
    matrix = conditions.authored_matchup
    for game in games:
        if game.deal != conditions.deal:
            raise ValueError("card evidence differs from scheduled deal")
        seats = names if game.leg % 2 == 0 else names[::-1]
        decks = matrix.decks(game.leg) if matrix else selected_decks(game.leg)
        setup = (
            matrix.match(game.leg)
            if matrix
            else MatchConfig.authored("ur-lessons-vs-gw-allies", *decks)
        ).model_copy(update={"hero": seats[0], "villain": seats[1]})
        if (
            game.seat_players != seats
            or game.seat_decks != decks
            or game.setup != setup
        ):
            raise ValueError(
                "card evidence differs from scheduled seat/deck assignment"
            )
        runtime = game.trace.data.get("runtime_identities")
        if not isinstance(runtime, dict) or any(
            runtime.get(field) != value for field, value in identity.items()
        ):
            raise ValueError(
                "card evidence differs from scheduled runtime/content identity"
            )
        setup_data = setup.model_dump()
        if (
            runtime.get("matchup") != setup_data
            or runtime.get("matchup_sha256") != canonical_sha256(setup_data)
            or any(
                runtime.get(field) != getattr(player, field)
                for player in players
                for field in ("observation_abi_sha256", "action_abi_sha256")
            )
        ):
            raise ValueError("card evidence differs from scheduled runtime setup/ABI")
    return names
