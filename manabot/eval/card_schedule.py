"""Bind each card game to its scheduled deal and arena matrix, and to its runtime.

CardConditions is shared by production and audit. A game is admitted when its
match is played; an audit re-checks it, with runtime identity taken from the
eval run's declared environment, never the reader's installed runtime."""

from __future__ import annotations

from typing import Literal

from etude.eval.records import Environment, Match

from manabot.eval.card_metrics import METRIC_VERSION
from manabot.eval.models import EvaluationKey
from manabot.infra.config import MatchConfig
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.sim.card_evidence import CAPTURE_VERSION, GameEvidence
from manabot.sim.matchup import AuthoredMatchup, selected_decks
from manabot.sim.registration import PlayerRegistration

RUNTIME_FIELDS = (
    "world",
    "engine_source_sha256",
    "content_manifest_sha256",
    "content_digest",
)


class CardConditions(StrictModel):
    deal: int
    key: EvaluationKey
    authored_matchup: AuthoredMatchup | None
    game_seconds: float
    max_commands: int
    capture_version: Literal[1] = CAPTURE_VERSION
    metric_version: Literal["gih-copies-v1"] = METRIC_VERSION


def card_match_players(match: Match) -> tuple[PlayerRegistration, PlayerRegistration]:
    """Resolve the same registered participant pair for execution and admission."""
    if len(match.players) != 2 or match.games != 8:
        raise ValueError("card evaluation schedules eight legs per deal")
    first, second = (
        PlayerRegistration.model_validate(player.config["registration"])
        for player in match.players
    )
    if (first.player_id, second.player_id) != tuple(a.name for a in match.players):
        raise ValueError("scheduled participant differs from registration")
    return first, second


def admit_card_game(game: GameEvidence, match: Match) -> tuple[str, str]:
    """Bind one game to its match's deal, seating, decks, setup and ABIs.

    Returns participant names in Etude row order. Each game is admitted on its
    own; the other legs of its deal need not exist.
    """
    conditions = CardConditions.model_validate(match.conditions)
    players = card_match_players(match)
    names = (players[0].player_id, players[1].player_id)
    if game.deal != conditions.deal:
        raise ValueError("card evidence differs from scheduled deal")
    matrix = conditions.authored_matchup
    seats = names if game.leg % 2 == 0 else names[::-1]
    decks = matrix.decks(game.leg) if matrix else selected_decks(game.leg)
    setup = (
        matrix.match(game.leg)
        if matrix
        else MatchConfig.authored("ur-lessons-vs-gw-allies", *decks)
    ).model_copy(update={"hero": seats[0], "villain": seats[1]})
    if game.seat_players != seats or game.seat_decks != decks or game.setup != setup:
        raise ValueError("card evidence differs from scheduled seat/deck assignment")
    runtime = game.trace.data.get("runtime_identities")
    setup_data = setup.model_dump()
    if (
        not isinstance(runtime, dict)
        or runtime.get("matchup") != setup_data
        or runtime.get("matchup_sha256") != canonical_sha256(setup_data)
        or any(
            runtime.get(field) != getattr(player, field)
            for player in players
            for field in ("observation_abi_sha256", "action_abi_sha256")
        )
    ):
        raise ValueError("card evidence differs from scheduled runtime setup/ABI")
    return names


def admit_card_runtime(game: GameEvidence, environment: Environment) -> None:
    """Bind a game to the runtime and content its eval run declared."""
    identity = {field: environment.config.get(field) for field in RUNTIME_FIELDS}
    if (
        any(not isinstance(value, str) or not value for value in identity.values())
        or environment.files != canonical_sha256(identity)
        or environment.config.get("capture_version") != CAPTURE_VERSION
        or environment.config.get("metric_version") != METRIC_VERSION
    ):
        raise ValueError("incompatible card environment")
    runtime = game.trace.data.get("runtime_identities")
    if not isinstance(runtime, dict) or any(
        runtime.get(field) != value for field, value in identity.items()
    ):
        raise ValueError("card evidence differs from declared runtime/content identity")
