"""Bounded, viewer-specific Arena log projection from retained native facts.

This is an offline analysis interchange, not an Arena replay or 17Lands server
submission. It carries own deck/hand, publicly visible zone objects, turns and
results. No mulligan prompt exists in these worlds; none is fabricated. The
pinned client consequently reports its unavailable mulligan count (-1)."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import JsonValue, model_validator

from manabot.infra.identity import StrictModel
from manabot.sim.card_evidence import CardSnapshot, GameEvidence, Seat

EXPORT_VERSION = "arena-card-log-v1"

PUBLIC_ZONES = {"Battlefield", "Graveyard", "Stack", "Exile", "Command"}


class ArenaPrinting(StrictModel):
    """Retained Scryfall printing response supplies the actual Arena ID.

    The native content-bound key is an explicit authored binding to a printing;
    names verify that binding but are never the join key.
    """

    card: str
    scryfall: dict[str, JsonValue]

    @model_validator(mode="after")
    def verified(self) -> ArenaPrinting:
        data = self.scryfall
        if (
            data.get("object") != "card"
            or type(data.get("arena_id")) is not int
            or data["arena_id"] <= 0
            or not isinstance(data.get("name"), str)
            or data.get("uri") != f"https://api.scryfall.com/cards/{data.get('id')}"
            or not data.get("set")
            or not data.get("collector_number")
        ):
            raise ValueError(
                "mapping requires a retained Scryfall Arena printing response"
            )
        return self


class CardMapping(StrictModel):
    version: Literal[1] = 1
    printings: tuple[ArenaPrinting, ...]

    @model_validator(mode="after")
    def unique(self) -> CardMapping:
        if len({p.card for p in self.printings}) != len(self.printings):
            raise ValueError("duplicate native card mapping")
        return self


def export_arena_log(game: GameEvidence, viewer: Seat, mapping: CardMapping) -> str:
    if viewer not in (0, 1):
        raise ValueError("viewer must be 0 or 1")
    if game.outcome not in ("win_loss", "draw") or not game.replay.get("passed"):
        raise ValueError("export requires a replay-verified completed game")
    printings = {p.card: p.scryfall for p in mapping.printings}
    content = game.cards.capture.content_digest

    def arena_id(card: CardSnapshot) -> int:
        key = f"{content}:{card.definition}"
        if key not in printings or printings[key]["name"] != card.name:
            raise ValueError(
                f"verified Arena printing unavailable: {key} ({card.name})"
            )
        return int(printings[key]["arena_id"])

    lines = [
        f"# Manabot {EXPORT_VERSION}; synthetic offline projection; never submit to 17Lands"
    ]

    def emit(blob: dict[str, JsonValue], label: str = "") -> None:
        lines.append(
            "[UnityCrossThreadLogger]" + label + json.dumps(blob, separators=(",", ":"))
        )

    def gre(message: dict[str, JsonValue]) -> None:
        emit({"greToClientEvent": {"greToClientMessages": [message]}})

    initial = game.cards.capture.initial
    gre(
        {
            "type": "GREMessageType_ConnectResp",
            "systemSeatIds": [viewer + 1],
            "connectResp": {
                "deckMessage": {
                    "deckCards": [
                        arena_id(c)
                        for c in initial
                        if c.owner == viewer and c.origin == "deck"
                    ],
                    "sideboardCards": [
                        arena_id(c)
                        for c in initial
                        if c.owner == viewer and c.origin == "outside"
                    ],
                }
            },
        }
    )
    cards = {c.copy_id: c for c in initial}
    objects: dict[tuple[int, int], int] = {}

    def state(turn: int, opening: bool = False) -> None:
        visible = [
            c
            for c in cards.values()
            if c.zone in PUBLIC_ZONES or c.owner == viewer and c.zone == "Hand"
        ]
        for card in visible:
            objects.setdefault((card.copy_id, card.incarnation), len(objects) + 1)
        game_objects: list[JsonValue] = [
            {
                "type": "GameObjectType_Card",
                "ownerSeatId": c.owner + 1,
                "instanceId": objects[c.copy_id, c.incarnation],
                "overlayGrpId": arena_id(c),
            }
            for c in visible
        ]
        # Opponent hand/library identities and even hidden object IDs are absent.
        zones: list[JsonValue] = []
        for seat in (0, 1):
            for zone_id, zone in enumerate(
                (
                    "Hand",
                    "Library",
                    "Battlefield",
                    "Graveyard",
                    "Stack",
                    "Exile",
                    "Command",
                ),
                start=seat * 10 + 1,
            ):
                zone_cards = [
                    c for c in cards.values() if c.owner == seat and c.zone == zone
                ]
                visible_zone = zone in PUBLIC_ZONES or seat == viewer and zone == "Hand"
                ids = (
                    [objects[c.copy_id, c.incarnation] for c in zone_cards]
                    if visible_zone
                    else []
                )
                if opening and seat == viewer and zone == "Hand":
                    ids = [
                        objects[c, cards[c].incarnation]
                        for c in game.cards.capture.opening[viewer]
                    ]
                zones.append(
                    {
                        "zoneId": zone_id,
                        "type": "ZoneType_" + zone,
                        "ownerSeatId": seat + 1,
                        "objectInstanceIds": ids,
                        "objectCount": len(zone_cards),
                    }
                )
        turn_info: dict[str, JsonValue] = {
            "turnNumber": turn,
        }
        if opening:
            turn_info.update(
                phase="Phase_Beginning", step="Step_Upkeep", activePlayer=1
            )
        gre(
            {
                "type": "GREMessageType_GameStateMessage",
                "systemSeatIds": [viewer + 1],
                "gameStateMessage": {
                    "gameInfo": {"matchID": game.game},
                    "turnInfo": turn_info,
                    "gameObjects": game_objects,
                    "zones": zones,
                },
            }
        )

    state(1, opening=True)
    for move in game.cards.capture.moves:
        cards[move.card.copy_id] = move.card
        state(move.turn)
    state(game.cards.turn)
    # The client needs >5 actual state messages to retain game data. Very short
    # synthetic fixtures are outside this subset; do not pad invented transitions.
    if len(game.cards.capture.moves) + 2 <= 5:
        raise ValueError("game too short for pinned client's retained-history subset")
    result = "ResultType_Draw" if game.winner is None else "ResultType_WinLoss"
    winner = None if game.winner is None else game.winner + 1
    emit(
        {
            "StartingTeamId": 1,
            "WinningTeamId": winner,
            "WinningType": result,
            "WinningReason": game.trace.data.get("termination_reason"),
            "GameNumber": 1,
        },
        "LogBusinessEvents ",
    )
    emit(
        {
            "matchGameRoomStateChangedEvent": {
                "gameRoomInfo": {
                    "finalMatchResult": {
                        "resultList": [
                            {
                                "scope": "MatchScope_Game",
                                "winningTeamId": winner,
                                "result": result,
                                "reason": game.trace.data.get("termination_reason"),
                            }
                        ]
                    }
                }
            }
        }
    )
    return "\n".join(lines) + "\n"
