"""Bounded, viewer-specific Arena log projection from retained native facts.

This is an offline analysis interchange, not an Arena replay or 17Lands server
submission. It carries own deck/hand, publicly visible zone objects, turns and
results. No mulligan prompt exists in these worlds; none is fabricated. The
pinned client consequently reports its unavailable mulligan count (-1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

import managym
from managym.card_capture import CapturedCards, CardSnapshot, Seat
from managym.recording import GameRecord
from managym.replay import walk

EXPORT_VERSION = "arena-card-log-v1"

PUBLIC_ZONES = {"Battlefield", "Graveyard", "Stack", "Exile", "Command"}


class ArenaPrinting(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
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


class CardMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    printings: tuple[ArenaPrinting, ...]

    @model_validator(mode="after")
    def unique(self) -> CardMapping:
        if len({p.card for p in self.printings}) != len(self.printings):
            raise ValueError("duplicate native card mapping")
        return self


def export_capture(
    captured: CapturedCards,
    viewer: Seat,
    mapping: CardMapping,
    *,
    match_id: str,
    winner: int | None,
    reason: str | None = None,
) -> str:
    """Project verified native facts; retained evaluation consumers share this writer."""
    if viewer not in (0, 1):
        raise ValueError("viewer must be 0 or 1")
    printings = {p.card: p.scryfall for p in mapping.printings}
    content = captured.capture.content_digest

    def arena_id(card: CardSnapshot) -> int:
        key = f"{content}:{card.definition}"
        if key not in printings:
            return 0
        if card.name not in str(printings[key]["name"]).split(" // "):
            raise ValueError(f"verified Arena printing differs: {key} ({card.name})")
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

    initial = captured.capture.initial
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
                        for c in captured.capture.opening[viewer]
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
                    "gameInfo": {"matchID": match_id},
                    "turnInfo": turn_info,
                    "gameObjects": game_objects,
                    "zones": zones,
                },
            }
        )

    state(1, opening=True)
    for move in captured.capture.moves:
        cards[move.card.copy_id] = move.card
        state(move.turn)
    state(captured.turn)
    # The client needs >5 actual state messages to retain game data. Very short
    # synthetic fixtures are outside this subset; do not pad invented transitions.
    if len(captured.capture.moves) + 2 <= 5:
        raise ValueError("game too short for pinned client's retained-history subset")
    result = "ResultType_Draw" if winner is None else "ResultType_WinLoss"
    winner = None if winner is None else winner + 1
    emit(
        {
            "StartingTeamId": 1,
            "WinningTeamId": winner,
            "WinningType": result,
            "WinningReason": reason,
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
                                "reason": reason,
                            }
                        ]
                    }
                }
            }
        }
    )
    return "\n".join(lines) + "\n"


def coverage() -> dict[str, int]:
    """Current-world definition coverage, including unmapped faces and tokens."""
    ids = json.loads(Path(__file__).with_name("arena_ids.json").read_text())
    definitions = managym.content_pool("world")["definitions"]
    mapped = sum(
        bool(ids.get(row["scryfall_id"], {}).get("arena_id"))
        for row in definitions
        if row["scryfall_id"]
    )
    return {
        "definitions": len(definitions),
        "mapped": mapped,
        "unmapped": len(definitions) - mapped,
    }


def _world_mapping() -> CardMapping:
    ids = json.loads(Path(__file__).with_name("arena_ids.json").read_text())
    pool = managym.content_pool("world")
    return CardMapping(
        printings=tuple(
            ArenaPrinting(
                card=f"{pool['content_digest']}:{row['card_def_id']}",
                scryfall={
                    "object": "card",
                    "id": row["scryfall_id"],
                    "uri": f"https://api.scryfall.com/cards/{row['scryfall_id']}",
                    **ids[row["scryfall_id"]],
                },
            )
            for row in pool["definitions"]
            if ids.get(row["scryfall_id"], {}).get("arena_id")
        )
    )


def export_log(
    record: GameRecord, viewer: Seat = 0, *, mapping: CardMapping | None = None
) -> str:
    """Export a completed, verified game without a model or training package."""
    replay = walk(record, capture_cards=True)
    for _ in replay:
        pass
    if replay.stop is None or replay.stop.status != "complete":
        raise ValueError(f"export requires a completed verified game: {replay.stop}")
    cards = replay.cards()
    return export_capture(
        cards,
        viewer,
        mapping or _world_mapping(),
        match_id=record.sha256,
        winner=replay.stop.winner,
        reason=record.games[0].end.reason if record.games[0].end else None,
    )
