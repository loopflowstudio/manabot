"""Bind one viewer's committed native facts to their Academy history.

The engine supplies the subjects, names and cause at the moment of the event.
Each projector owns its viewer's sequence numbers. Untold facts inherit their
nearest told ancestor so a trigger can still refer to it in a later step.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from managym.experience import PresentationEvent


class PresentationLine(TypedDict, total=False):
    kind: dict[str, Any]
    labels: dict[str, dict[str, str]]
    combat: bool


class SinkFact(TypedDict):
    id: int
    cause: int | None
    rule: str | None
    lines: list[PresentationLine]
    position: int | None


# Importance, duration in milliseconds and optional sound for the table.
CUES: dict[str, tuple[str, int, str | None]] = {
    "cast": ("emphasized", 650, "spell.cast"),
    "targeted": ("normal", 500, None),
    "resolved": ("emphasized", 450, "spell.resolve"),
    "countered": ("normal", 450, None),
    "damage": ("critical", 700, "damage.fire"),
    "combat_damage": ("critical", 500, "combat.damage"),
    "land_played": ("normal", 400, None),
    "discarded": ("normal", 450, None),
    "revealed": ("normal", 450, None),
    "drew": ("normal", 450, None),
    "ability_resolved": ("normal", 450, None),
    "activated": ("emphasized", 550, None),
    "triggered": ("normal", 550, None),
    "counters": ("normal", 450, None),
    "passed": ("ambient", 0, None),
    "attack_group": ("emphasized", 500, "combat.attack"),
    "blocked": ("normal", 450, "combat.block"),
    "died": ("critical", 650, "creature.died"),
    "turn_started": ("emphasized", 350, "turn.start"),
}


class PresentationProjector:
    """Append events at consecutive addresses on one viewer's track."""

    def __init__(self) -> None:
        self.next_seq = 0
        self._ancestor: dict[int, int] = {}
        self._roots: dict[int, int] = {}
        self._positions: dict[int, int | None] = {}

    def position(self, seq: int) -> int | None:
        """Index of this step's kept board, or None for its final board."""
        return self._positions[seq]

    def project(
        self,
        payload: str,
        *,
        from_revision: int,
        to_revision: int,
        caused_by: str | None,
    ) -> list[dict[str, Any]]:
        facts: list[SinkFact] = json.loads(payload)
        events: list[dict[str, Any]] = []
        self._positions.clear()
        for fact in facts:
            parent = fact["cause"]
            cause = self._ancestor.get(parent) if parent is not None else None
            if cause is not None:
                self._ancestor[fact["id"]] = cause
            for index, line in enumerate(fact["lines"]):
                kind = line["kind"]
                name = kind["kind"]
                # Multiple simultaneous discards or reveals share a line, but
                # a new cause, board or intervening event starts a new one.
                previous = events[-1] if events else None
                if (
                    previous is not None
                    and name in {"discarded", "revealed"}
                    and previous["kind"]["kind"] == name
                    and previous["kind"]["player"] == kind["player"]
                    and previous.get("cause") == cause
                    and self.position(previous["seq"]) == fact["position"]
                ):
                    key = "objects" if name == "discarded" else "cards"
                    previous["kind"][key].extend(kind[key])
                    for table, names in line["labels"].items():
                        previous.setdefault("labels", {}).setdefault(table, {}).update(
                            names
                        )
                    self._ancestor[fact["id"]] = previous["seq"]
                    continue
                seq = self.next_seq
                importance, ms, sound = CUES[
                    "combat_damage" if line.get("combat") else name
                ]
                group = seq if cause is None else self._roots[cause]
                event = PresentationEvent.model_validate(
                    {
                        "seq": seq,
                        "from_revision": from_revision,
                        "to_revision": to_revision,
                        "caused_by": caused_by,
                        "group": group,
                        "importance": importance,
                        "suggested_ms": ms,
                        "sound": sound,
                        "kind": kind,
                        "cause": cause,
                        "rule": fact["rule"],
                        "labels": line["labels"] or None,
                    }
                )
                events.append(event.model_dump(mode="json"))
                self._positions[seq] = fact["position"]
                self._roots[seq] = group
                self.next_seq += 1
                if index == 0:
                    self._ancestor[fact["id"]] = seq
                    if name == "cast":
                        # The cast is also the parent of each of its targets.
                        cause = seq
        return events
