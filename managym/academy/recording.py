"""Bounded read projections over retained Academy recordings.

Recording owns positions, viewer event prefixes and legacy transition joins. It
never executes a game. ReadCache validates once per persisted revision; access
is checked separately on every request, including cache hits.
"""

from bisect import bisect_right
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Any, Literal

from managym.academy.replay_index import (
    CanonicalReplayV1,
    DecisionAddressV2,
    ReplayDecision,
    canonical_projection_sha256,
    load_canonical_replay,
    project_replay,
)
from managym.academy.trace import load_trace, prepare_trace_payload
from managym.experience import PresentationEvent


@dataclass(frozen=True)
class Position:
    key: str
    kind: Literal["choice", "consequences", "state"]
    event_index: int
    decision: ReplayDecision | None = None


class Recording:
    def __init__(self, trace_id: str, payload: dict[str, Any]) -> None:
        self.id = trace_id
        self.payload = payload
        self.human_seat = payload["config"].get("human_seat", 0)
        self.watch = bool(payload["config"].get("watch_players"))
        self.completed_mirror = bool(
            payload["config"].get("mirror") is not None
            and payload.get("end_reason") == "game_over"
            and payload["final_observation"].get("game_over")
        )
        # Historical formats need no new schema, but must still have a board
        # the shared renderer can read. Reject malformed records at this boundary.
        for observation in [
            payload["final_observation"],
            *(event["observation"] for event in payload["events"]),
        ]:
            if not isinstance(observation, dict) or not isinstance(
                observation.get("turn"), dict
            ):
                raise ValueError("recording_board_unavailable")
            if any(
                not isinstance(observation["turn"].get(key), str)
                for key in ("phase", "step")
            ):
                raise ValueError("recording_board_unavailable")
            for side in ("agent", "opponent"):
                player = observation.get(side)
                if not isinstance(player, dict) or any(
                    not isinstance(player.get(zone), list)
                    for zone in ("hand", "battlefield", "graveyard", "exile", "stack")
                ):
                    raise ValueError("recording_board_unavailable")
        self.replay: CanonicalReplayV1 | None = (
            load_canonical_replay(payload) if payload.get("canonical_replay") else None
        )
        self.positions: list[Position] = []
        self.study_sources: dict[int, dict[str, Any]] = {}
        self.tracks: dict[int, list[PresentationEvent]] = {}
        self.track_revisions: dict[int, list[int]] = {}
        decisions: dict[int, ReplayDecision] = {}
        if self.replay:
            indices = [
                i
                for i, event in enumerate(payload["events"])
                if not event.get("auto") and not event.get("control")
            ]
            if len(indices) != len(self.replay.decisions):
                raise ValueError("recording decisions and transitions differ")
            decisions = dict(zip(indices, self.replay.decisions, strict=True))
            for track in self.replay.presentation_tracks:
                self.tracks[track.viewer] = track.events
                self.track_revisions[track.viewer] = [
                    event.to_revision for event in track.events
                ]
        else:
            events = [
                PresentationEvent.model_validate(event)
                for row in payload["events"]
                for event in row.get("presentation", [])
            ]
            self.tracks[0] = events
            self.track_revisions[0] = [event.to_revision for event in events]
        for i, event in enumerate(payload["events"]):
            row = decisions.get(i)
            if row and (
                self.watch or self.completed_mirror or row.viewer == self.human_seat
            ):
                self.positions.append(
                    Position(
                        DecisionAddressV2.from_decision(self.replay, row).serialize(),
                        "choice",
                        i,
                        row,
                    )
                )
            elif not self.replay and not event.get("control"):
                self.positions.append(Position(f"legacy-choice:{i}", "choice", i))
            self.positions.append(Position(f"after:{i}", "consequences", i, row))
        self.positions.append(Position("latest", "state", len(payload["events"])))
        self.by_key = {position.key: position for position in self.positions}
        self.decisions = decisions
        # Revision joins come from retained decisions/events, never ordinals.
        self.revisions = {i: row.revision for i, row in decisions.items()}
        for i, event in enumerate(payload["events"]):
            if event.get("control"):
                self.revisions[i] = event["control"]["revision"]
            elif event.get("presentation"):
                self.revisions[i] = event["presentation"][0]["from_revision"]

        self.event_by_revision = {revision: i for i, revision in self.revisions.items()}

    def _board(self, position: Position) -> dict[str, Any]:
        # Reuse the established redaction boundary on only the selected board.
        return prepare_trace_payload(
            {
                "config": self.payload["config"],
                "events": [],
                "final_observation": self._raw_board(position),
                "end_reason": self.payload.get("end_reason"),
            }
        )["final_observation"]

    def _cursor(self, position: Position, viewer: int) -> int:
        if position.kind == "choice" and position.decision:
            return position.decision.presentation_cursor
        track = self.tracks.get(viewer, [])
        if position.kind == "state":
            return len(track)
        revision = self.revisions.get(position.event_index)
        if revision is None:
            # A legacy no-event transition has no semantic coordinates.
            return sum(
                len(row.get("presentation", []))
                for row in self.payload["events"][
                    : position.event_index + (position.kind == "consequences")
                ]
            )
        revisions = self.track_revisions.get(viewer, [])
        return (
            bisect_right(revisions, revision + 1)
            if position.kind == "consequences"
            else bisect_right(revisions, revision)
        )

    def metadata(self, position: Position) -> dict[str, Any]:
        row = position.decision
        turn = (
            row.frame.projection.turn.model_dump(mode="json")
            if position.kind == "choice" and row
            else self._raw_board(position).get("turn", {})
        )
        return {
            "key": position.key,
            "kind": position.kind,
            "turn": turn.get("turn_number"),
            "phase": turn.get("phase"),
            "step": turn.get("step"),
            "actor": row.viewer if row else None,
            "can_reveal": bool(
                position.kind == "choice"
                and row
                and self.watch
                and self.payload["events"][position.event_index].get(
                    "analysis_observation"
                )
            ),
            "automatic": bool(self.payload["events"][position.event_index].get("auto"))
            if position.kind == "consequences"
            else False,
        }

    def _raw_board(self, position: Position) -> dict[str, Any]:
        index = position.event_index + (position.kind == "consequences")
        return (
            self.payload["events"][index]["observation"]
            if index < len(self.payload["events"])
            else self.payload["final_observation"]
        )

    def _history_labels(
        self, event: PresentationEvent, viewer: int
    ) -> dict[str, dict[str, str]]:
        # Names come from the event's own retained transition. A later board,
        # including spectator analysis, must never fill historical private facts.
        labels: dict[str, dict[str, str]] = {"objects": {}, "players": {}, "stacks": {}}
        index = self.event_by_revision.get(event.from_revision)
        if index is None:
            return labels
        row = self.decisions.get(index)
        before = (
            row.frame.projection.model_dump(mode="json")
            if row and row.viewer == viewer
            else self.payload["events"][index]["observation"]
        )
        after = self._raw_board(Position("", "consequences", index))
        names: dict[str, str] = {}
        stacks: dict[str, str] = {}
        for observation, private in (
            (before, bool(row and row.viewer == viewer)),
            (after, False),
        ):
            for side in ("agent", "opponent"):
                player = observation.get(side, {})
                seat = player.get("player_index")
                for key in (player.get("id"), seat):
                    if key is not None:
                        labels["players"][str(key)] = (
                            f"Seat {seat + 1}" if seat is not None else "Player"
                        )
                zones = ["battlefield", "graveyard", "exile", "stack"]
                if private and seat == viewer:
                    zones.extend(["hand", "inspected"])
                for zone in zones:
                    for card in player.get(zone, []):
                        if card.get("name"):
                            names[f"{card['id']}:0"] = card["name"]
                            if zone == "stack":
                                stacks[str(card["id"])] = card["name"]

        def retain(value: Any) -> None:
            if isinstance(value, dict):
                if "entity" in value and "incarnation" in value:
                    key = f"{value['entity']}:{value['incarnation']}"
                    if key in names:
                        labels["objects"][key] = names[key]
                if value.get("kind") == "stack" and str(value.get("id")) in stacks:
                    labels["stacks"][str(value["id"])] = stacks[str(value["id"])]
                for child in value.values():
                    retain(child)
            elif isinstance(value, list):
                for child in value:
                    retain(child)

        retain(event.kind.model_dump(mode="json"))
        return labels

    def study_source(self, key: str) -> dict[str, Any]:
        position = self.by_key[key]
        if not self.replay or position.kind != "choice" or not position.decision:
            raise ValueError("study_source_unavailable")
        viewer = position.decision.viewer
        if viewer not in self.study_sources:
            projection = project_replay(self.replay, viewer)
            self.study_sources[viewer] = {
                "replay_id": projection.replay_id,
                "match_id": projection.match_id,
                "viewer": viewer,
                "content_hash": projection.content_hash,
                "asset_manifest_hash": projection.asset_manifest_hash,
                "sha256": canonical_projection_sha256(projection),
            }
        return dict(self.study_sources[viewer])

    def read(
        self,
        key: str,
        *,
        reveal: bool = False,
        before: int | None = None,
        history_limit: int = 100,
    ) -> dict[str, Any]:
        position = self.by_key[key]
        row = position.decision if position.kind == "choice" else None
        viewer = row.viewer if row else self.human_seat
        cursor = self._cursor(position, viewer)
        end = min(cursor, before) if before is not None else cursor
        start = max(0, end - history_limit)
        track = self.tracks.get(viewer, [])
        events = track[start:end]
        decision = (
            {**row.model_dump(mode="json"), "address": position.key} if row else None
        )
        observation = (
            row.frame.projection.model_dump(mode="json")
            if row
            else self._board(position)
        )
        event = self.payload["events"][position.event_index] if row else {}
        analysis = event.get("analysis_observation") if self.watch else None
        if reveal and analysis is None:
            raise ValueError("spectator_view_unavailable")
        transition_start = cursor
        if position.kind == "consequences":
            revision = self.revisions.get(position.event_index)
            if revision is not None:
                transition_start = bisect_right(
                    self.track_revisions.get(viewer, []), revision
                )
        description: str | None = None
        note_key = position.key
        if position.kind == "consequences":
            source_event = self.payload["events"][position.event_index]
            description = (
                source_event.get("action_description")
                if source_event.get("actor") == "hero"
                else "Opponent decision"
            )
            if source_event.get("auto"):
                description = f"Auto: {description}"
            note_key = f"replay-{position.event_index}"
        return {
            **self.metadata(position),
            "description": description,
            "note_key": note_key,
            "observation": deepcopy(analysis) if reveal else observation,
            "decision": decision,
            "outputs": deepcopy(event.get("bot_decision")),
            "viewer": viewer,
            "revealed": reveal,
            "can_reveal": analysis is not None,
            "history": [event.model_dump(mode="json") for event in events],
            "history_labels": [self._history_labels(event, viewer) for event in events],
            "history_before": start if start else None,
            "cursor": cursor,
            "transition_start": transition_start,
            "consequences_recorded": bool(self.replay) or bool(track),
        }


class ReadCache:
    """At most two retained recordings, expiring after sixty seconds."""

    def __init__(self) -> None:
        self.entries: OrderedDict[tuple[Path, str], tuple[object, float, Recording]] = (
            OrderedDict()
        )
        self.lock = RLock()

    def load(self, directory: Path, trace_id: str, stamp: object) -> Recording:
        key = (directory, trace_id)
        with self.lock:
            now = monotonic()
            for old_key, (_, expiry, _) in list(self.entries.items()):
                if expiry <= now:
                    del self.entries[old_key]
            cached = self.entries.get(key)
            if cached and cached[0] == stamp:
                self.entries.move_to_end(key)
                return cached[2]
            recording = Recording(trace_id, load_trace(trace_id, directory))
            self.entries[key] = (stamp, now + 60, recording)
            while len(self.entries) > 2:
                self.entries.popitem(last=False)
            return recording


read_cache = ReadCache()
