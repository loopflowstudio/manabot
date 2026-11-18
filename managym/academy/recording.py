"""Bounded read projections over retained Academy recordings.

Recording owns positions, viewer event prefixes and legacy transition joins. It
never executes a game. ReadCache validates once per persisted revision; access
is checked separately on every request, including cache hits.
"""

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
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
from managym.academy.trace import load_trace, normalize_observation_to_hero
from managym.experience import PresentationEvent
from managym.experience_projection import redact_observation

# Native decision-context stages that continue announcing a spell or ability
# (`managym/src/decision_context.rs`): the move is named once it is on the stack.
ANNOUNCING_STAGES = frozenset(
    {"payment", "additional_cost", "targets", "mode", "announcement"}
)
# Stages of a choice made while a spell or ability resolves (the suspended
# decisions of `decision_context.rs`). Mana payment, Learn and casting during
# resolution are named by their own lines.
RESOLVING_STAGES = frozenset(
    {
        "resolution_choice",
        "private_inspection",
        "scry",
        "surveil",
        "order_library_bottom",
        "order_library_top",
        "order_replacement",
        "resolution_payment",
        "private_search",
        "name_card",
        "resolution_mode",
        "discard",
        "select_cards",
        "revealed_hand",
        "distribute_counters",
        "retarget",
        "choose_to_exile",
        "keep_creatures",
        "linked_exile",
        "split_revealed",
    }
)


@dataclass(frozen=True)
class OwnStep:
    """One committed choice of the person a narration is for: their own, which
    they may be told about. Another seat's private choice is never an OwnStep."""

    stage: str | None
    chosen: str
    action_type: str
    offered: tuple[str, ...]
    chosen_index: int
    # What the choice is for: the resolving spell or ability, by its label.
    source: str | None = None


def own_steps(events: Sequence[Mapping[str, Any]], index: int) -> list[OwnStep]:
    """The person's own steps of the move that `events[index]` (theirs) ends.

    Trigger ordering takes one step per trigger placed, and a resolving spell's
    choices one per choice, so those moves are the run of the person's
    consecutive steps of that kind (for one source); any other move ends in one
    step.
    `events` are trace events (the `TraceEvent` fields, as a mapping).
    """

    def step(event: Mapping[str, Any]) -> OwnStep:
        actions = event["actions"]
        context = event["observation"].get("decision_context") or {}
        return OwnStep(
            stage=context.get("stage"),
            chosen=event["action_description"],
            action_type=str(actions[event["action"]].get("type", "")),
            offered=tuple(str(action.get("description", "")) for action in actions),
            chosen_index=int(event["action"]),
            source=(context.get("source") or {}).get("label"),
        )

    def same_move(earlier: OwnStep, last: OwnStep) -> bool:
        if last.stage == "order_triggers":
            return earlier.stage == last.stage
        return (
            last.stage in RESOLVING_STAGES
            and earlier.stage in RESOLVING_STAGES
            and earlier.source == last.source
        )

    # A legacy trace event retains neither the offered actions nor the choice.
    if "actions" not in events[index] or "action" not in events[index]:
        return []
    last = step(events[index])
    run = [last]
    for event in reversed(events[:index]):
        if "actions" not in event or "action" not in event:
            break
        earlier = step(event)
        if event["actor"] != events[index]["actor"] or not same_move(earlier, last):
            break
        run.insert(0, earlier)
    return run


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
                i for i, event in enumerate(payload["events"]) if not event.get("auto")
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
            elif not self.replay:
                self.positions.append(Position(f"legacy-choice:{i}", "choice", i))
            self.positions.append(Position(f"after:{i}", "consequences", i, row))
        self.positions.append(Position("latest", "state", len(payload["events"])))
        self.by_key = {position.key: position for position in self.positions}
        self.decisions = decisions
        # Revision joins come from retained decisions/events, never ordinals.
        self.revisions = {i: row.revision for i, row in decisions.items()}
        for i, event in enumerate(payload["events"]):
            if event.get("presentation"):
                self.revisions[i] = event["presentation"][0]["from_revision"]

        self.event_by_revision = {revision: i for i, revision in self.revisions.items()}
        self._narrations: dict[int, str | None] = {}
        self._boards: dict[int, dict[str, Any]] = {}
        self._events_by_revision: dict[int, list[PresentationEvent]] | None = None

    def _board(self, position: Position) -> dict[str, Any]:
        """The recorded person's redacted board at `position`. Callers must not mutate it."""
        index = position.event_index + (position.kind == "consequences")
        if index not in self._boards:
            self._boards[index] = person_board(
                self._raw_board(position), self.human_seat
            )
        return self._boards[index]

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
            # A summary of what `read` returns at this position, so a score can
            # name and place every position without reading each one.
            "revision": row.revision
            if position.kind == "choice" and row
            else self.revisions.get(position.event_index)
            if position.kind == "consequences"
            else None,
            "label": row.offer.label
            if position.kind == "choice" and row
            else self._description(position),
            "value": self._value(position),
        }

    def _description(self, position: Position) -> str | None:
        """What a consequences position narrates.

        The recorded person's own choice is its action. Another seat's choice is
        narrated from what the recorded person saw (`_narration`), never from
        that seat's private action; a step of it nobody could see apart is None.
        """
        if position.kind != "consequences":
            return None
        source = self.payload["events"][position.event_index]
        if source.get("auto"):
            return f"Auto: {source.get('action_description')}"
        return self._narration(position.event_index)

    # ———————————————— Narrating another seat's move

    def _narration(self, index: int) -> str | None:
        """The move at event `index`, as the recorded person saw it (`narrate_move`)."""
        if index not in self._narrations:
            revision = self.revisions.get(index)
            events = self._transition_events(revision) if revision is not None else []
            self._narrations[index] = narrate_move(
                self._board(Position("", "choice", index)),
                self._board(Position("", "consequences", index)),
                [event.kind.model_dump(mode="json") for event in events],
                own_steps(self.payload["events"], index)
                if self.payload["events"][index].get("actor") == "hero"
                else (),
            )
        return self._narrations[index]

    def _transition_events(self, revision: int) -> list[PresentationEvent]:
        if self._events_by_revision is None:
            self._events_by_revision = {}
            for event in self.tracks.get(self.human_seat, []):
                self._events_by_revision.setdefault(event.from_revision, []).append(
                    event
                )
        return self._events_by_revision.get(revision, [])

    def _value(self, position: Position) -> float | None:
        """The chooser's recorded value at a choice, as `read` returns it in `outputs`."""
        if position.kind != "choice" or not position.decision:
            return None
        recorded = self.payload["events"][position.event_index].get("bot_decision")
        value = recorded.get("value") if isinstance(recorded, dict) else None
        return float(value) if isinstance(value, int | float) else None

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
        # Another seat's recorded board is read through the redaction boundary.
        before = (
            row.frame.projection.model_dump(mode="json")
            if row and row.viewer == viewer
            else self._board(Position("", "choice", index))
        )
        after = self._board(Position("", "consequences", index))
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
                            # A board names an object by entity; every
                            # incarnation the event cites is that same object.
                            names[str(card["id"])] = card["name"]
                            if zone == "stack":
                                stacks[str(card["id"])] = card["name"]

        def retain(value: Any) -> None:
            if isinstance(value, dict):
                if "entity" in value and "incarnation" in value:
                    key = f"{value['entity']}:{value['incarnation']}"
                    if str(value["entity"]) in names:
                        labels["objects"][key] = names[str(value["entity"])]
                if value.get("kind") == "stack" and str(value.get("id")) in stacks:
                    labels["stacks"][str(value["id"])] = stacks[str(value["id"])]
                for child in value.values():
                    retain(child)
            elif isinstance(value, list):
                for child in value:
                    retain(child)

        kind = event.kind.model_dump(mode="json")
        retain(kind)
        if kind["kind"] == "cast":
            key = f"{kind['object']['entity']}:{kind['object']['incarnation']}"
            if key not in labels["objects"]:
                # A spell from a hidden hand that resolved within its transition
                # has no object on either board; the public card it became
                # (on the stack, a permanent, in a graveyard) names it.
                arrived = _new_cards(before, after, _side_of(after, kind["controller"]))
                if arrived["spells"]:
                    labels["objects"][key] = arrived["spells"][0]
                    labels["stacks"].setdefault(
                        str(kind["stack"]), arrived["spells"][0]
                    )
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
            else deepcopy(self._board(position))
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
        note_key = (
            f"replay-{position.event_index}"
            if position.kind == "consequences"
            else position.key
        )
        return {
            **self.metadata(position),
            "description": self._description(position),
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


# ———————————————— Narrating a move


def person_board(observation: dict[str, Any], human_seat: int) -> dict[str, Any]:
    """A retained board as the person saw it: hero-oriented and redacted (the
    established viewer boundary, `prepare_trace_payload`)."""
    board = deepcopy(observation)
    normalize_observation_to_hero(board, human_seat)
    redact_observation(board)
    return board


def narrate_move(
    before: dict[str, Any],
    after: dict[str, Any],
    events: list[dict[str, Any]],
    own: Sequence[OwnStep] = (),
) -> str | None:
    """A move, as the person saw it: Review's move list and live play's log
    read this for both players.

    `before` and `after` are the person's boards on either side of the move
    (`person_board`); `before` carries the mover's decision context. `events`
    are the kinds of the person's presentation events for this transition.
    `own` is the person's own steps of this move (`own_steps`) when the move is
    theirs, and empty for another seat, whose hidden choices are never named.
    What was visible becomes the move ("Cast Yip Yip!, targeting Earth Kingdom
    Jailer", "Play Forest", "Attack with Ally, Otter-Penguin"); a priority pass
    is "Pass priority". The person's own moves add what only they saw:
    "Activate Fire Nation Cadets", "Learn: take Accumulate Wisdom", "Ordered
    triggers: Tiger-Seal, then Ally". A step that completes later (announcing a
    spell before it reaches the stack, one attacker of several, one trigger of
    several placed) is None: its move is named where it completes.
    """
    context = before.get("decision_context") or {}
    actor = context.get("actor")
    # Names of what either board shows, by object; a spell on the stack is
    # named by its card.
    names = {
        card["id"]: card["name"]
        for board in (before, after)
        for side_name in ("agent", "opponent")
        for zone in ("battlefield", "graveyard", "exile", "stack", "hand")
        for card in board.get(side_name, {}).get(zone, [])
        if isinstance(card, dict) and card.get("name") and "id" in card
    }
    side = _side_of(after, actor)
    arrived = _new_cards(before, after, side)

    def name(subject: dict[str, Any]) -> str | None:
        if subject.get("kind") == "player":
            return f"Seat {int(subject['id']) + 1}"
        if subject.get("kind") == "stack":
            return names.get(int(subject["id"]))
        identity = subject.get("id", subject)
        return names.get(int(identity["entity"]))

    phrases: list[str] = []
    spells: list[str] = []
    for kind in events:
        if kind["kind"] == "cast" and kind["controller"] == actor:
            spell = (
                name({"kind": "stack", "id": kind["stack"]})
                or name({"kind": "object", "id": kind["object"]})
                or next(
                    (card for card in arrived["spells"] if card not in spells), None
                )
            )
            targets = [
                target
                for other in events
                if other["kind"] == "targeted"
                and other["source"]
                in (
                    {"kind": "stack", "id": kind["stack"]},
                    {"kind": "object", "id": kind["object"]},
                )
                and (target := name(other["target"]))
            ]
            spells.append(spell or "a spell")
            phrases.append(
                f"Cast {spell or 'a spell'}"
                + (f", targeting {_listed(targets)}" if targets else "")
            )
        elif kind["kind"] == "attack_group":
            attackers = [n for a in kind["attackers"] if (n := name({"id": a}))]
            phrases.append(f"Attack with {_listed(attackers) or 'creatures'}")
        elif kind["kind"] == "blocked":
            blockers = [n for b in kind["blockers"] if (n := name({"id": b}))]
            attacker = name({"id": kind["attacker"]}) or "an attacker"
            phrases.append(f"Block {attacker} with {_listed(blockers) or 'creatures'}")
    if not spells:
        phrases[:0] = [f"Play {land}" for land in arrived["lands"]]
        if arrived["discarded"]:
            phrases.append(f"Discard {_listed(arrived['discarded'])}")
    if phrases:
        return "; ".join(phrases)
    following = after.get("decision_context") or {}
    continues = following.get("actor") == actor and (
        following.get("stage") in ANNOUNCING_STAGES
        or (own and following.get("stage") == own[-1].stage == "order_triggers")
        or (
            own
            and own[-1].stage in RESOLVING_STAGES
            and following.get("stage") in RESOLVING_STAGES
            and (following.get("source") or {}).get("label") == own[-1].source
        )
    )
    if continues:
        # The same seat goes straight on with what it began (targets, costs,
        # modes, the next trigger to place).
        return None
    if own:
        return _own_move(context, own)
    if context.get("stage") != "priority":
        return None
    return "Pass priority"


def _own_move(context: dict[str, Any], own: Sequence[OwnStep]) -> str | None:
    """The person's own move that left nothing on the board to name it by."""
    last = own[-1]
    stage = context.get("stage")
    if stage == "order_triggers":
        placed = [step.offered[step.chosen_index] for step in own] + [
            offer for i, offer in enumerate(last.offered) if i != last.chosen_index
        ]
        return "Ordered triggers: " + ", then ".join(_trigger_source(p) for p in placed)
    if stage == "learn":
        if last.chosen == "Decline Learn":
            return "Declined to learn"
        return f"Learn: {last.chosen[:1].lower()}{last.chosen[1:]}"
    if last.stage in RESOLVING_STAGES:
        return _resolution(own)
    source = (context.get("source") or {}).get("label")
    if context.get("announcing") == "activated_ability" and source:
        targets = [t["label"] for t in context.get("targets") or [] if t.get("label")]
        return f"Activate {source}" + (
            f", targeting {_listed(targets)}" if targets else ""
        )
    if last.chosen.startswith("Activate ability: "):
        return "Activate " + last.chosen.removeprefix("Activate ability: ")
    if stage in ("declare_attackers", "declare_blockers"):
        # Attacks and blocks are named by their events; holding back is no move.
        return None
    if stage == "priority" and last.action_type == "PRIORITY_PASS_PRIORITY":
        return "Pass priority"
    # A choice while something resolves ("Put Island into your hand") is
    # named as the person made it.
    return last.chosen


def _resolution(own: Sequence[OwnStep]) -> str:
    """ "Accumulate Wisdom: kept Island, put Pop Quiz, then Firebending Lesson
    on the bottom": the person's choices for one resolving source, in order.

    A library ordering takes one step per card placed and places the last one
    itself, so a run of ordering steps is one phrase naming every card.
    """
    phrases: list[str] = []
    index = 0
    while index < len(own):
        step = own[index]
        if step.stage in ("order_library_bottom", "order_library_top"):
            run = [step]
            while index + 1 < len(own) and own[index + 1].stage == step.stage:
                index += 1
                run.append(own[index])
            last = run[-1]
            placed = [s.offered[s.chosen_index] for s in run] + [
                offer for i, offer in enumerate(last.offered) if i != last.chosen_index
            ]
            cards = [_ORDERED.sub(r"\1", offer) for offer in placed]
            end = "on the bottom" if step.stage == "order_library_bottom" else "on top"
            phrases.append(f"put {_in_order(cards)} {end}")
        else:
            kept = _KEPT.fullmatch(step.chosen)
            phrases.append(
                f"kept {kept.group(1)}"
                if kept
                else f"{step.chosen[:1].lower()}{step.chosen[1:]}"
            )
        index += 1
    text = ", ".join(phrases)
    return (
        f"{own[-1].source}: {text}" if own[-1].source else text[:1].upper() + text[1:]
    )


# Offer labels of library choices (`managym/src/flow/action.rs`).
_ORDERED = re.compile(r"^Put (.+) next on the (?:bottom|top)$")
_KEPT = re.compile(r"Put (.+) into your hand")


def _in_order(names: list[str]) -> str:
    """ "Island ×2, then Pop Quiz": names in order, a repeated name once with
    its count when the copies are next to each other."""
    runs: list[tuple[str, int]] = []
    for name in names:
        if runs and runs[-1][0] == name:
            runs[-1] = (name, runs[-1][1] + 1)
        else:
            runs.append((name, 1))
    return ", then ".join(
        f"{name} ×{count}" if count > 1 else name for name, count in runs
    )


def _trigger_source(offer: str) -> str:
    """ "Put on stack next — Tiger-Seal — At the beginning…" names Tiger-Seal."""
    parts = offer.split(" — ")
    return parts[1] if len(parts) > 2 else offer


def _side_of(observation: dict[str, Any], seat: object) -> str | None:
    for side in ("agent", "opponent"):
        if observation.get(side, {}).get("player_index") == seat:
            return side
    return None


def _new_cards(
    before: dict[str, Any], after: dict[str, Any], side: str | None
) -> dict[str, list[str]]:
    """Public cards that reached `side`'s zones in a transition, by what they are.

    Lands new to the battlefield, other new public cards (a spell reaching the
    stack or becoming a permanent, by name), and cards that left the hand for
    the graveyard while nothing was cast: a discard.
    """
    found: dict[str, list[str]] = {"lands": [], "spells": [], "discarded": []}
    if side is None:
        return found
    old, new = before.get(side, {}), after.get(side, {})
    seen = {
        card.get("id")
        for zone in ("battlefield", "graveyard", "exile", "stack")
        for card in old.get(zone, [])
    }
    for zone in ("stack", "battlefield", "graveyard", "exile"):
        for card in new.get(zone, []):
            if card.get("id") in seen or not card.get("name"):
                continue
            seen.add(card.get("id"))
            if zone == "battlefield" and (card.get("types") or {}).get("is_land"):
                found["lands"].append(card["name"])
            else:
                found["spells"].append(card["name"])
    hand_before = len(old.get("hand", [])) + int(old.get("hand_hidden_count") or 0)
    hand_after = len(new.get("hand", [])) + int(new.get("hand_hidden_count") or 0)
    new_graveyard = [
        card["name"]
        for card in new.get("graveyard", [])
        if card.get("id") not in {c.get("id") for c in old.get("graveyard", [])}
        and card.get("name")
    ]
    if hand_after < hand_before:
        found["discarded"] = new_graveyard[: hand_before - hand_after]
    return found


def _listed(names: list[str]) -> str:
    """ "Ally", "Ally and Forest", "Badgermole Cub ×2, Forest and Ally"."""
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    parts = [
        f"{name} ×{count}" if count > 1 else name for name, count in counts.items()
    ]
    return (
        parts[0]
        if len(parts) == 1
        else ", ".join(parts[:-1]) + " and " + parts[-1]
        if parts
        else ""
    )
