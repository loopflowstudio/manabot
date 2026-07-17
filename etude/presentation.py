"""Authority-side projection of committed engine facts into presentation events.

The projector recognizes the authored Lightning Bolt and combat-to-turn
sequences. Meaning comes from server-authored interaction offers confirmed by
``recent_events``; observation snapshots are viewer-safe label catalogs only
and are never diffed to invent game facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import managym

from .enums import EventTypeEnum
from .experience_protocol import PresentationEvent


@dataclass(frozen=True)
class _PresentationFact:
    importance: str
    suggested_ms: int
    sound: str | None
    kind: dict[str, Any]


@dataclass(frozen=True)
class _ActiveBolt:
    source_event_id: int
    object_id: dict[str, int]
    stack_id: int
    controller: int
    targets: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _PendingBolt:
    object_id: dict[str, int]
    controller: int
    targets: tuple[dict[str, Any], ...] = ()


def _event_type(event: managym.EventData) -> str:
    try:
        return EventTypeEnum(int(event.event_type)).name
    except (TypeError, ValueError):
        return "UNKNOWN"


def _object_id(value: int) -> dict[str, int]:
    # The transitional ExperienceFrame still exposes legacy render IDs and no
    # incarnation field. Protocol v1 therefore certifies incarnation zero at
    # this adapter boundary; the engine remains the rules authority.
    return {"entity": int(value), "incarnation": 0}


def _event_object_id(event: managym.EventData, *, source: bool) -> dict[str, int]:
    entity = event.source_id if source else event.target_id
    incarnation = event.source_incarnation if source else event.target_incarnation
    return {"entity": int(entity), "incarnation": int(incarnation)}


class PresentationProjector:
    """Stateful match-local projection of committed authored sequences."""

    def __init__(self) -> None:
        self._next_seq = 0
        self._facts: list[_PresentationFact] = []
        self._active_bolts: dict[int, _ActiveBolt] = {}
        self._pending_bolt: _PendingBolt | None = None

    def reset(self) -> None:
        self._next_seq = 0
        self._facts = []
        self._active_bolts = {}
        self._pending_bolt = None

    @property
    def next_seq(self) -> int:
        """Next authority-owned event address for recovery cursors."""
        return self._next_seq

    def note_action(
        self,
        observation: managym.Observation,
        action: dict[str, Any],
        *,
        actor_index: int,
    ) -> bool:
        """Stage exact identities selected through an authority offer.

        Casting and target selection can complete resolution in the same
        engine step, so the post-step stack is intentionally not consulted.
        Staged choices never become presentation facts unless the engine then
        commits the matching spell/damage/resolution domain events.

        Returns whether the transitional Lightning Bolt presentation branch
        dispatched on a card name. Authority receipts count that branch as an
        exercised fallback instead of inferring its absence.
        """

        focus = action.get("focus")
        if not isinstance(focus, list) or not focus:
            return False
        selected = int(focus[0])
        if action.get("type") == "PRIORITY_CAST_SPELL":
            cards = [*observation.agent_cards, *observation.opponent_cards]
            card = next(
                (candidate for candidate in cards if int(candidate.id) == selected),
                None,
            )
            used_card_name_dispatch = card is not None and card.name == "Lightning Bolt"
            self._pending_bolt = (
                _PendingBolt(
                    object_id=_object_id(selected),
                    controller=actor_index,
                )
                if used_card_name_dispatch
                else None
            )
            return used_card_name_dispatch

        if action.get("type") != "CHOOSE_TARGET" or self._pending_bolt is None:
            return False
        permanent_ids = {
            int(permanent.id)
            for permanent in [
                *observation.agent_permanents,
                *observation.opponent_permanents,
            ]
        }
        subject = (
            {"kind": "object", "id": _object_id(selected)}
            if selected in permanent_ids
            else {"kind": "player", "id": selected}
        )
        self._pending_bolt = _PendingBolt(
            object_id=self._pending_bolt.object_id,
            controller=self._pending_bolt.controller,
            targets=(*self._pending_bolt.targets, subject),
        )
        return False

    def observe(self, observation: managym.Observation) -> None:
        """Consume the committed semantic events produced by one engine step."""

        events = list(observation.recent_events)
        cast_events = [event for event in events if _event_type(event) == "SPELL_CAST"]
        for event in cast_events:
            self._observe_cast(observation, event)

        resolved_events = [
            event for event in events if _event_type(event) == "SPELL_RESOLVED"
        ]
        for event in resolved_events:
            self._observe_resolution(observation, event, events)
        self._observe_combat_death_and_turn(events)

    def drain(
        self,
        *,
        from_revision: int,
        to_revision: int,
        caused_by: str | None,
    ) -> list[dict[str, Any]]:
        """Bind pending facts to one authoritative revision transition."""

        payloads: list[dict[str, Any]] = []
        for fact in self._facts:
            seq = self._next_seq
            self._next_seq += 1
            event = PresentationEvent.model_validate(
                {
                    "seq": seq,
                    "from_revision": from_revision,
                    "to_revision": to_revision,
                    "caused_by": caused_by,
                    "group": seq,
                    "importance": fact.importance,
                    "suggested_ms": fact.suggested_ms,
                    "sound": fact.sound,
                    "kind": fact.kind,
                }
            )
            payloads.append(event.model_dump(mode="json"))
        self._facts = []
        return payloads

    def _observe_cast(
        self,
        observation: managym.Observation,
        event: managym.EventData,
    ) -> None:
        del observation
        pending = self._pending_bolt
        self._pending_bolt = None
        if pending is None:
            return

        # The legacy projection cannot name a separate StackRenderId, so the
        # adapter intentionally reuses the visible spell card's render ID.
        stack_id = int(pending.object_id["entity"])
        bolt = _ActiveBolt(
            source_event_id=int(event.source_id),
            object_id=pending.object_id,
            stack_id=stack_id,
            controller=pending.controller,
            targets=pending.targets,
        )
        self._active_bolts[bolt.source_event_id] = bolt
        self._facts.append(
            _PresentationFact(
                importance="emphasized",
                suggested_ms=650,
                sound="spell.cast",
                kind={
                    "kind": "cast",
                    "object": bolt.object_id,
                    "controller": bolt.controller,
                    "stack": bolt.stack_id,
                },
            )
        )
        source = {"kind": "stack", "id": bolt.stack_id}
        for target in bolt.targets:
            self._facts.append(
                _PresentationFact(
                    importance="normal",
                    suggested_ms=500,
                    sound=None,
                    kind={"kind": "targeted", "source": source, "target": target},
                )
            )

    def _observe_resolution(
        self,
        observation: managym.Observation,
        resolved: managym.EventData,
        events: list[managym.EventData],
    ) -> None:
        del observation
        bolt = self._active_bolts.get(int(resolved.source_id))
        if bolt is None:
            return

        self._facts.append(
            _PresentationFact(
                importance="emphasized",
                suggested_ms=450,
                sound="spell.resolve",
                kind={"kind": "resolved", "stack": bolt.stack_id},
            )
        )

        damage_events = [
            event
            for event in events
            if _event_type(event) == "DAMAGE_DEALT"
            and int(event.source_id) == bolt.source_event_id
        ]
        for index, damage in enumerate(damage_events):
            target = bolt.targets[index] if index < len(bolt.targets) else None
            if target is None:
                continue
            self._facts.append(
                _PresentationFact(
                    importance="critical",
                    suggested_ms=700,
                    sound="damage.fire",
                    kind={
                        "kind": "damage",
                        "source": {"kind": "stack", "id": bolt.stack_id},
                        "target": target,
                        "amount": int(damage.amount),
                    },
                )
            )

        del self._active_bolts[bolt.source_event_id]

    def _observe_combat_death_and_turn(self, events: list[managym.EventData]) -> None:
        index = 0
        while index < len(events):
            event = events[index]
            event_type = _event_type(event)

            if event_type == "COMBAT_ATTACKERS_DECLARED":
                attackers: list[dict[str, int]] = []
                defender = int(event.target_id)
                while (
                    index < len(events)
                    and _event_type(events[index]) == "COMBAT_ATTACKERS_DECLARED"
                ):
                    attackers.append(_event_object_id(events[index], source=True))
                    index += 1
                self._facts.append(
                    _PresentationFact(
                        importance="emphasized",
                        suggested_ms=500,
                        sound="combat.attack",
                        kind={
                            "kind": "attack_group",
                            "attackers": attackers,
                            "defender": {"kind": "player", "id": defender},
                        },
                    )
                )
                continue

            if event_type == "BLOCKERS_DECLARED":
                assignments: dict[tuple[int, int], list[dict[str, int]]] = {}
                while (
                    index < len(events)
                    and _event_type(events[index]) == "BLOCKERS_DECLARED"
                ):
                    current = events[index]
                    attacker = _event_object_id(current, source=True)
                    key = (attacker["entity"], attacker["incarnation"])
                    assignments.setdefault(key, []).append(
                        _event_object_id(current, source=False)
                    )
                    index += 1
                for (entity, incarnation), blockers in assignments.items():
                    self._facts.append(
                        _PresentationFact(
                            importance="normal",
                            suggested_ms=450,
                            sound="combat.block",
                            kind={
                                "kind": "blocked",
                                "attacker": {
                                    "entity": entity,
                                    "incarnation": incarnation,
                                },
                                "blockers": blockers,
                            },
                        )
                    )
                continue

            if event_type == "COMBAT_DAMAGE_DEALT":
                target = (
                    {
                        "kind": "object",
                        "id": _event_object_id(event, source=False),
                    }
                    if int(event.target_kind) == int(managym.EventEntityKindEnum.OBJECT)
                    else {"kind": "player", "id": int(event.target_id)}
                )
                self._facts.append(
                    _PresentationFact(
                        importance="critical",
                        suggested_ms=500,
                        sound="combat.damage",
                        kind={
                            "kind": "damage",
                            "source": {
                                "kind": "object",
                                "id": _event_object_id(event, source=True),
                            },
                            "target": target,
                            "amount": int(event.amount),
                        },
                    )
                )
                index += 1
                continue

            if event_type == "PERMANENTS_DIED":
                objects: list[dict[str, int]] = []
                batch_size = max(1, int(event.amount))
                while (
                    index < len(events)
                    and _event_type(events[index]) == "PERMANENTS_DIED"
                    and len(objects) < batch_size
                ):
                    objects.append(_event_object_id(events[index], source=True))
                    index += 1
                self._facts.append(
                    _PresentationFact(
                        importance="critical",
                        suggested_ms=650,
                        sound="creature.died",
                        kind={"kind": "died", "objects": objects},
                    )
                )
                continue

            if event_type == "TURN_STARTED":
                self._facts.append(
                    _PresentationFact(
                        importance="emphasized",
                        suggested_ms=350,
                        sound="turn.start",
                        kind={
                            "kind": "turn_started",
                            "player": int(event.source_id),
                            "turn_number": int(event.amount),
                        },
                    )
                )

            index += 1
