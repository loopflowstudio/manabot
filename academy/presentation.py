"""Authority-side projection of committed engine facts into presentation events.

The projector follows committed spells and combat-to-turn sequences. Meaning
comes from server-authored interaction offers confirmed by
``recent_events``; observation snapshots are viewer-safe label catalogs only
and are never diffed to invent game facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from managym.enums import EventTypeEnum
from managym.experience import InteractionOffer, PresentationEvent

import managym


@dataclass(frozen=True)
class _PresentationFact:
    importance: str
    suggested_ms: int
    sound: str | None
    kind: dict[str, Any]


@dataclass(frozen=True)
class _PendingSpell:
    object_id: dict[str, int]
    controller: int
    targets: tuple[dict[str, Any], ...] = ()


def _event_type(event: managym.EventData) -> str:
    try:
        return EventTypeEnum(int(event.event_type)).name
    except (TypeError, ValueError):
        return "UNKNOWN"


def _event_object_id(event: managym.EventData, *, source: bool) -> dict[str, int]:
    entity = event.source_id if source else event.target_id
    incarnation = event.source_incarnation if source else event.target_incarnation
    return {"entity": int(entity), "incarnation": int(incarnation)}


class PresentationProjector:
    """Stateful match-local projection of committed authored sequences."""

    def __init__(self) -> None:
        self._next_seq = 0
        self._facts: list[_PresentationFact] = []
        self._spell_stacks: dict[int, int] = {}
        self._pending_spell: _PendingSpell | None = None

    def reset(self) -> None:
        self._next_seq = 0
        self._facts = []
        self._spell_stacks = {}
        self._pending_spell = None

    @property
    def next_seq(self) -> int:
        """Next authority-owned event address for recovery cursors."""
        return self._next_seq

    def note_action(self, offer: InteractionOffer) -> None:
        """Stage native subjects until matching cast/resolution events commit.

        Casting and target selection can resolve in the same engine step, so
        the post-step stack cannot supply their identities. Historical offers
        without typed subjects provide no new presentation facts.
        """
        if offer.action_type == "PRIORITY_CAST_SPELL":
            source = offer.source
            self._pending_spell = (
                _PendingSpell(object_id=source.id.model_dump(), controller=offer.actor)
                if source is not None and source.kind == "object"
                else None
            )
        elif offer.action_type == "CHOOSE_TARGET" and self._pending_spell is not None:
            targets = tuple(
                subject.object.subject.model_dump(mode="json")
                for subject in offer.subjects
                if subject.role == "target"
            )
            self._pending_spell = _PendingSpell(
                object_id=self._pending_spell.object_id,
                controller=self._pending_spell.controller,
                targets=(*self._pending_spell.targets, *targets),
            )

    def observe(
        self,
        observation: managym.Observation,
        *,
        event_subjects: dict[str, dict[str, Any]],
    ) -> None:
        """Consume the committed semantic events produced by one engine step."""

        events = list(observation.recent_events)
        cast_events = [event for event in events if _event_type(event) == "SPELL_CAST"]
        for event in cast_events:
            self._observe_cast(event)

        for event in events:
            if _event_type(event) == "SPELL_COUNTERED":
                self._spell_stacks.pop(int(event.source_id), None)
        resolved_events = [
            event for event in events if _event_type(event) == "SPELL_RESOLVED"
        ]
        for event in resolved_events:
            self._observe_resolution(event, events, event_subjects)
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
        event: managym.EventData,
    ) -> None:
        pending = self._pending_spell
        self._pending_spell = None
        if pending is None:
            return

        # The legacy projection cannot name a separate StackRenderId, so the
        # adapter intentionally reuses the visible spell card's render ID.
        stack_id = int(pending.object_id["entity"])
        self._spell_stacks[int(event.source_id)] = stack_id
        self._facts.append(
            _PresentationFact(
                importance="emphasized",
                suggested_ms=650,
                sound="spell.cast",
                kind={
                    "kind": "cast",
                    "object": pending.object_id,
                    "controller": pending.controller,
                    "stack": stack_id,
                },
            )
        )
        source = {"kind": "stack", "id": stack_id}
        for target in pending.targets:
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
        resolved: managym.EventData,
        events: list[managym.EventData],
        event_subjects: dict[str, dict[str, Any]],
    ) -> None:
        source_id = int(resolved.source_id)
        stack_id = self._spell_stacks.pop(source_id, None)
        if stack_id is None:
            return

        self._facts.append(
            _PresentationFact(
                importance="emphasized",
                suggested_ms=450,
                sound="spell.resolve",
                kind={"kind": "resolved", "stack": stack_id},
            )
        )

        damage_events = [
            event
            for event in events
            if _event_type(event) == "DAMAGE_DEALT"
            and int(event.source_id) == source_id
        ]
        for damage in damage_events:
            namespace = {2: "permanent", 3: "player"}.get(int(damage.target_kind))
            target = event_subjects.get(f"{namespace}:{int(damage.target_id)}")
            if target is None:
                continue
            self._facts.append(
                _PresentationFact(
                    importance="critical",
                    suggested_ms=700,
                    sound="damage.fire",
                    kind={
                        "kind": "damage",
                        "source": {"kind": "stack", "id": stack_id},
                        "target": target,
                        "amount": int(damage.amount),
                    },
                )
            )

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
