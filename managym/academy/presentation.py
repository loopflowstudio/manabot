"""Authority-side projection of committed engine facts into presentation events.

The engine's presentation sink (``managym/src/flow/presentation.rs``) tells
one step's facts in order, each with the fact it follows from and the rule
that did it, through the observation's own viewer-safe event filter. The
projector words them as typed ``PresentationEvent``s and binds each fact's
cause to the ``seq`` of the nearest told ancestor. Targets come from the
committed offers, which name them before the step resolves them; the
post-step ``presentation_subjects`` name battlefield and graveyard objects.
Observation snapshots are label catalogs only and are never diffed to invent
game facts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any

import managym
from managym.enums import EventTypeEnum, ZoneEnum
from managym.experience import InteractionOffer, PresentationEvent


@dataclass(frozen=True)
class _PresentationFact:
    importance: str
    suggested_ms: int
    sound: str | None
    kind: dict[str, Any]
    fact: int
    rule: str | None = None


@dataclass(frozen=True)
class _PendingSpell:
    object_id: dict[str, int]
    controller: int
    targets: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class EventRow:
    """One viewer-safe event row, as ``managym.EventData`` names its fields."""

    event_type: int
    source_kind: int
    source_id: int
    target_kind: int
    target_id: int
    amount: int
    controller_id: int
    from_zone: int
    to_zone: int
    source_incarnation: int
    target_incarnation: int


@dataclass(frozen=True)
class SinkFact:
    """One fact the engine told: its event rows or a note, and why."""

    id: int
    cause: int | None
    rule: str | None
    events: tuple[EventRow, ...]
    note: dict[str, Any] | None


def read_facts(payload: str) -> list[SinkFact]:
    """Decode ``Env.take_presentation_facts_json``."""
    return [
        SinkFact(
            id=int(fact["id"]),
            cause=None if fact["cause"] is None else int(fact["cause"]),
            rule=fact["rule"],
            events=tuple(EventRow(**row) for row in fact["events"]),
            note=fact["note"],
        )
        for fact in json.loads(payload)
    ]


def _event_type(event: EventRow) -> str:
    try:
        return EventTypeEnum(int(event.event_type)).name
    except (TypeError, ValueError):
        return "UNKNOWN"


def _event_object_id(event: EventRow, *, source: bool) -> dict[str, int]:
    entity = event.source_id if source else event.target_id
    incarnation = event.source_incarnation if source else event.target_incarnation
    return {"entity": int(entity), "incarnation": int(incarnation)}


def _render_id(reference: Mapping[str, Any]) -> dict[str, int]:
    return {
        "entity": int(reference["entity"]),
        "incarnation": int(reference["incarnation"]),
    }


class PresentationProjector:
    """Stateful match-local projection of the engine's told facts."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._next_seq = 0
        self._facts: list[_PresentationFact] = []
        self._spell_stacks: dict[int, int] = {}
        self._pending_spell: _PendingSpell | None = None
        self._pending_land = False
        self._resolving_spell: tuple[int, int] | None = None
        # Every fact's cause, told or not, so a cause climbs to a told one.
        self._causes: dict[int, int | None] = {}
        self._told: dict[int, int] = {}
        self._roots: dict[int, int] = {}

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
        self._pending_land = offer.action_type == "PRIORITY_PLAY_LAND"
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
        facts: list[SinkFact],
        *,
        event_subjects: dict[str, dict[str, Any]],
        definition_names: Mapping[int, str],
    ) -> None:
        """Word the facts one engine step told, in the order they happened.

        ``definition_names`` names the game's card definitions by id
        (``definition_names(env)``); a reveal reports only a definition. One
        step's discards and reveals by one player are one event each, at the
        first of them.
        """
        played = self._pending_land
        self._pending_land = False
        discarded: dict[int, tuple[int, list[dict[str, int]]]] = {}
        revealed: dict[int, tuple[int, list[str]]] = {}
        for fact in facts:
            self._causes[fact.id] = fact.cause
            if fact.note is not None:
                self._observe_note(fact, fact.note)
            rows = fact.events
            if not rows:
                continue
            event_type = _event_type(rows[0])
            if event_type == "SPELL_CAST":
                self._observe_cast(rows[0], fact)
            elif event_type == "SPELL_COUNTERED":
                stack = self._spell_stacks.pop(int(rows[0].source_id), None)
                if stack is not None:
                    self._tell(
                        fact, "normal", 450, None, {"kind": "countered", "stack": stack}
                    )
            elif event_type == "DAMAGE_DEALT":
                self._observe_damage(rows[0], fact, event_subjects)
            elif event_type == "CARD_MOVED":
                row = rows[0]
                subject = event_subjects.get(f"object:{int(row.source_id)}")
                if (
                    int(row.source_id) < 0
                    or subject is None
                    or subject.get("kind") != "object"
                ):
                    continue
                moved = (int(row.from_zone), int(row.to_zone))
                if played and moved == (int(ZoneEnum.HAND), int(ZoneEnum.BATTLEFIELD)):
                    played = False
                    self._tell(
                        fact,
                        "normal",
                        400,
                        None,
                        {
                            "kind": "land_played",
                            "object": subject["id"],
                            "player": int(row.controller_id),
                        },
                    )
                elif moved == (int(ZoneEnum.HAND), int(ZoneEnum.GRAVEYARD)):
                    first, objects = discarded.setdefault(
                        int(row.controller_id), (fact.id, [])
                    )
                    objects.append(subject["id"])
            elif event_type == "CARD_REVEALED":
                name = definition_names.get(int(rows[0].source_id))
                if name is not None:
                    _, cards = revealed.setdefault(
                        int(rows[0].controller_id), (fact.id, [])
                    )
                    cards.append(name)
            else:
                self._observe_combat_death_and_turn(rows, fact)
        for player, (first, objects) in discarded.items():
            self._facts.append(
                _PresentationFact(
                    importance="normal",
                    suggested_ms=450,
                    sound=None,
                    kind={"kind": "discarded", "objects": objects, "player": player},
                    fact=first,
                )
            )
        for player, (first, cards) in revealed.items():
            self._facts.append(
                _PresentationFact(
                    importance="normal",
                    suggested_ms=450,
                    sound=None,
                    kind={"kind": "revealed", "cards": cards, "player": player},
                    fact=first,
                )
            )

    def _tell(
        self,
        fact: SinkFact,
        importance: str,
        suggested_ms: int,
        sound: str | None,
        kind: dict[str, Any],
    ) -> None:
        self._facts.append(
            _PresentationFact(
                importance=importance,
                suggested_ms=suggested_ms,
                sound=sound,
                kind=kind,
                fact=fact.id,
                rule=fact.rule,
            )
        )

    def _observe_note(self, fact: SinkFact, note: Mapping[str, Any]) -> None:
        kind = note["kind"]
        if kind == "resolving":
            self._resolving_spell = None
            if note["spell"]:
                if fact.events:
                    slot = int(fact.events[0].source_id)
                    stack = self._spell_stacks.pop(slot, None)
                    if stack is not None:
                        self._resolving_spell = (slot, stack)
                        self._tell(
                            fact,
                            "emphasized",
                            450,
                            "spell.resolve",
                            {"kind": "resolved", "stack": stack},
                        )
            elif note["source"] is not None:
                self._tell(
                    fact,
                    "normal",
                    450,
                    None,
                    {
                        "kind": "ability_resolved",
                        "source": _render_id(note["source"]),
                        "stack": int(note["stack"]),
                    },
                )
        elif kind in {"activated", "triggered"} and note["source"] is not None:
            self._tell(
                fact,
                "emphasized" if kind == "activated" else "normal",
                550,
                None,
                {
                    "kind": kind,
                    "source": _render_id(note["source"]),
                    "controller": int(note["controller"]),
                    "stack": int(note["stack"]),
                },
            )
        elif kind == "counters":
            self._tell(
                fact,
                "normal",
                450,
                None,
                {
                    "kind": "counters",
                    "object": _render_id(note["object"]),
                    "counter": str(note["counter"]),
                    "count": int(note["count"]),
                    "total": int(note["total"]),
                },
            )

    def drain(
        self,
        *,
        from_revision: int,
        to_revision: int,
        caused_by: str | None,
    ) -> list[dict[str, Any]]:
        """Bind pending facts to one authoritative revision transition.

        An event's ``cause`` is the first told event of its fact's nearest
        told ancestor (or of its own fact, for a cast's targets), and its
        ``group`` is the root of that chain.
        """

        payloads: list[dict[str, Any]] = []
        for fact in self._facts:
            seq = self._next_seq
            self._next_seq += 1
            cause = self._told.get(fact.fact)
            if cause is None:
                self._told[fact.fact] = seq
                ancestor = self._causes.get(fact.fact)
                while ancestor is not None and ancestor not in self._told:
                    ancestor = self._causes.get(ancestor)
                cause = None if ancestor is None else self._told[ancestor]
            group = seq if cause is None else self._roots[cause]
            self._roots[seq] = group
            event = PresentationEvent.model_validate(
                {
                    "seq": seq,
                    "from_revision": from_revision,
                    "to_revision": to_revision,
                    "caused_by": caused_by,
                    "group": group,
                    "importance": fact.importance,
                    "suggested_ms": fact.suggested_ms,
                    "sound": fact.sound,
                    "kind": fact.kind,
                    "cause": cause,
                    "rule": fact.rule,
                }
            )
            payloads.append(event.model_dump(mode="json"))
        self._facts = []
        return payloads

    def _observe_cast(self, event: EventRow, fact: SinkFact) -> None:
        pending = self._pending_spell
        self._pending_spell = None
        if pending is None:
            return

        # The legacy projection cannot name a separate StackRenderId, so the
        # adapter intentionally reuses the visible spell card's render ID.
        stack_id = int(pending.object_id["entity"])
        self._spell_stacks[int(event.source_id)] = stack_id
        self._tell(
            fact,
            "emphasized",
            650,
            "spell.cast",
            {
                "kind": "cast",
                "object": pending.object_id,
                "controller": pending.controller,
                "stack": stack_id,
            },
        )
        source = {"kind": "stack", "id": stack_id}
        for target in pending.targets:
            self._tell(
                fact,
                "normal",
                500,
                None,
                {"kind": "targeted", "source": source, "target": target},
            )

    def _observe_damage(
        self,
        damage: EventRow,
        fact: SinkFact,
        event_subjects: dict[str, dict[str, Any]],
    ) -> None:
        """Damage from the resolving spell, or from a named object."""
        namespace = {2: "permanent", 3: "player"}.get(int(damage.target_kind))
        target = event_subjects.get(f"{namespace}:{int(damage.target_id)}")
        if target is None:
            return
        resolving = self._resolving_spell
        if resolving is not None and int(damage.source_id) == resolving[0]:
            source: dict[str, Any] | None = {"kind": "stack", "id": resolving[1]}
        elif int(damage.source_kind) == int(managym.EventEntityKindEnum.OBJECT):
            source = {"kind": "object", "id": _event_object_id(damage, source=True)}
        else:
            source = event_subjects.get(f"object:{int(damage.source_id)}")
        self._tell(
            fact,
            "critical",
            700,
            "damage.fire",
            {
                "kind": "damage",
                "source": source,
                "target": target,
                "amount": int(damage.amount),
            },
        )

    def _observe_combat_death_and_turn(
        self, rows: tuple[EventRow, ...], fact: SinkFact
    ) -> None:
        """One fact's rows: an attack, the blocks, combat damage, a batch of
        deaths or a turn's start."""
        event_type = _event_type(rows[0])
        if event_type == "COMBAT_ATTACKERS_DECLARED":
            self._tell(
                fact,
                "emphasized",
                500,
                "combat.attack",
                {
                    "kind": "attack_group",
                    "attackers": [_event_object_id(row, source=True) for row in rows],
                    "defender": {"kind": "player", "id": int(rows[0].target_id)},
                },
            )
        elif event_type == "BLOCKERS_DECLARED":
            assignments: dict[tuple[int, int], list[dict[str, int]]] = {}
            for row in rows:
                attacker = _event_object_id(row, source=True)
                key = (attacker["entity"], attacker["incarnation"])
                assignments.setdefault(key, []).append(
                    _event_object_id(row, source=False)
                )
            for (entity, incarnation), blockers in assignments.items():
                self._tell(
                    fact,
                    "normal",
                    450,
                    "combat.block",
                    {
                        "kind": "blocked",
                        "attacker": {"entity": entity, "incarnation": incarnation},
                        "blockers": blockers,
                    },
                )
        elif event_type == "COMBAT_DAMAGE_DEALT":
            row = rows[0]
            target = (
                {"kind": "object", "id": _event_object_id(row, source=False)}
                if int(row.target_kind) == int(managym.EventEntityKindEnum.OBJECT)
                else {"kind": "player", "id": int(row.target_id)}
            )
            self._tell(
                fact,
                "critical",
                500,
                "combat.damage",
                {
                    "kind": "damage",
                    "source": {
                        "kind": "object",
                        "id": _event_object_id(row, source=True),
                    },
                    "target": target,
                    "amount": int(row.amount),
                },
            )
        elif event_type == "PERMANENTS_DIED":
            self._tell(
                fact,
                "critical",
                650,
                "creature.died",
                {
                    "kind": "died",
                    "objects": [_event_object_id(row, source=True) for row in rows],
                },
            )
        elif event_type == "TURN_STARTED":
            self._tell(
                fact,
                "emphasized",
                350,
                "turn.start",
                {
                    "kind": "turn_started",
                    "player": int(rows[0].source_id),
                    "turn_number": int(rows[0].amount),
                },
            )
