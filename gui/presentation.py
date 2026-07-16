"""Authority-side projection of committed engine facts into presentation events.

This vertical slice intentionally recognizes one authored sequence: Lightning
Bolt cast at one target, resolving for damage, and (when the committed zone
event proves it) that target dying. Meaning comes from server-authored
interaction offers confirmed by ``recent_events``; observation snapshots are
used only as a viewer-safe identity/name catalog, never diffed to invent game
facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from manabot.env.observation import EventTypeEnum, ZoneEnum
import managym

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


class PresentationProjector:
    """Stateful match-local projector for the authored Bolt sequence."""

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
    ) -> None:
        """Stage exact identities selected through an authority offer.

        Casting and target selection can complete resolution in the same
        engine step, so the post-step stack is intentionally not consulted.
        Staged choices never become presentation facts unless the engine then
        commits the matching spell/damage/resolution domain events.
        """

        focus = action.get("focus")
        if not isinstance(focus, list) or not focus:
            return
        selected = int(focus[0])
        if action.get("type") == "PRIORITY_CAST_SPELL":
            cards = [*observation.agent_cards, *observation.opponent_cards]
            card = next(
                (candidate for candidate in cards if int(candidate.id) == selected),
                None,
            )
            self._pending_bolt = (
                _PendingBolt(
                    object_id=_object_id(selected),
                    controller=actor_index,
                )
                if card is not None and card.name == "Lightning Bolt"
                else None
            )
            return

        if action.get("type") != "CHOOSE_TARGET" or self._pending_bolt is None:
            return
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

        death_move_committed = any(
            _event_type(event) == "CARD_MOVED"
            and int(event.from_zone) == int(ZoneEnum.BATTLEFIELD)
            and int(event.to_zone) == int(ZoneEnum.GRAVEYARD)
            for event in events
        )
        battlefield_ids = {
            int(permanent.id)
            for permanent in [
                *observation.agent_permanents,
                *observation.opponent_permanents,
            ]
        }
        dead_targets = [
            target["id"]
            for target in bolt.targets
            if target["kind"] == "object"
            and int(target["id"]["entity"]) not in battlefield_ids
        ]
        if death_move_committed and dead_targets:
            self._facts.append(
                _PresentationFact(
                    importance="critical",
                    suggested_ms=650,
                    sound="creature.died",
                    kind={"kind": "died", "objects": dead_targets},
                )
            )

        del self._active_bolts[bolt.source_event_id]
