"""Pure-Python mirror of managym's shared semantic decision contract.

These dataclasses parse the canonical JSON produced by the PyO3 methods
``Env.semantic_decision_frame_json``, ``Env.semantic_observation_json``, and
``Env.execute_semantic_command_json`` (see ``managym/src/decision.rs``).

This is the cross-package contract surface for docs/ARCHITECTURE.md step 1 /
Rules R1: manabot (agent/search) and etude (experience) both consume it so
the two sides share one revision-bound, viewer-safe, fail-closed authority.
Positional action indices and offer decoding remain private to Rust; match
identity stays Etude-owned; possible-world semantics (RUL-8) are not here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

SEMANTIC_DECISION_VERSION: int = 2


class SemanticContractError(Exception):
    """The shared semantic contract rejected a command or projection."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DecisionFrame:
    """One revision-bound semantic decision: actor, offer-set fingerprint,
    and the legal offers projected from the authoritative action space."""

    schema_version: int
    revision: int
    actor: int
    fingerprint: str
    offers: tuple[Mapping[str, Any], ...]
    object_candidates: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_json(cls, text: str) -> "DecisionFrame":
        payload = json.loads(text)
        return cls(
            schema_version=int(payload["schema_version"]),
            revision=int(payload["revision"]),
            actor=int(payload["actor"]),
            fingerprint=str(payload["fingerprint"]),
            offers=tuple(payload["offers"]),
            object_candidates=tuple(payload.get("object_candidates", ())),
        )

    def offer(self, offer_id: int) -> Mapping[str, Any]:
        for offer in self.offers:
            if offer["id"] == offer_id:
                return offer
        raise SemanticContractError(
            f"offer {offer_id} absent from frame at revision {self.revision}"
        )

    def find_verb(self, verb: str) -> Mapping[str, Any]:
        for offer in self.offers:
            if offer["verb"] == verb:
                return offer
        raise SemanticContractError(f"no {verb} offer at revision {self.revision}")


@dataclass(frozen=True)
class Command:
    """One atomic, revision-bound semantic commitment."""

    command_id: str
    expected_revision: int
    offer_id: int
    answers: tuple[Mapping[str, Any], ...] = ()
    object_preconditions: tuple[Mapping[str, Any], ...] = ()

    def to_json(self) -> str:
        return json.dumps(
            {
                "command_id": self.command_id,
                "expected_revision": self.expected_revision,
                "offer_id": self.offer_id,
                "answers": list(self.answers),
                "object_preconditions": list(self.object_preconditions),
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class TransitionReceipt:
    """Fail-closed record of one accepted transition."""

    schema_version: int
    before_revision: int
    after_revision: int
    command_id: str
    events: tuple[str, ...]
    next_decision: str | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TransitionReceipt":
        return cls(
            schema_version=int(payload["schema_version"]),
            before_revision=int(payload["before_revision"]),
            after_revision=int(payload["after_revision"]),
            command_id=str(payload["command_id"]),
            events=tuple(payload["events"]),
            next_decision=payload["next_decision"],
        )


@dataclass(frozen=True)
class Observation:
    """Composite viewer-safe Observation at one revision."""

    schema_version: int
    revision: int
    viewer: int
    viewer_state_hash: str
    viewer_state: Mapping[str, Any]
    events: tuple[str, ...]
    decision: DecisionFrame | None

    @classmethod
    def from_json(cls, text: str) -> "Observation":
        payload = json.loads(text)
        identity = payload["identity"]
        decision = None
        if payload["decision"] is not None:
            decision = DecisionFrame.from_json(json.dumps(payload["decision"]))
        return cls(
            schema_version=int(identity["schema_version"]),
            revision=int(identity["revision"]),
            viewer=int(identity["viewer"]),
            viewer_state_hash=str(identity["viewer_state_hash"]),
            viewer_state=payload["viewer_state"],
            events=tuple(payload["events"]),
            decision=decision,
        )

    def opponent_hand_is_hidden(self) -> bool:
        """Viewer-safety invariant: no opponent-private hand identities leak."""
        hand_zone = 1
        return all(
            card.get("zone") != hand_zone
            for card in self.viewer_state.get("opponent_cards", [])
        )


@dataclass(frozen=True)
class SemanticTransition:
    """Receipt plus next composite Observation for the command's actor."""

    receipt: TransitionReceipt
    observation: Observation

    @classmethod
    def from_json(cls, text: str) -> "SemanticTransition":
        payload = json.loads(text)
        return cls(
            receipt=TransitionReceipt.from_payload(payload["receipt"]),
            observation=Observation.from_json(json.dumps(payload["observation"])),
        )


def apply_semantic_command(
    env: Any, command: Command | Mapping[str, Any]
) -> SemanticTransition:
    """Apply one revision-bound Command through managym's authority.

    Engine rejections (stale revision, unknown offer, illegal answers) are
    re-raised as :class:`SemanticContractError` so consumers do not depend on
    managym's own error type. No mutation occurs on any rejection.
    """
    text = (
        command.to_json()
        if isinstance(command, Command)
        else json.dumps(dict(command), sort_keys=True, separators=(",", ":"))
    )
    try:
        result = env.execute_semantic_command_json(text)
    except Exception as error:  # managym.AgentError
        message = str(error)
        code = message.split(":", 1)[0] if ":" in message else None
        raise SemanticContractError(message, code=code) from error
    return SemanticTransition.from_json(result)
