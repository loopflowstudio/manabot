"""Game-owned canonical replay decision addresses and viewer projections.

The complete replay mixes decisions made from both players' private
perspectives and therefore never crosses a client boundary.  Consumers receive
one closed projection for an authorized viewer and restore rows through a
viewer-bound ``erd1`` address.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import Field, model_validator

from .experience_protocol import (
    Command,
    ExperienceFrame,
    InteractionOffer,
    PresentationEvent,
    ProtocolModel,
    UInt8,
    UInt64,
)

CANONICAL_REPLAY_VERSION: Literal[1] = 1
ADDRESS_PREFIX = "erd1."
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL_PATTERN = re.compile(r"^(0|[1-9][0-9]*)$")


class ReplayIndexError(ValueError):
    """Base error for canonical replay validation and restoration."""


class InvalidAddressError(ReplayIndexError):
    """The deep-link string is not a canonical erd1 address."""


class DecisionNotFoundError(ReplayIndexError):
    """No decision is visible at this address for the authorized viewer."""


class CanonicalReplayUnavailableError(ReplayIndexError):
    """A legacy trace has no captured canonical replay authority record."""


class DecisionSource(str, Enum):
    CLIENT = "client"
    POLICY = "policy"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def decision_payload_sha256(
    frame: ExperienceFrame,
    offer: InteractionOffer,
    command: Command,
    presentation_cursor: int,
) -> str:
    payload = {
        "frame": frame.model_dump(mode="json"),
        "offer": offer.model_dump(mode="json"),
        "command": command.model_dump(mode="json"),
        "presentation_cursor": presentation_cursor,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def canonical_projection_sha256(
    projection: "CanonicalReplayProjectionV1",
) -> str:
    """Hash the closed semantic viewer projection, independent of formatting."""
    return hashlib.sha256(
        _canonical_json(projection.model_dump(mode="json"))
    ).hexdigest()


class ReplayDecision(ProtocolModel):
    ordinal: UInt64
    viewer: UInt8
    source: DecisionSource
    revision: UInt64
    prompt_id: UInt64
    offer_id: int = Field(ge=0, le=2**32 - 1)
    command_id: str
    presentation_cursor: UInt64
    frame: ExperienceFrame
    offer: InteractionOffer
    command: Command

    @model_validator(mode="after")
    def validate_bindings(self) -> "ReplayDecision":
        prompt = self.frame.prompt
        if (
            prompt is None
            or prompt.id != self.prompt_id
            or prompt.actor != self.viewer
            or self.frame.revision != self.revision
            or self.offer.id != self.offer_id
            or self.offer.actor != self.viewer
            or self.command.command_id != self.command_id
            or self.command.match_id != self.frame.match_id
            or self.command.expected_revision != self.revision
            or self.command.prompt_id != self.prompt_id
            or self.command.offer_id != self.offer_id
        ):
            raise ValueError("decision frame, offer, and command bindings drifted")
        frame_offer = next(
            (
                candidate
                for candidate in self.frame.offers
                if candidate.id == self.offer_id
            ),
            None,
        )
        if frame_offer is None or frame_offer != self.offer:
            raise ValueError("decision offer differs from the captured frame offer")
        projection = self.frame.projection
        if projection.agent.player_index != self.viewer:
            raise ValueError("decision frame is not oriented to its acting viewer")
        if projection.opponent.hand:
            raise ValueError("decision frame exposes opponent-private hand identities")
        return self

    def digest_payload(self) -> dict[str, Any]:
        return {
            "frame": self.frame.model_dump(mode="json"),
            "offer": self.offer.model_dump(mode="json"),
            "command": self.command.model_dump(mode="json"),
            "presentation_cursor": self.presentation_cursor,
        }

    def sha256(self) -> str:
        return decision_payload_sha256(
            self.frame,
            self.offer,
            self.command,
            self.presentation_cursor,
        )


class ViewerPresentationTrack(ProtocolModel):
    viewer: UInt8
    head: UInt64
    events: list[PresentationEvent]

    @model_validator(mode="after")
    def validate_sequence(self) -> "ViewerPresentationTrack":
        for expected, event in enumerate(self.events):
            if event.seq != expected:
                raise ValueError(
                    f"viewer {self.viewer} presentation sequence gap at {expected}"
                )
        if self.head != len(self.events):
            raise ValueError(f"viewer {self.viewer} presentation head drifted")
        return self


class CanonicalReplayV1(ProtocolModel):
    version: Literal[1]
    replay_id: str
    match_id: str
    content_hash: str
    asset_manifest_hash: str
    decisions: list[ReplayDecision]
    presentation_tracks: list[ViewerPresentationTrack]

    @model_validator(mode="after")
    def validate_authority(self) -> "CanonicalReplayV1":
        if not self.replay_id:
            raise ValueError("canonical replay requires replay_id")
        tracks = {track.viewer: track for track in self.presentation_tracks}
        if len(tracks) != len(self.presentation_tracks):
            raise ValueError("canonical replay has duplicate viewer tracks")
        command_ids: set[str] = set()
        prompt_ids: set[tuple[int, int]] = set()
        for expected, decision in enumerate(self.decisions):
            if decision.ordinal != expected:
                raise ValueError(f"canonical replay ordinal gap at {expected}")
            if decision.frame.match_id != self.match_id:
                raise ValueError("decision match differs from canonical replay")
            if (
                decision.frame.content_hash != self.content_hash
                or decision.frame.asset_manifest_hash != self.asset_manifest_hash
            ):
                raise ValueError(
                    "decision content identity differs from canonical replay"
                )
            if decision.command_id in command_ids:
                raise ValueError("canonical replay has duplicate command id")
            command_ids.add(decision.command_id)
            prompt_key = (decision.revision, decision.prompt_id)
            if prompt_key in prompt_ids:
                raise ValueError(
                    "canonical replay has duplicate revision/prompt identity"
                )
            prompt_ids.add(prompt_key)
            track = tracks.get(decision.viewer)
            if track is None or decision.presentation_cursor > track.head:
                raise ValueError("decision cursor is outside its viewer track")
        for viewer, rows in _group_by_viewer(self.decisions).items():
            cursors = [row.presentation_cursor for row in rows]
            if cursors != sorted(cursors):
                raise ValueError(f"viewer {viewer} decision cursors move backwards")
        return self

    def metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "ordinal": row.ordinal,
                "viewer": row.viewer,
                "source": row.source.value,
                "revision": row.revision,
                "prompt_id": row.prompt_id,
                "offer_id": row.offer_id,
                "command_id": row.command_id,
                "presentation_cursor": row.presentation_cursor,
            }
            for row in self.decisions
        ]


class CanonicalReplayProjectionV1(ProtocolModel):
    version: Literal[1]
    replay_id: str
    match_id: str
    content_hash: str
    asset_manifest_hash: str
    viewer: UInt8
    decisions: list[ReplayDecision]
    presentation_head: UInt64
    presentation: list[PresentationEvent]

    @model_validator(mode="after")
    def validate_projection(self) -> "CanonicalReplayProjectionV1":
        if not self.replay_id:
            raise ValueError("canonical projection requires replay_id")
        previous_ordinal = -1
        command_ids: set[str] = set()
        prompt_ids: set[tuple[int, int]] = set()
        previous_cursor = 0
        for row in self.decisions:
            if row.viewer != self.viewer:
                raise ValueError("canonical projection mixes viewer decision rows")
            if (
                row.frame.match_id != self.match_id
                or row.frame.content_hash != self.content_hash
                or row.frame.asset_manifest_hash != self.asset_manifest_hash
            ):
                raise ValueError("decision identity differs from canonical projection")
            if row.ordinal <= previous_ordinal:
                raise ValueError("canonical projection ordinals are not increasing")
            if row.command_id in command_ids:
                raise ValueError("canonical projection has duplicate command id")
            prompt_key = (row.revision, row.prompt_id)
            if prompt_key in prompt_ids:
                raise ValueError("canonical projection has duplicate prompt identity")
            if row.presentation_cursor < previous_cursor:
                raise ValueError("canonical projection decision cursors move backwards")
            if row.presentation_cursor > self.presentation_head:
                raise ValueError(
                    "canonical projection cursor exceeds presentation head"
                )
            previous_ordinal = row.ordinal
            previous_cursor = row.presentation_cursor
            command_ids.add(row.command_id)
            prompt_ids.add(prompt_key)
        for expected, event in enumerate(self.presentation):
            if event.seq != expected:
                raise ValueError(f"canonical projection presentation gap at {expected}")
        if self.presentation_head != len(self.presentation):
            raise ValueError("canonical projection presentation head drifted")
        return self


class ReplayDecisionAddress(ProtocolModel):
    version: Literal[1]
    replay_id: str
    match_id: str
    ordinal: UInt64
    viewer: UInt8
    revision: UInt64
    prompt_id: UInt64
    offer_id: int = Field(ge=0, le=2**32 - 1)
    command_id: str
    presentation_cursor: UInt64
    decision_sha256: str

    @model_validator(mode="after")
    def validate_closed_values(self) -> "ReplayDecisionAddress":
        if not self.replay_id or not self.match_id or not self.command_id:
            raise ValueError("replay address identities must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.decision_sha256):
            raise ValueError("replay address decision_sha256 is not canonical")
        return self

    @classmethod
    def from_decision(
        cls,
        replay: CanonicalReplayV1 | CanonicalReplayProjectionV1,
        row: ReplayDecision,
    ) -> "ReplayDecisionAddress":
        return cls(
            version=CANONICAL_REPLAY_VERSION,
            replay_id=replay.replay_id,
            match_id=replay.match_id,
            ordinal=row.ordinal,
            viewer=row.viewer,
            revision=row.revision,
            prompt_id=row.prompt_id,
            offer_id=row.offer_id,
            command_id=row.command_id,
            presentation_cursor=row.presentation_cursor,
            decision_sha256=row.sha256(),
        )

    def serialize(self) -> str:
        payload = [
            self.version,
            self.replay_id,
            self.match_id,
            str(self.ordinal),
            str(self.viewer),
            str(self.revision),
            str(self.prompt_id),
            str(self.offer_id),
            self.command_id,
            str(self.presentation_cursor),
            self.decision_sha256,
        ]
        encoded = base64.urlsafe_b64encode(_canonical_json(payload)).decode("ascii")
        return ADDRESS_PREFIX + encoded.rstrip("=")

    @classmethod
    def parse(cls, value: str) -> "ReplayDecisionAddress":
        try:
            if not isinstance(value, str) or not value.startswith(ADDRESS_PREFIX):
                raise ValueError("missing erd1 prefix")
            encoded = value[len(ADDRESS_PREFIX) :]
            if not encoded or "=" in encoded:
                raise ValueError("address is not unpadded base64url")
            padding = "=" * (-len(encoded) % 4)
            raw = base64.b64decode(
                encoded + padding,
                altchars=b"-_",
                validate=True,
            )
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, list) or len(payload) != 11:
                raise ValueError("address payload shape")
            if any(not isinstance(payload[index], str) for index in range(1, 11)):
                raise ValueError("address payload types")
            numeric = [payload[index] for index in (3, 4, 5, 6, 7, 9)]
            if any(not _DECIMAL_PATTERN.fullmatch(item) for item in numeric):
                raise ValueError("address numeric encoding")
            address = cls(
                version=payload[0],
                replay_id=payload[1],
                match_id=payload[2],
                ordinal=int(payload[3]),
                viewer=int(payload[4]),
                revision=int(payload[5]),
                prompt_id=int(payload[6]),
                offer_id=int(payload[7]),
                command_id=payload[8],
                presentation_cursor=int(payload[9]),
                decision_sha256=payload[10],
            )
            if address.serialize() != value:
                raise ValueError("non-canonical address encoding")
            return address
        except Exception as exc:
            if isinstance(exc, InvalidAddressError):
                raise
            raise InvalidAddressError("invalid replay decision address") from exc


class RestoredReplayDecision(ProtocolModel):
    address: str
    ordinal: UInt64
    viewer: UInt8
    revision: UInt64
    presentation_cursor: UInt64
    frame: ExperienceFrame
    offer: InteractionOffer
    command: Command
    continuation: list[PresentationEvent]


def _group_by_viewer(
    decisions: list[ReplayDecision],
) -> dict[int, list[ReplayDecision]]:
    grouped: dict[int, list[ReplayDecision]] = {}
    for decision in decisions:
        grouped.setdefault(int(decision.viewer), []).append(decision)
    return grouped


def project_replay(
    replay: CanonicalReplayV1,
    authorized_viewer: int,
) -> CanonicalReplayProjectionV1:
    track = next(
        (
            candidate
            for candidate in replay.presentation_tracks
            if candidate.viewer == authorized_viewer
        ),
        None,
    )
    if track is None:
        raise DecisionNotFoundError("decision not found")
    return CanonicalReplayProjectionV1(
        version=CANONICAL_REPLAY_VERSION,
        replay_id=replay.replay_id,
        match_id=replay.match_id,
        content_hash=replay.content_hash,
        asset_manifest_hash=replay.asset_manifest_hash,
        viewer=authorized_viewer,
        decisions=[
            row.model_copy(deep=True)
            for row in replay.decisions
            if row.viewer == authorized_viewer
        ],
        presentation_head=track.head,
        presentation=[event.model_copy(deep=True) for event in track.events],
    )


def projection_with_addresses(
    projection: CanonicalReplayProjectionV1,
) -> dict[str, Any]:
    payload = projection.model_dump(mode="json")
    for row_payload, row in zip(
        payload["decisions"], projection.decisions, strict=True
    ):
        row_payload["address"] = ReplayDecisionAddress.from_decision(
            projection, row
        ).serialize()
    return payload


def restore_decision(
    replay: CanonicalReplayV1,
    raw_address: str,
    authorized_viewer: int,
) -> RestoredReplayDecision:
    address = ReplayDecisionAddress.parse(raw_address)
    if address.viewer != authorized_viewer:
        raise DecisionNotFoundError("decision not found")
    if address.replay_id != replay.replay_id or address.match_id != replay.match_id:
        raise DecisionNotFoundError("decision not found")
    if address.ordinal >= len(replay.decisions):
        raise DecisionNotFoundError("decision not found")
    row = replay.decisions[address.ordinal]
    expected = ReplayDecisionAddress.from_decision(replay, row)
    if address != expected or row.viewer != authorized_viewer:
        raise DecisionNotFoundError("decision not found")
    track = next(
        (
            candidate
            for candidate in replay.presentation_tracks
            if candidate.viewer == authorized_viewer
        ),
        None,
    )
    if track is None:
        raise DecisionNotFoundError("decision not found")
    later_cursor = next(
        (
            candidate.presentation_cursor
            for candidate in replay.decisions[address.ordinal + 1 :]
            if candidate.viewer == authorized_viewer
        ),
        track.head,
    )
    has_earlier_viewer_row = any(
        candidate.viewer == authorized_viewer
        for candidate in replay.decisions[: address.ordinal]
    )
    continuation_start = row.presentation_cursor if has_earlier_viewer_row else 0
    continuation = [
        event.model_copy(deep=True)
        for event in track.events
        if continuation_start <= event.seq < later_cursor
    ]
    return RestoredReplayDecision(
        address=raw_address,
        ordinal=row.ordinal,
        viewer=row.viewer,
        revision=row.revision,
        presentation_cursor=row.presentation_cursor,
        frame=row.frame.model_copy(deep=True),
        offer=row.offer.model_copy(deep=True),
        command=row.command.model_copy(deep=True),
        continuation=continuation,
    )


def load_canonical_replay(payload: dict[str, Any]) -> CanonicalReplayV1:
    raw = payload.get("canonical_replay")
    if raw is None:
        raise CanonicalReplayUnavailableError("canonical replay unavailable")
    return CanonicalReplayV1.model_validate(deepcopy(raw))


def _load_trace(path: Path) -> CanonicalReplayV1:
    return load_canonical_replay(json.loads(path.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a canonical replay index")
    parser.add_argument("trace", type=Path, nargs="?")
    parser.add_argument(
        "--pinned-match",
        action="store_true",
        help="generate the deterministic curated authority fixture in memory",
    )
    parser.add_argument("--list-authority", action="store_true")
    parser.add_argument("--viewer", type=int, choices=(0, 1))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--restore")
    args = parser.parse_args(argv)

    if args.pinned_match:
        from scripts.generate_replay_fixtures import _play_pinned_match

        generated = _play_pinned_match()
        # ``python -m`` loads this module as ``__main__`` while the generator
        # imports its package name. Re-validate the wire value so Pydantic does
        # not see two nominal copies of the same model class.
        replay = CanonicalReplayV1.model_validate(generated.model_dump(mode="json"))
    elif args.trace is not None:
        replay = _load_trace(args.trace)
    else:
        parser.error("provide a trace path or --pinned-match")
    if args.list_authority:
        print(json.dumps(replay.metadata(), indent=2))
        return 0
    if args.viewer is None:
        parser.error("--viewer is required for projection or restoration")
    if args.restore:
        restored = restore_decision(replay, args.restore, args.viewer)
        print(json.dumps(restored.model_dump(mode="json", exclude_none=True), indent=2))
        return 0
    if args.list:
        print(
            json.dumps(
                projection_with_addresses(project_replay(replay, args.viewer)), indent=2
            )
        )
        return 0
    parser.error("choose --list-authority, --list, or --restore")


if __name__ == "__main__":
    raise SystemExit(main())
