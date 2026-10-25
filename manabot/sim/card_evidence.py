"""Private native card capture and retained replay evidence; no metric reduction."""

from __future__ import annotations

import gzip
from hashlib import sha256
from pathlib import Path
from typing import Literal

from etude.run.storage import DirectoryStorage
from pydantic import Field, JsonValue, model_validator

from manabot.infra.config import MatchConfig
from manabot.infra.identity import StrictModel, canonical_json, canonical_sha256
from manabot.sim.replay import replay_game

CAPTURE_VERSION = 1

type Zone = Literal[
    "Library", "Hand", "Battlefield", "Graveyard", "Stack", "Exile", "Command"
]

type Seat = Literal[0, 1]


class CardSnapshot(StrictModel):
    copy_id: int = Field(alias="copy", ge=0)
    definition: int = Field(ge=0)
    name: str
    incarnation: int = Field(ge=0)
    owner: Seat
    zone: Zone | None
    origin: Literal["deck", "outside", "generated"]


class CardMove(StrictModel):
    sequence: int = Field(ge=0)
    card: CardSnapshot
    from_: Zone | None = Field(alias="from")
    controller: Seat
    cause: Literal["draw", "selection", "return", "outside", "generated", "move"]
    turn: int = Field(ge=1)


class NativeCapture(StrictModel):
    version: Literal[1]
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial: tuple[CardSnapshot, ...]
    opening: tuple[tuple[int, ...], tuple[int, ...]]
    moves: tuple[CardMove, ...]


class CapturedCards(StrictModel):
    capture: NativeCapture
    final_cards: tuple[CardSnapshot, ...]
    turn: int = Field(ge=1)

    @model_validator(mode="after")
    def complete(self) -> CapturedCards:
        cards = {card.copy_id: card for card in self.capture.initial}
        if len(cards) != len(self.capture.initial):
            raise ValueError("duplicate initial copy")
        for seat in (0, 1):
            opening = self.capture.opening[seat]
            expected = {
                c.copy_id
                for c in cards.values()
                if c.owner == seat and c.zone == "Hand"
            }
            if len(opening) != len(set(opening)) or set(opening) != expected:
                raise ValueError("opening hand differs from setup")
        turn = 1
        for sequence, move in enumerate(self.capture.moves):
            card = move.card
            old = cards.get(card.copy_id)
            if move.sequence != sequence or not turn <= move.turn <= self.turn:
                raise ValueError("card facts out of sequence")
            turn = move.turn
            if old is None:
                if card.origin != "generated" or move.from_ is not None:
                    raise ValueError("unrecorded physical copy")
            elif (
                old.zone != move.from_
                or old.definition != card.definition
                or old.owner != card.owner
                or old.origin != card.origin
                or old.name != card.name
                or card.incarnation
                != old.incarnation + int(old.zone is not None and card.zone is not None)
            ):
                raise ValueError("card movement does not continue its physical copy")
            if move.cause == "draw" and (move.from_, card.zone) != ("Library", "Hand"):
                raise ValueError("actual draw must move library to hand")
            if card.zone == "Hand" and move.cause == "move":
                raise ValueError("hand acquisition needs a cause")
            cards[card.copy_id] = card
        if cards != {card.copy_id: card for card in self.final_cards} or len(
            cards
        ) != len(self.final_cards):
            raise ValueError("capture omitted a committed movement")
        return self


class GameTrace(StrictModel):
    """Typed view at the existing, extensible arena receipt boundary."""

    data: dict[str, JsonValue]


class GameEvidence(StrictModel):
    version: Literal[1] = 1
    # Frozen v1 envelope tag; counting rules are owned by eval.card_metrics.
    metric_version: Literal["gih-copies-v1"] = "gih-copies-v1"
    game: str
    deal: int
    leg: int
    seat_players: tuple[str, str]
    seat_decks: tuple[str, str]
    setup: MatchConfig
    winner: Seat | None
    outcome: Literal[
        "win_loss", "draw", "crash", "timeout", "truncated", "replay_failed"
    ]
    trace: GameTrace
    trace_sha256: str
    replay: dict[str, int | bool]
    replay_sha256: str
    cards: CapturedCards

    @model_validator(mode="after")
    def bound(self) -> GameEvidence:
        trace = dict(self.trace.data)
        digest = trace.pop("game_trace_sha256", None)
        if digest != self.trace_sha256 or canonical_sha256(trace) != digest:
            raise ValueError("trace digest mismatch")
        if canonical_sha256(self.replay) != self.replay_sha256:
            raise ValueError("replay receipt digest mismatch")
        runtime = trace.get("runtime_identities")
        if (
            not isinstance(runtime, dict)
            or runtime.get("content_digest") != self.cards.capture.content_digest
        ):
            raise ValueError("captured content differs from retained runtime")
        if (
            self.game != trace["match_id"]
            or self.deal != trace["deal_seed"]
            or self.leg != trace["leg"]
            or list(self.seat_players) != trace["seat_players"]
            or self.winner != trace["winner"]
            or self.setup.model_dump() != trace["match_hypers"]
        ):
            raise ValueError("evidence differs from trace")
        if list(self.seat_decks) != trace.get("seat_decks", ["mirror", "mirror"]):
            raise ValueError("evidence deck labels differ from trace")
        if self.outcome == "win_loss" and (
            self.winner is None
            or not self.replay.get("passed")
            or trace.get("failure")
            or not trace.get("terminated")
            or trace.get("truncated")
        ):
            raise ValueError("ineligible outcome cannot claim win/loss")
        return self


class EvidenceRef(StrictModel):
    path: str
    sha256: str
    bytes: int
    game: str


def capture_verified_game(trace: GameTrace) -> GameEvidence:
    data = trace.data
    identities = data.get("runtime_identities")
    if not isinstance(identities, dict) or not isinstance(
        identities.get("content_digest"), str
    ):
        raise ValueError("card capture requires recorded runtime/content identity")
    receipt, captured = replay_game(data, capture_cards=True)
    if captured is None:
        raise ValueError("native capture unavailable")
    failure = data.get("failure")
    outcome = (
        "replay_failed"
        if not receipt.passed
        else "timeout"
        if data.get("termination_reason") == "timeout"
        else "crash"
        if failure
        else "truncated"
        if data.get("truncated")
        else "win_loss"
        if data["winner"] is not None
        else "draw"
    )
    replay = receipt.to_dict()
    return GameEvidence.model_validate(
        dict(
            game=data["match_id"],
            deal=data["deal_seed"],
            leg=data["leg"],
            seat_players=data["seat_players"],
            seat_decks=data.get("seat_decks", ["mirror", "mirror"]),
            setup=data["match_hypers"],
            winner=data["winner"],
            outcome=outcome,
            trace=trace,
            trace_sha256=data["game_trace_sha256"],
            replay=replay,
            replay_sha256=canonical_sha256(replay),
            cards=captured,
        )
    )


def retain_evidence(directory: Path, evidence: GameEvidence) -> EvidenceRef:
    body = gzip.compress(
        canonical_json(evidence.model_dump(mode="json", by_alias=True)), mtime=0
    )
    digest = sha256(body).hexdigest()
    key = f"evidence/{digest}.json.gz"
    store = DirectoryStorage(directory)
    if (directory / key).exists():
        if store.get(key) != body:
            raise ValueError("immutable evidence collision")
    else:
        store.put(key, body)
    return EvidenceRef(path=key, sha256=digest, bytes=len(body), game=evidence.game)


def read_evidence(body: bytes, reference: EvidenceRef) -> GameEvidence:
    if len(body) != reference.bytes or sha256(body).hexdigest() != reference.sha256:
        raise ValueError("card evidence bytes differ from reference")
    evidence = GameEvidence.model_validate_json(gzip.decompress(body))
    if evidence.game != reference.game:
        raise ValueError("card evidence game differs from reference")
    return evidence
