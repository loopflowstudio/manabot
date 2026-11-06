"""Private native card capture and retained replay evidence; no metric reduction."""

from __future__ import annotations

import gzip
from hashlib import sha256
from pathlib import Path
from typing import Literal

from etude.run.storage import DirectoryStorage
from pydantic import JsonValue, model_validator

from manabot.env.card_capture import CapturedCards, Seat
from manabot.infra.config import MatchConfig
from manabot.infra.identity import StrictModel, canonical_json, canonical_sha256
from manabot.sim.replay import replay_game

CAPTURE_VERSION = 1


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


def evidence_key(digest: str) -> str:
    """Where `retain_evidence` writes the evidence whose bytes have SHA256 `digest`."""
    return f"evidence/{digest}.json.gz"


def retain_evidence(directory: Path, evidence: GameEvidence) -> EvidenceRef:
    """Write a game's evidence once under `directory`, named by its content.

    This hands the replay worker's evidence to the evaluation that played the
    game; that evaluation returns the bytes to Etude as a blob of its match
    (`manabot.eval.card_evaluation`), and the file is scratch.
    """
    body = gzip.compress(
        canonical_json(evidence.model_dump(mode="json", by_alias=True)), mtime=0
    )
    digest = sha256(body).hexdigest()
    key = evidence_key(digest)
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
