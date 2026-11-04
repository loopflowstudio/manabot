"""Observed evaluation choices from the existing replay-linked game evidence.

Card draws remain owned by card_metrics. This adds counts of committed native
choices per participant/kind/action, not speculative mechanic activations. The
trace omits complete offer sets and resolution attribution: those quantities
are unavailable, never zero. No second replay or training instrumentation runs.
"""

from collections import Counter
import gzip
from hashlib import sha256
from typing import Literal

from etude.eval.match import Row
from pydantic import BaseModel, TypeAdapter

from manabot.sim.card_evidence import GameEvidence


class _Offer(BaseModel):
    action_type: str


class _Decision(BaseModel):
    actor: Literal[0, 1]
    player_id: str
    action_space_kind: str
    chosen_offer: _Offer
    post_state_digest: str


def decision_rows(blobs: list[bytes]) -> list[Row]:
    """Retain separate seat totals, even when a participant made no decisions."""
    rows: list[Row] = []
    for body in blobs:
        game = GameEvidence.model_validate_json(gzip.decompress(body))
        decisions = TypeAdapter(list[_Decision]).validate_python(
            game.trace.data["decisions"]
        )
        counts: Counter[tuple[int, str, str]] = Counter()
        for seat in (0, 1):
            counts[seat, "all", "all"] = 0
        for decision in decisions:
            if game.seat_players[decision.actor] != decision.player_id:
                raise ValueError("decision actor differs from retained seating")
            seat = ("candidate", "opponent").index(decision.player_id)
            counts[seat, "all", "all"] += 1
            counts[
                seat, decision.action_space_kind, decision.chosen_offer.action_type
            ] += 1
        digest = sha256(body).hexdigest()
        eligible = (
            game.outcome in ("win_loss", "draw") and game.replay.get("passed") is True
        )
        for (seat, kind, action), count in sorted(counts.items()):
            rows.append(
                {
                    "seat": seat,
                    "game": game.leg,
                    "game_seat": game.seat_players.index(
                        ("candidate", "opponent")[seat]
                    ),
                    "decision_kind": kind,
                    "action_type": action,
                    "committed_choices": count,
                    "eligible": eligible,
                    "evidence_sha256": digest,
                    "metric_version": "committed-native-choices-v1",
                }
            )
    return rows
