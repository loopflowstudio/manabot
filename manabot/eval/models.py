"""Evaluation identities and the per-game match row.

`EvaluationKey` names the arena a game was played in; `MatchRow` is one played
leg. Wire fields retain their recorded names because frozen evidence carries them."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from manabot.infra.identity import SHA256_PATTERN, StrictModel
from manabot.sim.registration import WorldVersion


class EvaluationKey(StrictModel):
    world: WorldVersion
    content_suite: str
    viewer_boundary: str
    arena_version: str
    rating_model_version: str
    rating_prior_sha256: str = Field(pattern=SHA256_PATTERN)
    anchor_cohort_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluation_compute_envelope_id: str
    seed_compute_envelope_id: str | None = Field(
        default=None, exclude_if=lambda v: v is None
    )


class MatchRow(StrictModel):
    arena_key: EvaluationKey
    cell_id: str
    deal_block: int = Field(ge=0)
    deal_seed: int
    deal_seed_set_sha256: str = Field(pattern=SHA256_PATTERN)
    leg: Literal[0, 1, 2, 3]
    seat_decks: tuple[str, str] | None = None
    player_a: str
    player_b: str
    player_a_registration_sha256: str
    player_b_registration_sha256: str
    player_a_compute_class: str
    player_b_compute_class: str
    player_a_seed: int = Field(ge=0)
    player_b_seed: int = Field(ge=0)
    player_a_seat: Literal[0, 1]
    winner: Literal[0, 1] | None
    score_a: float | None = Field(ge=0.0, le=1.0)
    failed_player_id: str | None = None
    failure: str | None = None
    terminated: bool
    truncated: bool
    termination_reason: Literal[
        "terminal", "draw", "truncated", "crash", "timeout", "command_cap"
    ]
    decisions: int = Field(ge=0)
    game_trace_sha256: str = Field(pattern=SHA256_PATTERN)
    trace_path: str
    trace_sha256: str = Field(pattern=SHA256_PATTERN)
    trace_shard_sha256: str = Field(pattern=SHA256_PATTERN)
    replay_passed: bool
    integrity: dict[str, Any]
    latency: dict[str, Any]
    game_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_match_result(self) -> "MatchRow":
        if self.player_a == self.player_b:
            raise ValueError("arena match players must be distinct")
        if self.cell_id != "__".join(sorted((self.player_a, self.player_b))):
            raise ValueError("match cell identity mismatch")
        if self.player_a_seat != self.leg % 2:
            raise ValueError("match leg and player-A seat mismatch")
        if self.arena_key.world == "w2" and self.leg > 1:
            raise ValueError("historical mirror has only two legs")
        failed = self.termination_reason in {
            "truncated",
            "crash",
            "timeout",
            "command_cap",
        }
        if self.failed_player_id not in {None, self.player_a, self.player_b}:
            raise ValueError("failed player is outside this match")
        if self.failed_player_id and not failed:
            raise ValueError("completed match cannot name a failed player")
        if failed:
            expected_score = (
                None
                if self.failed_player_id is None
                else float(self.failed_player_id == self.player_b)
            )
        else:
            expected_score = (
                0.5 if self.winner is None else float(self.winner == self.player_a_seat)
            )
        if self.score_a != expected_score:
            raise ValueError("match score does not agree with winner and seat")
        if self.truncated != (self.termination_reason == "truncated"):
            raise ValueError("match truncation reason mismatch")
        if not failed and not self.terminated:
            raise ValueError("nonterminal match cannot enter arena evidence")
        if self.seat_decks is not None and self.seat_decks != (
            ("ur_lessons", "gw_allies") if self.leg < 2 else ("gw_allies", "ur_lessons")
        ):
            raise ValueError("deck assignment and leg mismatch")
        if self.arena_key.world != "w2" and self.seat_decks is None:
            raise ValueError("selected match must retain its deck assignment")
        if not failed and self.failure is not None:
            raise ValueError("completed match cannot carry a failure")
        if failed and (self.terminated or self.winner is not None or not self.failure):
            raise ValueError("failed match must retain a failure, not a game winner")
        if not failed and (self.termination_reason == "draw") != (self.winner is None):
            raise ValueError("draw requires authoritative termination without a winner")
        if self.trace_sha256 != self.game_trace_sha256:
            raise ValueError("match game-trace digest mismatch")
        trace_path = Path(self.trace_path)
        if trace_path.is_absolute() or ".." in trace_path.parts:
            raise ValueError("match trace locator must stay artifact-relative")
        if self.trace_path not in {
            f"traces/{self.cell_id}.commands.jsonl.gz",  # frozen panel evidence
            f"traces/{self.cell_id}.{self.deal_seed}.commands.jsonl.gz",
        }:
            raise ValueError("match trace locator does not agree with cell identity")
        return self
