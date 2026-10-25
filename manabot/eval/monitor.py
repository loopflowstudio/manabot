"""Monitoring protocols, result schemas and retained-row statistics.

Training coordinates describe the measured checkpoint; evaluation never advances
them. experiment.checkpoint_evaluation owns checkpoint admission and execution.
import_saved_rows projects existing receipts without loading a model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from manabot.eval.models import EvaluationKey
from manabot.infra.files import atomic_json
from manabot.infra.identity import file_sha256
from manabot.infra.provenance import source_bundle_sha256
from manabot.sim import players as players
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.matchup import (
    AuthoredMatchup,
    selected_decks,
)
from manabot.sim.registration import PlayerRegistration
from manabot.training.records import TrainingCoordinates
from manabot.training.regime import ArtifactReference, Strict

EvaluationPurpose = Literal[
    "monitoring-not-scientific-evaluation",
    "frozen-study-evaluation",
    "supplemental-matchup-diagnostic",
]


class MonitorProtocol(Strict):
    authored_matchup: AuthoredMatchup | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    purpose: EvaluationPurpose = Field(
        default="monitoring-not-scientific-evaluation",
        exclude_if=lambda v: v == "monitoring-not-scientific-evaluation",
    )
    study_protocol_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$", exclude_if=lambda v: v is None
    )
    # Reserved monitoring namespace; callers must also exclude these from any
    # scientific cohort they construct. These deals are repeatedly inspected.
    deal_seeds: tuple[int, ...] = tuple(range(1_910_101_000, 1_910_101_025))
    include_mirrors: bool = Field(default=False, exclude_if=lambda value: not value)

    @property
    def legs(self) -> int:
        return 8 if self.include_mirrors else 4

    bootstrap_seed: int = 101
    bootstrap_replicates: int = Field(default=2000, ge=1)
    game_seconds: float = Field(default=120, gt=0)
    max_commands: int = Field(default=10_000, ge=1)
    opponent: Literal["scripted_greedy", "heuristic", "random"] = Field(
        default="scripted_greedy", exclude_if=lambda value: value == "scripted_greedy"
    )
    # A frozen external comparison owns seeds, selection and analysis. This binds
    # its exact bytes; the evaluator itself supplies no promotion decision.
    comparison_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda value: value is None
    )

    @property
    def result_purpose(
        self,
    ) -> EvaluationPurpose | Literal["predeclared-comparison-not-admission"]:
        """Preserve the external-comparison label without changing frozen bytes."""
        return (
            "predeclared-comparison-not-admission"
            if self.comparison_sha256
            else self.purpose
        )

    @model_validator(mode="after")
    def unique_deals(self) -> MonitorProtocol:
        if (self.purpose == "frozen-study-evaluation") != (
            self.study_protocol_sha256 is not None
        ):
            raise ValueError(
                "held-out evaluation requires a frozen study protocol binding"
            )
        if not self.deal_seeds or len(set(self.deal_seeds)) != len(self.deal_seeds):
            raise ValueError("monitoring deals must be nonempty and unique")
        return self


class EvaluationRow(BaseModel):
    """Narrow the arena JSON boundary while retaining every additional field."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    arena_key: EvaluationKey
    deal_seed: int
    leg: int = Field(ge=0, le=7)
    player_a: str
    player_b: str
    player_a_registration_sha256: str
    player_b_registration_sha256: str
    score_a: float | None = Field(strict=True)
    failure: str | None
    terminated: bool
    truncated: bool
    replay_passed: bool
    trace_path: str
    game_seconds: float = Field(ge=0)
    integrity: dict[str, int]

    @field_validator("score_a")
    @classmethod
    def valid_score(cls, value: float | None) -> float | None:
        if value not in (None, 0.0, 0.5, 1.0):
            raise ValueError("arena score must be loss, draw, win or unavailable")
        return value

    @property
    def valid(self) -> bool:
        return (
            self.failure is None
            and self.terminated
            and not self.truncated
            and self.replay_passed
            and self.score_a is not None
            and not any(self.integrity.values())
        )


class RateInterval(Strict):
    mean: float
    lower: float
    upper: float


class MonitorResult(Strict):
    schema_version: Literal[1] = 1
    purpose: EvaluationPurpose | Literal["predeclared-comparison-not-admission"] = (
        "monitoring-not-scientific-evaluation"
    )
    run_id: str
    regime_digest: str
    training_seed: int
    artifact: ArtifactReference
    coordinates: TrainingCoordinates
    protocol: MonitorProtocol
    key: EvaluationKey
    candidate: PlayerRegistration
    opponent: PlayerRegistration
    status: Literal["running", "completed", "incomplete"] = "running"
    imported: bool = False
    source_rows_path: str | None = None
    source_rows_sha256: str | None = None
    rows: list[EvaluationRow] = []
    expected_games: int
    win: RateInterval | None = None
    draw: RateInterval | None = None
    score: RateInterval | None = None
    evaluation_seconds: float | None = None
    started_unix: float | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    finished_unix: float | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    coordinator_cpu_seconds: float | None = None
    reaped_worker_cpu_seconds: float | None = None
    coordinator_rss_before_bytes: int | None = None
    coordinator_rss_after_bytes: int | None = None
    host_load_before: tuple[float, float, float] | None = None
    host_load_after: tuple[float, float, float] | None = None
    concurrent_activity: str = "unknown; host load is not contention attribution"
    resource_limit: str = (
        "RSS is coordinator-only, not peak; reaped worker CPU includes any other "
        "children reaped by this process during evaluation"
    )
    error: str | None = None
    evaluation_identities: dict[str, JsonValue] = {}

    def validate_cohort(self) -> None:
        """Recompute completeness and paired-deal intervals from retained rows."""
        _summarize(self)

    @model_validator(mode="after")
    def purpose_matches_protocol(self) -> "MonitorResult":
        if self.purpose != self.protocol.result_purpose:
            raise ValueError(
                "evaluation result purpose differs from its frozen protocol"
            )
        return self


def _summarize(result: MonitorResult) -> None:
    expected = {
        (seed, leg)
        for seed in result.protocol.deal_seeds
        for leg in range(result.protocol.legs)
    }
    seen: set[tuple[int, int]] = set()
    for row in result.rows:
        coordinate = (row.deal_seed, row.leg)
        if coordinate not in expected or coordinate in seen:
            raise ValueError("unexpected or duplicate monitoring deal/leg")
        seen.add(coordinate)
        if result.protocol.include_mirrors:
            expected_decks = list(
                result.protocol.authored_matchup.decks(row.leg)
                if result.protocol.authored_matchup
                else selected_decks(row.leg)
            )
            extra = row.model_extra or {}
            if (
                extra.get("seat_decks") != expected_decks
                or extra.get("player_a_seat") != row.leg % 2
            ):
                raise ValueError(
                    "mirror row differs from its actual authored deck/seat leg"
                )
        if (
            row.arena_key != result.key
            or row.player_a != result.candidate.player_id
            or row.player_b != result.opponent.player_id
            or row.player_a_registration_sha256 != result.candidate.identity_sha256
            or row.player_b_registration_sha256 != result.opponent.identity_sha256
        ):
            raise ValueError("arena row differs from frozen monitoring registrations")
    result.win = result.draw = result.score = None
    if seen != expected or result.error or not all(row.valid for row in result.rows):
        result.status = "incomplete"
        return
    result.status = "completed"
    # Resample whole deals, keeping their four correlated seat/deck legs together.
    blocks = np.array(
        [
            [row.score_a for row in result.rows if row.deal_seed == seed]
            for seed in result.protocol.deal_seeds
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(result.protocol.bootstrap_seed)
    indices = rng.integers(
        0, len(blocks), (result.protocol.bootstrap_replicates, len(blocks))
    )
    for name, values in (
        ("win", blocks == 1),
        ("draw", blocks == 0.5),
        ("score", blocks),
    ):
        means = values.mean(axis=1)
        draws = means[indices].mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        setattr(
            result,
            name,
            RateInterval(
                mean=float(means.mean()), lower=float(lower), upper=float(upper)
            ),
        )


OPPONENT_NAMES = {
    "scripted_greedy": "Frozen scripted greedy",
    "heuristic": "Rule-of-thumb heuristic",
    "random": "Uniform legal random",
}


def _opponent_source_sha256(opponent: str) -> str:
    """v1 and random keep their frozen `players.py` identity."""
    if opponent != "heuristic":
        return file_sha256(Path(players.__file__))
    return source_bundle_sha256([Path(path) for path in heuristic_source_paths()])


def import_saved_rows(
    manifest: MonitorResult,
    rows_path: Path,
    output_dir: Path,
) -> MonitorResult:
    """Import original arena rows against their saved manifest, without model loading.

    Paths remain relative to the source rows directory. Replay flags are historical
    claims, not a fresh replay. Original costs/coordinates remain unchanged.
    """
    payload: Any = json.loads(rows_path.read_text())  # Untrusted JSON boundary.
    if not isinstance(payload, list):
        raise ValueError("saved arena rows must be a list")
    result = manifest.model_copy(deep=True)
    result.rows = [EvaluationRow.model_validate(row) for row in payload]
    result.imported = True
    result.source_rows_path = str(rows_path.resolve())
    result.source_rows_sha256 = file_sha256(rows_path)
    _summarize(result)
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "monitor.json", result.model_dump(mode="json"))
    return result
