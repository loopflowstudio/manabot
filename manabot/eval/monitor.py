"""Monitoring protocols and retained-row statistics. Training coordinates describe the
measured checkpoint; experiment.checkpoint_evaluation owns execution and admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    Field,
    JsonValue,
    model_validator,
)

from manabot.eval.admission import AnchorKind
from manabot.eval.evaluation import EvaluationRequest, EvaluationResult
from manabot.eval.models import EvaluationKey
from manabot.eval.results import EvaluationRow, RateInterval
from manabot.infra.files import atomic_json
from manabot.infra.identity import file_sha256
from manabot.sim.matchup import AuthoredMatchup
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
    opponent: AnchorKind = Field(
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


def _request(
    result: MonitorResult, seeds: tuple[int, ...] | None = None
) -> EvaluationRequest:
    return EvaluationRequest(
        key=result.key,
        candidate=result.candidate,
        opponent=result.opponent,
        checkpoint_paths={result.candidate.player_id: result.artifact["path"]},
        deal_seeds=result.protocol.deal_seeds if seeds is None else seeds,
        purpose=result.purpose,
        matchup_mode="all" if result.protocol.include_mirrors else "cross",
        authored_matchup=result.protocol.authored_matchup,
        game_seconds=result.protocol.game_seconds,
        max_commands=result.protocol.max_commands,
        bootstrap_seed=result.protocol.bootstrap_seed,
        bootstrap_replicates=result.protocol.bootstrap_replicates,
    )


def _summarize(result: MonitorResult) -> None:
    summary = EvaluationResult(
        request=_request(result), rows=result.rows, error=result.error
    )
    summary.validate_cohort()
    result.status = summary.status
    result.win, result.draw, result.score = summary.win, summary.draw, summary.score


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
