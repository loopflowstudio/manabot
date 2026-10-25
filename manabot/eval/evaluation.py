"""Placement-independent evaluation requests, execution and retained results.

An EvaluationRequest names registrations, deals and arena semantics, without a
training run. evaluate uses an allocation supplied by any caller; evaluate_local
owns a bounded CPU allocation without callback transport or remote services.
Only complete replayed deal cohorts produce paired-deal score intervals.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from manabot.eval.execution import bind_execution
from manabot.eval.match import MatchupMode, play_cell
from manabot.eval.models import EvaluationKey
from manabot.eval.results import EvaluationRow, RateInterval
from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel
from manabot.sim.execution import EvalExecution
from manabot.sim.matchup import AuthoredMatchup, selected_decks
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import PlayerRegistration
from manabot.sim.serving import InferenceRuntime

if TYPE_CHECKING:
    from manabot.eval.endpoint import InferenceEndpoint


class EvaluationRequest(StrictModel):
    key: EvaluationKey
    candidate: PlayerRegistration
    opponent: PlayerRegistration
    checkpoint_paths: dict[str, str] = Field(default_factory=dict)
    deal_seeds: tuple[int, ...]
    purpose: str = Field(default="local-check-not-admission", min_length=1)
    matchup_mode: MatchupMode = "cross"
    authored_matchup: AuthoredMatchup | None = None
    game_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    max_commands: int = Field(default=10_000, ge=1)
    bootstrap_seed: int = 101
    bootstrap_replicates: int = Field(default=2000, ge=1)

    @model_validator(mode="after")
    def valid_cohort(self) -> "EvaluationRequest":
        if not self.deal_seeds or len(set(self.deal_seeds)) != len(self.deal_seeds):
            raise ValueError("evaluation deals must be nonempty and unique")
        if self.candidate.player_id == self.opponent.player_id:
            raise ValueError("evaluation players must be distinct")
        for player in (self.candidate, self.opponent):
            if (
                player.runner_kind == "checkpoint"
                and player.player_id not in self.checkpoint_paths
            ):
                raise ValueError("checkpoint candidate bytes are unavailable")
        return self

    @property
    def legs(self) -> range:
        if self.matchup_mode == "all":
            return range(8)
        if self.matchup_mode == "mirrors":
            return range(4, 8)
        return range(2 if self.key.content_suite == "w2-interactive-mirror-v1" else 4)

    @property
    def expected_games(self) -> int:
        return len(self.deal_seeds) * len(self.legs)

    def bind(self, execution: EvalExecution) -> "EvaluationRequest":
        key, (candidate, opponent) = bind_execution(
            self.key, (self.candidate, self.opponent), execution
        )
        return self.model_copy(
            update=dict(key=key, candidate=candidate, opponent=opponent)
        )


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    request: EvaluationRequest
    execution: EvalExecution | None = None
    status: Literal["running", "completed", "incomplete"] = "running"
    rows: list[EvaluationRow] = Field(default_factory=list)
    win: RateInterval | None = None
    draw: RateInterval | None = None
    score: RateInterval | None = None
    error: str | None = None
    evaluation_seconds: float | None = None

    def validate_cohort(self) -> None:
        """Recompute completeness without treating failed or absent rows as losses."""
        request = self.request
        expected = {(seed, leg) for seed in request.deal_seeds for leg in request.legs}
        seen: set[tuple[int, int]] = set()
        self.win = self.draw = self.score = None
        self.status = "incomplete"
        for row in self.rows:
            coordinate = (row.deal_seed, row.leg)
            if coordinate not in expected or coordinate in seen:
                raise ValueError("unexpected or duplicate evaluation deal/leg")
            seen.add(coordinate)
            if request.matchup_mode != "cross":
                decks = list(
                    request.authored_matchup.decks(row.leg)
                    if request.authored_matchup
                    else selected_decks(row.leg)
                )
                extra = row.model_extra or {}
                if (
                    extra.get("seat_decks") != decks
                    or extra.get("player_a_seat") != row.leg % 2
                ):
                    raise ValueError(
                        "mirror row differs from its actual authored deck/seat leg"
                    )
            if (
                row.arena_key != request.key
                or row.player_a != request.candidate.player_id
                or row.player_b != request.opponent.player_id
                or row.player_a_registration_sha256 != request.candidate.identity_sha256
                or row.player_b_registration_sha256 != request.opponent.identity_sha256
            ):
                raise ValueError(
                    "arena row differs from frozen evaluation registrations"
                )
        if seen != expected or self.error or not all(row.valid for row in self.rows):
            return
        self.status = "completed"
        # A deal's correlated deck/seat legs always move together in the bootstrap.
        blocks = np.array(
            [
                [row.score_a for row in self.rows if row.deal_seed == seed]
                for seed in request.deal_seeds
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(request.bootstrap_seed)
        indices = rng.integers(
            0, len(blocks), (request.bootstrap_replicates, len(blocks))
        )
        for name, values in (
            ("win", blocks == 1),
            ("draw", blocks == 0.5),
            ("score", blocks),
        ):
            means = values.mean(axis=1)
            lower, upper = np.quantile(means[indices].mean(axis=1), [0.025, 0.975])
            setattr(
                self,
                name,
                RateInterval(
                    mean=float(means.mean()), lower=float(lower), upper=float(upper)
                ),
            )


def evaluate(
    request: EvaluationRequest,
    *,
    out_dir: Path,
    execution: EvalExecution | None = None,
    inference: InferenceRuntime | InferenceEndpoint | None = None,
    card_evidence_dir: Path | None = None,
) -> EvaluationResult:
    """Execute and retain one panel through the same arena finalizer for all callers.

    Claim a new directory atomically; even an empty existing directory is rejected
    before writing evidence or starting games. Callers own allocation lifetime.
    Admission/execution failures retain an incomplete result; the arena retains detailed
    failure journals, unattempted identities and performance receipts.
    """
    if inference is not None:
        if execution is not None and execution != inference.execution:
            raise ValueError("conflicting execution declaration")
        execution = inference.execution
    if execution is not None:
        request = request.bind(execution)
    out_dir.mkdir(parents=True, exist_ok=False)
    result = EvaluationResult(request=request, execution=execution)
    atomic_json(out_dir / "evaluation.json", result.model_dump(mode="json"))
    started = time.perf_counter()
    performance = EvalPerformance(
        "evaluation", execution.instrumentation if execution else True
    )
    try:
        runner = (
            inference.play_cell
            if inference is not None and not isinstance(inference, InferenceRuntime)
            else partial(play_cell, execution=execution, inference=inference)
        )
        rows, trace, replay = runner(
            key=request.key,
            player_a=request.candidate,
            player_b=request.opponent,
            deal_seeds=request.deal_seeds,
            out_dir=out_dir,
            checkpoint_paths=request.checkpoint_paths,
            game_seconds=request.game_seconds,
            max_commands=request.max_commands,
            matchup_mode=request.matchup_mode,
            authored_matchup=request.authored_matchup,
            **(
                {"card_evidence_dir": card_evidence_dir}
                if card_evidence_dir is not None
                else {}
            ),
        )
        for name, payload in (("rows", rows), ("trace", trace), ("replay", replay)):
            atomic_json(out_dir / f"{name}.json", payload)
        result.rows = [EvaluationRow.model_validate(row) for row in rows]
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        if not (out_dir / "performance.json").exists():
            performance.count("expected_games", request.expected_games)
            performance.count("unattempted_games", request.expected_games)
            performance.write(out_dir / "performance.json")
        if not (out_dir / "unattempted.json").exists():
            cell = "__".join(
                sorted((request.candidate.player_id, request.opponent.player_id))
            )
            atomic_json(
                out_dir / "unattempted.json",
                [
                    f"{request.key.arena_version}:{cell}:{seed}:{leg}"
                    for seed in request.deal_seeds
                    for leg in request.legs
                ],
            )
    finally:
        result.evaluation_seconds = time.perf_counter() - started
        try:
            result.validate_cohort()
        except ValueError as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        atomic_json(out_dir / "evaluation.json", result.model_dump(mode="json"))
    return result


def evaluate_local(
    request: EvaluationRequest, *, limits: EvalExecution, out_dir: Path
) -> EvaluationResult:
    """Own a fresh CPU attempt. No endpoint, provider lookup or cloud fallback.

    Registration preparation is separate; admission of the registered bytes,
    inference, native play and replay all consume this absolute deadline.
    """
    if limits.device != "cpu":
        raise ValueError("local evaluation requires explicit CPU execution")
    return evaluate(request, out_dir=out_dir, execution=limits)
