"""Placement-independent evaluation requests, execution and retained results.

An EvaluationRequest names registrations, deals and arena semantics, without a
training run. evaluate uses an allocation supplied by any caller; evaluate_local
owns a bounded CPU allocation without callback transport or remote services.
Only complete replayed deal cohorts produce paired-deal score intervals.
"""

from __future__ import annotations

from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from manabot.eval.execution import bind_execution, resolve_execution
from manabot.eval.match import MatchupMode, play_cell
from manabot.eval.models import EvaluationKey
from manabot.eval.results import EvaluationRow, RateInterval
from manabot.infra.artifacts import verify_file
from manabot.infra.files import atomic_json, exclusive_file_lock
from manabot.infra.identity import StrictModel, canonical_sha256, file_sha256
from manabot.sim.execution import EvalExecution, EvalExecutionV1
from manabot.sim.matchup import AuthoredMatchup, selected_decks
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import PlayerRegistration
from manabot.sim.serving_protocol import InferenceRuntime

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
    evaluation_id: str | None = None
    card_evidence_dir: str | None = None
    # Retained v1 allocations stay readable; they are never continued under v2.
    execution: EvalExecution | EvalExecutionV1 | None = None
    status: Literal["running", "completed", "incomplete"] = "running"
    rows: list[EvaluationRow] = Field(default_factory=list)
    win: RateInterval | None = None
    draw: RateInterval | None = None
    score: RateInterval | None = None
    error: str | None = None
    evaluation_seconds: float | None = None

    def complete_deals(self) -> tuple[int, ...]:
        """Validate identities and return only whole, replayed, valid blocks."""
        return tuple(self._complete_blocks())

    def _complete_blocks(self) -> dict[int, list[EvaluationRow]]:
        """Group once so validation, statistics and continuation use the same rows."""
        request = self.request
        blocks: dict[int, dict[int, EvaluationRow]] = {
            seed: {} for seed in request.deal_seeds
        }
        legs = request.legs
        candidate_digest = request.candidate.identity_sha256
        opponent_digest = request.opponent.identity_sha256
        for row in self.rows:
            if (
                row.deal_seed not in blocks
                or row.leg not in legs
                or row.leg in blocks[row.deal_seed]
            ):
                raise ValueError("unexpected or duplicate evaluation deal/leg")
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
                or row.player_a_registration_sha256 != candidate_digest
                or row.player_b_registration_sha256 != opponent_digest
            ):
                raise ValueError(
                    "arena row differs from frozen evaluation registrations"
                )
            blocks[row.deal_seed][row.leg] = row
        return {
            seed: list(rows.values())
            for seed, rows in blocks.items()
            if len(rows) == len(legs) and all(row.valid for row in rows.values())
        }

    def validate_cohort(self) -> None:
        """Recompute completeness without treating failed or absent rows as losses."""
        self.win = self.draw = self.score = None
        self.status = "incomplete"
        request = self.request
        complete = self._complete_blocks()
        if len(complete) != len(request.deal_seeds) or self.error:
            return
        self.status = "completed"
        # A deal's correlated deck/seat legs always move together in the bootstrap.
        blocks = np.array(
            [[row.score_a for row in rows] for rows in complete.values()],
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


def _execute_attempt(
    request: EvaluationRequest,
    *,
    out_dir: Path,
    evaluation_id: str,
    execution: EvalExecution | None = None,
    inference: InferenceRuntime | InferenceEndpoint | None = None,
    card_evidence_dir: Path | None = None,
) -> EvaluationResult:
    """Execute in an exclusively claimed directory; retain every failure."""
    result = EvaluationResult(
        request=request,
        execution=execution,
        evaluation_id=evaluation_id,
        card_evidence_dir=str(card_evidence_dir)
        if card_evidence_dir is not None
        else None,
    )
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
            card_evidence_dir=card_evidence_dir,
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
        if not result.rows and (out_dir / "rows.json").exists():
            result.rows = [
                EvaluationRow.model_validate(row)
                for row in json.loads((out_dir / "rows.json").read_bytes())
            ]
        result.evaluation_seconds = time.perf_counter() - started
        try:
            result.validate_cohort()
        except ValueError as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        atomic_json(out_dir / "evaluation.json", result.model_dump(mode="json"))
    return result


def evaluate(
    request: EvaluationRequest,
    *,
    out_dir: Path,
    execution: EvalExecution | None = None,
    inference: InferenceRuntime | InferenceEndpoint | None = None,
    card_evidence_dir: Path | None = None,
) -> EvaluationResult:
    """Start a fresh logical evaluation; existing directories are never replaced.

    Explicit continuation uses continue_evaluation. The first attempt stays at
    this root; subsequent attempts retain their own subdirectories and receipts.
    """
    execution = resolve_execution(execution, inference)
    if execution is not None:
        request = request.bind(execution)
    out_dir.mkdir(parents=True, exist_ok=False)
    with exclusive_file_lock(out_dir / ".evaluation.lock"):
        return _execute_attempt(
            request,
            out_dir=out_dir,
            evaluation_id=uuid4().hex,
            execution=execution,
            inference=inference,
            card_evidence_dir=card_evidence_dir,
        )


def _retained_result(directory: Path) -> EvaluationResult:
    result = EvaluationResult.model_validate_json(
        (directory / "evaluation.json").read_bytes()
    )
    # rows.json is published after every block, before the final attempt receipt.
    # It remains authoritative if the coordinator was killed mid-panel.
    if (directory / "rows.json").exists():
        result.rows = [
            EvaluationRow.model_validate(row)
            for row in json.loads((directory / "rows.json").read_bytes())
        ]
    return result


def _verify_block(directory: Path, rows: list[EvaluationRow]) -> None:
    seed = rows[0].deal_seed
    commit = json.loads((directory / "blocks" / f"{seed}.json").read_bytes())
    block = sorted(rows, key=lambda row: row.leg)
    if (
        canonical_sha256([row.model_dump(mode="json") for row in block])
        != commit["rows_sha256"]
    ):
        raise ValueError("retained block rows digest mismatch")
    verified: dict[Path, str] = {}
    for row in rows:
        path = (directory / row.trace_path).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("retained trace escapes its attempt directory")
        digest = (row.model_extra or {}).get("trace_shard_sha256")
        if not isinstance(digest, str):
            raise ValueError("retained block lacks a trace digest")
        if path not in verified:
            verified[path] = file_sha256(path)
        if verified[path] != digest:
            raise ValueError("retained trace digest mismatch")


def evaluation_status(directory: Path) -> EvaluationResult:
    """Derive progress from retained evidence, without execution or a scheduler.

    Failed attempts remain on disk. Only complete valid blocks contribute; a
    duplicate complete block is rejected rather than counted as another sample.
    Atomic row snapshots allow inspection while an attempt is still running.
    """
    original = _retained_result(directory)
    if original.evaluation_id is None:
        # Frozen receipts predate continuation; they remain readable, but the
        # continuation entry point refuses to reinterpret them as resumable work.
        original.validate_cohort()
        return original
    result = EvaluationResult(
        request=original.request,
        execution=original.execution,
        evaluation_id=original.evaluation_id,
        card_evidence_dir=original.card_evidence_dir,
    )
    completed: set[int] = set()
    attempts = [directory, *sorted((directory / "attempts").glob("*/"))]
    for attempt in attempts:
        if not (attempt / "evaluation.json").exists():
            continue  # killed before admission; no published block can exist
        retained = original if attempt == directory else _retained_result(attempt)
        expected = original.request.model_copy(
            update={
                "deal_seeds": retained.request.deal_seeds,
                "checkpoint_paths": retained.request.checkpoint_paths,
            }
        )
        if (
            retained.evaluation_id != original.evaluation_id
            or retained.request != expected
            or not set(retained.request.deal_seeds) <= set(original.request.deal_seeds)
            or (retained.execution.compute if retained.execution else None)
            != (original.execution.compute if original.execution else None)
        ):
            raise ValueError("retained attempt differs from the evaluation contract")
        blocks = retained._complete_blocks()
        if blocks.keys() & completed:
            raise ValueError("duplicate completed evaluation block across attempts")
        for rows in blocks.values():
            _verify_block(attempt, rows)
            result.rows.extend(
                row.model_copy(
                    update={
                        "trace_path": str(
                            attempt.relative_to(directory) / row.trace_path
                        )
                    }
                )
                for row in rows
            )
        completed.update(blocks)
    order = {seed: index for index, seed in enumerate(original.request.deal_seeds)}
    result.rows.sort(key=lambda row: (order[row.deal_seed], row.leg))
    result.validate_cohort()
    if result.status != "completed":
        result.error = retained.error
    return result


def continue_evaluation(
    directory: Path,
    *,
    execution: EvalExecution | None = None,
    inference: InferenceRuntime | InferenceEndpoint | None = None,
    checkpoint_paths: dict[str, str] | None = None,
    request: EvaluationRequest | None = None,
) -> EvaluationResult:
    """Explicitly execute missing whole blocks with the frozen numerical contract.

    A managed callback may supply its request for admission before reuse. File
    relocation changes no checkpoint identity. This is filesystem continuation,
    not an object-store lease or an automatic retry after machine loss.
    """
    with exclusive_file_lock(directory / ".evaluation.lock"), ExitStack() as claims:
        prior = [directory, *sorted((directory / "attempts").glob("*/"))]
        for attempt in prior:
            claims.enter_context(exclusive_file_lock(attempt / ".execution.lock"))
        result = evaluation_status(directory)
        if result.evaluation_id is None:
            raise ValueError("historical evaluation has no continuation identity")
        original = result.request
        execution = resolve_execution(execution, inference)
        if (execution.compute if execution else None) != (
            result.execution.compute if result.execution else None
        ):
            raise ValueError("continuation requires the same execution contract")
        if request is not None:
            if execution is not None:
                request = request.bind(execution)
            if (
                request.model_copy(
                    update={"checkpoint_paths": original.checkpoint_paths}
                )
                != original
            ):
                raise ValueError("continuation request differs from frozen evaluation")
            checkpoint_paths = request.checkpoint_paths
        paths = (
            checkpoint_paths
            if checkpoint_paths is not None
            else original.checkpoint_paths
        )
        for player in (original.candidate, original.opponent):
            if player.runner_kind == "checkpoint":
                verify_file(
                    Path(paths[player.player_id]),
                    player.checkpoint_sha256,
                    player.checkpoint_bytes,
                )
        complete = set(result.complete_deals())
        missing = tuple(seed for seed in original.deal_seeds if seed not in complete)
        if not missing:
            return result
        # Fence delayed managed RPCs before admitting another attempt. A server
        # already executing holds the claim above, so continuation fails busy.
        for attempt in prior:
            (attempt / ".sealed").touch(exist_ok=True)
        pending = original.model_copy(
            update={"deal_seeds": missing, "checkpoint_paths": paths}
        )
        attempts = directory / "attempts"
        attempts.mkdir(exist_ok=True)
        out = attempts / f"{len(list(attempts.iterdir())) + 1:06d}"
        out.mkdir(exist_ok=False)
        _execute_attempt(
            pending,
            out_dir=out,
            evaluation_id=result.evaluation_id,
            execution=execution,
            inference=inference,
            card_evidence_dir=Path(result.card_evidence_dir)
            if result.card_evidence_dir
            else None,
        )
        return evaluation_status(directory)


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
