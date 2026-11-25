"""Placement-independent evaluation requests, execution and retained results.

An EvaluationRequest names registrations, deals and arena semantics, without a
training run. evaluate uses supplied game agents for managed Etude play;
evaluate_local bounds scalar CPU children without remote services.
Only complete replayed deal cohorts produce paired-deal score intervals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import fields
from itertools import batched
import json
import math
from pathlib import Path
import time
from typing import Literal
from uuid import uuid4

from etude.errors import RetryRequested
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from manabot.eval.behaviour import Play, PlayCounts
from manabot.eval.match import MatchupMode, play_cell
from manabot.eval.models import EvaluationKey
from manabot.eval.results import EvaluationRow, RateInterval, bootstrap_mean_interval
from manabot.infra.artifacts import verify_file
from manabot.infra.files import atomic_json, exclusive_file_lock
from manabot.infra.identity import StrictModel, canonical_sha256, file_sha256
from manabot.policy.protocol import BackendName
from manabot.sim.agent import GamePlayer
from manabot.sim.execution import EvalExecutionV1, RecordedExecution
from manabot.sim.matchup import AuthoredMatchup, DeckMatchup, selected_decks
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import HISTORICAL_MIRROR_SUITE, PlayerRegistration


class EvaluationRequest(StrictModel):
    key: EvaluationKey
    candidate: PlayerRegistration
    opponent: PlayerRegistration
    checkpoint_paths: dict[str, str] = Field(default_factory=dict)
    deal_seeds: tuple[int, ...]
    purpose: str = Field(default="local-check-not-admission", min_length=1)
    matchup_mode: MatchupMode = "cross"
    authored_matchup: AuthoredMatchup | DeckMatchup | None = None
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
        return range(2 if self.key.content_suite == HISTORICAL_MIRROR_SUITE else 4)

    @property
    def expected_games(self) -> int:
        return len(self.deal_seeds) * len(self.legs)


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    request: EvaluationRequest
    evaluation_id: str | None = None
    deadline_unix: float | None = Field(default=None, exclude_if=lambda v: v is None)
    card_evidence_dir: str | None = None
    # Retained v1 allocations stay readable; they are never continued under v2.
    execution: RecordedExecution | EvalExecutionV1 | None = None
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
            setattr(
                self,
                name,
                bootstrap_mean_interval(values.mean(axis=1), indices),
            )


def _execute_attempt(
    request: EvaluationRequest,
    *,
    out_dir: Path,
    evaluation_id: str,
    backend_name: BackendName = "torch",
    agents: Mapping[str, GamePlayer] | None = None,
    deadline: float | None = None,
    card_evidence_dir: Path | None = None,
) -> EvaluationResult:
    """Execute in an exclusively claimed directory; retain every failure."""
    result = EvaluationResult(
        request=request,
        deadline_unix=deadline,
        evaluation_id=evaluation_id,
        card_evidence_dir=str(card_evidence_dir)
        if card_evidence_dir is not None
        else None,
    )
    atomic_json(out_dir / "evaluation.json", result.model_dump(mode="json"))
    started = time.perf_counter()
    performance = EvalPerformance("evaluation")
    try:
        rows, trace, replay = play_cell(
            agents=agents,
            deadline=deadline,
            backend_name=backend_name,
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
            reducers={"play": Play()},
        )
        for name, payload in (("rows", rows), ("trace", trace), ("replay", replay)):
            atomic_json(out_dir / f"{name}.json", payload)
        result.rows = [EvaluationRow.model_validate(row) for row in rows]
    except RetryRequested:
        raise
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
    backend_name: BackendName = "torch",
    agents: Mapping[str, GamePlayer] | None = None,
    deadline: float | None = None,
    card_evidence_dir: Path | None = None,
) -> EvaluationResult:
    """Start a fresh logical evaluation; existing directories are never replaced.

    Explicit continuation uses continue_evaluation. The first attempt stays at
    this root; subsequent attempts retain their own subdirectories and receipts.
    """
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("deadline must be finite")
    out_dir.mkdir(parents=True, exist_ok=False)
    with exclusive_file_lock(out_dir / ".evaluation.lock"):
        return _execute_attempt(
            request,
            out_dir=out_dir,
            evaluation_id=uuid4().hex,
            backend_name=backend_name,
            agents=agents,
            deadline=deadline,
            card_evidence_dir=card_evidence_dir,
        )


def attempt_directories(directory: Path) -> list[Path]:
    """The original attempt lives at the root and retries in numbered directories."""
    return [directory, *sorted((directory / "attempts").glob("*/"))]


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


type GamePlayCounts = dict[tuple[int, int], tuple[PlayCounts, PlayCounts]]


def attempt_play_counts(
    directory: Path, rows: Sequence[EvaluationRow]
) -> GamePlayCounts | None:
    """Join the reducer's native-seat pairs to completed games in replay order."""
    path = directory / "statistics.json"
    try:
        statistics = json.loads(path.read_bytes())
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(statistics, dict) or statistics.get("analysis_error"):
        return None
    tables = statistics.get("tables")
    if not isinstance(tables, dict):
        return None
    table = tables.get("play")
    completed = [
        r for r in rows if r.failure is None and r.replay_passed and r.terminated
    ]
    # Construction defaults initialize the live counter. An absent saved count
    # is unavailable and must never be interpreted as a measured zero.
    columns = {field.name for field in fields(PlayCounts)}
    historical = columns - {"attackers_committed", "blockers_committed"}
    if not isinstance(table, list) or any(
        not isinstance(item, dict)
        or (item.keys() & columns) not in (historical, columns)
        for item in table
    ):
        return None
    try:
        counts = TypeAdapter(list[PlayCounts]).validate_python(table)
    except ValidationError:
        return None
    if len(counts) != 2 * len(completed):
        return None
    found: GamePlayCounts = {}
    for row, (first, second) in zip(completed, batched(counts, 2), strict=True):
        if (first.seat, second.seat) != (0, 1):
            return None
        if row.valid:
            found[row.deal_seed, row.leg] = (first, second)
    return found


def evaluation_play_counts(directory: Path) -> GamePlayCounts | None:
    """Read behaviour from the attempts that supplied the accepted deal blocks."""
    found: GamePlayCounts = {}
    for attempt in attempt_directories(directory):
        if not (attempt / "evaluation.json").is_file():
            continue
        retained = _retained_result(attempt)
        deals = set(retained.complete_deals())
        if not deals:
            continue
        counts = attempt_play_counts(attempt, retained.rows)
        if counts is None:
            return None
        accepted = {key: pair for key, pair in counts.items() if key[0] in deals}
        if found.keys() & accepted.keys():
            raise ValueError("duplicate completed evaluation block across attempts")
        found.update(accepted)
    return found


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
    for attempt in attempt_directories(directory):
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
    backend_name: BackendName = "torch",
    agents: Mapping[str, GamePlayer] | None = None,
    deadline: float | None = None,
    checkpoint_paths: dict[str, str] | None = None,
    request: EvaluationRequest | None = None,
) -> EvaluationResult:
    """Explicitly execute missing whole blocks with the frozen numerical contract.

    A managed callback may supply its request for admission before reuse. File
    relocation changes no checkpoint identity. This is filesystem continuation,
    not an object-store lease or an automatic retry after machine loss.
    """
    with exclusive_file_lock(directory / ".evaluation.lock"), ExitStack() as claims:
        prior = attempt_directories(directory)
        for attempt in prior:
            claims.enter_context(exclusive_file_lock(attempt / ".execution.lock"))
        result = evaluation_status(directory)
        if result.evaluation_id is None:
            raise ValueError("historical evaluation has no continuation identity")
        original = result.request
        if result.execution is not None:
            raise ValueError(
                "retained shared evaluation requires its recorded revision"
            )
        if request is not None:
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
            backend_name=backend_name,
            agents=agents,
            deadline=deadline,
            card_evidence_dir=Path(result.card_evidence_dir)
            if result.card_evidence_dir
            else None,
        )
        return evaluation_status(directory)


def evaluate_local(
    request: EvaluationRequest,
    *,
    deadline: float,
    out_dir: Path,
    backend_name: BackendName = "torch",
) -> EvaluationResult:
    """Run a scalar local comparison under one absolute wall-clock deadline."""
    return evaluate(
        request, out_dir=out_dir, deadline=deadline, backend_name=backend_name
    )
