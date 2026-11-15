"""Durable reward schedules and supervised scalar game/opening samples.

A dispatched slot always settles; unstarted slots stay missing. Whole bundles
must succeed before learning. Complete-game counts and successful opening
observations are separate, although both settle the game-error allowance.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from manabot.eval.admission import register_checkpoint
from manabot.infra.config import MatchConfig
from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel, canonical_sha256, file_sha256
from manabot.policy.protocol import Backend
from manabot.sim.match import play_game
from manabot.sim.supervision import replay_retained
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLedger,
    GameErrorPolicy,
    GameOutcome,
)
import managym


class RewardSample(StrictModel):
    id: str
    bundle: str
    mode: Literal["game", "value"]
    seed: int
    seat: Literal[0, 1]
    match: MatchConfig


RewardStatus = Literal[
    "completed",
    "scored",
    "unavailable",
    "engine_error",
    "truncated",
    "action_limit",
    "deadline",
    "infrastructure",
]


class RewardResult(StrictModel):
    id: str
    status: RewardStatus
    value: float | None = None
    opening: float | None = None
    commands: int = 0
    wall_seconds: float = 0
    evidence: str
    bundle: str
    game_seat: Literal[0, 1]
    deal_seed: int
    worker_cpu_seconds: float | None = None


@dataclass(frozen=True)
class RewardBatch:
    results: tuple[RewardResult, ...]
    rewards: dict[str, float | None]
    stop_reason: str | None = None
    timed_out: bool = False

    @property
    def stopped(self) -> bool:
        return self.stop_reason is not None


def _append(path: Path, row: dict[str, object]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def settle(
    sample: RewardSample, game: dict[str, Any], evidence: Path, ledger: GameErrorLedger
) -> RewardResult:
    """Narrow the worker wire boundary once; failures never become outcomes."""
    failure = game.get("failure")
    reason = game.get("termination_reason")
    opening = game.get("opening_value")
    value: float | None = None
    status: RewardStatus
    if game.get("infrastructure_failure"):
        status = "infrastructure"
    elif any(
        game.get("integrity", {}).get(key, 0)
        for key in ("private_exposures", "root_mutations", "illegal_actions")
    ):
        status = "infrastructure"
    elif game.get("truncated") or game.get("integrity", {}).get("truncations", 0):
        status = "truncated"
    elif failure:
        status = {
            "truncated": "truncated",
            "command_cap": "action_limit",
            "timeout": "deadline",
        }.get(reason, "infrastructure")
        if reason == "crash" and game.get("error_class") in (
            "AgentError",
            "PyAgentError",
            "NativePanic",
            "PanicException",
        ):
            status = "engine_error"
    elif (
        sample.mode == "value"
        and reason == "opening_scored"
        and not game.get("terminated")
    ):
        status, value = "scored", opening
    elif game.get("terminated") and sample.mode == "game":
        winner = game.get("winner")
        if winner not in (None, 0, 1):
            raise ValueError("invalid terminal winner")
        status, value = (
            "completed",
            0.0 if winner is None else (1.0 if winner == sample.seat else -1.0),
        )
    elif game.get("terminated") and sample.mode == "value" and opening is None:
        status = "unavailable"
    else:
        status = "infrastructure"
    if value is not None and (
        not isinstance(value, (float, int)) or not math.isfinite(value)
    ):
        status, value = "infrastructure", None
    if status == "scored" and value is None:
        status = "infrastructure"
    if opening is not None and (
        not isinstance(opening, (float, int)) or not math.isfinite(opening)
    ):
        status, value, opening = "infrastructure", None, None
    outcome: GameOutcome = {
        "completed": "completed",
        "scored": "opening_scored",
        "unavailable": "abandoned",
        "infrastructure": "abandoned",
        "engine_error": "engine_error",
        "truncated": "truncated",
        "action_limit": "action_limit",
        "deadline": "deadline",
    }[status]
    decisions = game.get("decisions", [])
    ledger.record(
        GameAttempt(
            stream=0,
            seed=sample.seed,
            seat=sample.seat,
            matchup=sample.match,
            outcome=outcome,
            actions=[int(row["chosen_offer"]["id"]) for row in decisions],
            error_class=game.get("error_class"),
            error=game.get("failure"),
            valid_samples=int(value is not None),
            discarded_samples=int(value is None),
            seconds=game.get("execution_seconds", 0),
        ),
        {
            "sample": sample.id,
            "mode": sample.mode,
            "status": status,
            "journal": str(evidence),
        },
    )
    return RewardResult(
        id=sample.id,
        bundle=sample.bundle,
        game_seat=sample.seat,
        deal_seed=sample.seed,
        status=status,
        value=value,
        opening=opening,
        commands=len(decisions),
        wall_seconds=game.get("execution_seconds", 0),
        evidence=str(evidence),
        worker_cpu_seconds=game.get("worker_performance", {}).get(
            "process_cpu_seconds"
        ),
    )


def score_schedule(
    samples: list[RewardSample],
    out: Path,
    run: Callable[[RewardSample, Path], dict[str, Any]],
    *,
    check: Callable[[], None],
    policy: GameErrorPolicy = GameErrorPolicy(),
    ledger: GameErrorLedger | None = None,
) -> RewardBatch:
    """Retain a schedule before dispatch; do not retry or replenish failed slots."""
    if len({s.id for s in samples}) != len(samples):
        raise ValueError("reward ids must be unique")
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "schedule.json", [s.model_dump(mode="json") for s in samples])
    ledger = ledger or GameErrorLedger(out / "game-errors.jsonl", policy)
    results: list[RewardResult] = []
    stop_reason = None
    timed_out = False
    for sample in samples:
        try:
            check()
        except Exception as error:
            timed_out = isinstance(error, TimeoutError)
            stop_reason = f"{type(error).__name__}: {error}"
            break
        journal = out / f"{len(results):05d}.journal.jsonl"
        _append(
            out / "events.jsonl",
            {"event": "dispatch", "id": sample.id, "journal": str(journal)},
        )
        try:
            game = run(sample, journal)
        except Exception as error:
            game = {
                "failure": f"{type(error).__name__}: {error}",
                "infrastructure_failure": not isinstance(error, TimeoutError),
                "termination_reason": "timeout"
                if isinstance(error, TimeoutError)
                else "crash",
            }
        atomic_json(out / f"{len(results):05d}.game.json", game)
        result = settle(sample, game, journal, ledger)
        results.append(result)
        _append(
            out / "events.jsonl", {"event": "settled", **result.model_dump(mode="json")}
        )
        if result.status in ("infrastructure", "deadline") or ledger.state.stop_reason:
            timed_out = result.status == "deadline"
            stop_reason = f"sample {sample.id}: {result.status}: {game.get('failure') or ledger.state.stop_reason}"
            break
    rewards = {}
    for bundle in dict.fromkeys(s.bundle for s in samples):
        ids = [s.id for s in samples if s.bundle == bundle]
        rows = [r for r in results if r.id in ids]
        rewards[bundle] = (
            sum(r.value for r in rows if r.value is not None) / len(ids)
            if len(rows) == len(ids) and all(r.value is not None for r in rows)
            else None
        )
    batch = RewardBatch(tuple(results), rewards, stop_reason, timed_out)
    atomic_json(
        out / "receipt.json",
        {
            "planned": len(samples),
            "attempted": len(results),
            "missing": len(samples) - len(results),
            "completed_games": sum(r.status == "completed" for r in results),
            "opening_scores": sum(r.status == "scored" for r in results),
            "failed_or_unavailable": sum(r.value is None for r in results),
            "stopped": batch.stopped,
            "stop_reason": stop_reason,
            "timed_out": timed_out,
            "rewards": rewards,
        },
    )
    return batch


class FrozenRewardRunner:
    """Admit player bytes and setups through maintained registration and worker."""

    def __init__(
        self,
        path: Path,
        backend: Backend,
        digest: str,
        deadline: float,
        *,
        game_seconds: float = 60,
        max_commands: int = 4000,
    ) -> None:
        self.path = path
        self.backend = backend
        self.digest = digest
        self.deadline = deadline
        self.game_seconds = game_seconds
        self.max_commands = max_commands

    def __call__(self, sample: RewardSample, journal: Path) -> dict[str, Any]:
        if file_sha256(self.path) != self.digest:
            raise ValueError("frozen player bytes changed")
        registrations = [
            register_checkpoint(
                self.path,
                backend=self.backend,
                match=sample.match,
                content_suite="rotisserie",
                player_id=f"pilot-{seat}",
                display_name="Frozen player",
                expected_sha256=self.digest,
            ).registration
            for seat in range(2)
        ]
        seconds = min(self.game_seconds, self.deadline - perf_counter())
        if seconds <= 0:
            return {"failure": "run deadline", "termination_reason": "timeout"}
        game = {
            "match_id": sample.id,
            "deal_seed": sample.seed,
            "seat_players": [r.player_id for r in registrations],
            "player_seeds": {
                r.player_id: sample.seed + 100000 + i
                for i, r in enumerate(registrations)
            },
            "match_hypers": sample.match.model_dump(mode="json"),
            "decisions": [],
            "winner": None,
            "terminated": False,
            "truncated": False,
            "failure": None,
            "integrity": {
                "private_exposures": 0,
                "root_mutations": 0,
                "illegal_actions": 0,
                "truncations": 0,
            },
            "opening_score_seat": sample.seat,
            "stop_after_opening": sample.mode == "value",
            "world": managym.world_identity(),
        }
        result = play_game(
            game,
            registrations,
            {r.player_id: str(self.path) for r in registrations},
            self.max_commands,
            seconds,
            journal,
            keep_completed=True,
        )

        if (
            result.get("terminated")
            and not result.get("failure")
            and sample.mode == "game"
        ):
            result["game_trace_sha256"] = canonical_sha256(result)
            trace_path = journal.with_suffix(".trace.json")
            atomic_json(trace_path, result)
            replay = replay_retained(
                [result], min(60, max(0, self.deadline - perf_counter()))
            )
            result["replay"] = {
                **asdict(replay),
                "trace": str(trace_path),
                "sha256": file_sha256(trace_path),
            }
            if (
                replay.error
                or not replay.receipts
                or not all(row["passed"] for row in replay.receipts)
            ):
                replay_deadline = replay.error in (
                    "allocation deadline exhausted before replay",
                    "replay wall-clock budget exhausted",
                )
                result["infrastructure_failure"] = not replay_deadline
                if replay_deadline:
                    result["termination_reason"] = "timeout"
                result["failure"] = replay.error or "completed-game replay diverged"
            result.pop("game_trace_sha256")
            result["game_trace_sha256"] = canonical_sha256(result)
        return result
