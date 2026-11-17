"""Killable standalone native inspection, retaining pending Commands before step.

Managed Etude evaluations execute and replay in their owning game process.
This resource is for the independent scalar inspection API, not an allocation
or a model server. A child is always terminated and reaped on block exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
import time
from typing import Any

from manabot.sim.card_evidence import (
    EvidenceRef,
    GameTrace,
    capture_verified_game,
    retain_evidence,
)
from manabot.sim.child import ChildProcess
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import PlayerRegistration
from manabot.sim.replay import apply_game_event, replay_games
from managym.journal import DurableJournal


def failure_record(
    game: dict[str, Any], reason: str, message: str, player_id: str | None
) -> None:
    """Mark the retained game as failed, clearing any apparent winning result."""
    game.setdefault("integrity", {})["execution_failures"] = 1
    game.update(
        termination_reason=reason,
        failure=message,
        failed_player_id=player_id,
        winner=None,
        terminated=False,
        truncated=reason == "truncated",
    )


def is_game_error(error: BaseException) -> bool:
    """Whether an error invalidates one game instead of ending the process.

    A native panic arrives as PyO3's `PanicException`, a `BaseException`.
    """
    return isinstance(error, Exception) or type(error).__name__ == "PanicException"


def supervise_game(
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    game_seconds: float,
    evidence_path: Path | None = None,
    *,
    worker: str,
    keep_completed: bool = False,
) -> dict[str, Any]:
    """Retain attempted Commands before native code can hang or kill its worker.

    `worker` is the child's `module:function` target, called with the
    connection, game, registrations, checkpoint paths, Command cap and the
    game's wall-clock deadline in Unix seconds.
    Successful games discard their journal unless `keep_completed`; their
    returned trace holds every decision. Failed games always keep the journal.
    """
    started = time.monotonic()
    perf = EvalPerformance(
        str(game.get("match_id", "game")), game.get("instrumentation", True)
    )

    journal: DurableJournal | None = None

    def retain(kind: str, value: Any) -> None:
        if journal is not None:
            journal.append([kind, value])

    game["execution_limits"] = {"commands": max_commands, "seconds": game_seconds}
    game["registrations"] = [
        registration.model_dump(mode="json") for registration in registrations
    ]
    # The caller has already clamped the game's allowance to its attempt deadline.
    seconds = game_seconds
    deadline = started + seconds
    deadline_unix = time.time() + seconds
    if evidence_path is not None:
        journal = DurableJournal(evidence_path)
        journal.append(["game", game], create=True)
    process: ChildProcess | None = None
    responsible = None
    try:
        process = ChildProcess(
            worker, (game, registrations, checkpoint_paths, max_commands, deadline_unix)
        )
        receiver = process.connection
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure_record(
                    game, "timeout", "game wall-clock budget exhausted", None
                )
                return game
            if not receiver.poll(min(remaining, 0.1)):
                if process.is_alive():
                    continue
                # Exit may race the timed poll. Drain queued phase/Command
                # events before deciding which player, if any, failed.
                if not receiver.poll():
                    failure_record(
                        game,
                        "crash",
                        f"worker exited {process.exitcode}",
                        responsible,
                    )
                    return game
            try:
                kind, value = receiver.recv()
            except EOFError:
                failure_record(
                    game, "crash", "worker exited without a result", responsible
                )
                return game
            with perf.span("durable_evidence"):
                retain(kind, value)
            game = apply_game_event(game, kind, value)
            if kind == "phase":
                responsible = value
            elif kind == "pending":
                try:
                    receiver.send("retained")
                except (BrokenPipeError, ConnectionResetError):
                    failure_record(
                        game, "crash", "worker exited before acknowledgement", None
                    )
                    return game
            elif kind == "result":
                return game
    finally:
        if process is not None:
            process.close()
        game["execution_seconds"] = time.monotonic() - started
        perf.record("worker_wall", started, request=game.get("match_id"))
        game["supervision_performance"] = perf.snapshot()
        retain("settled", game)
        if evidence_path is not None and game["failure"] is None and not keep_completed:
            evidence_path.unlink()


@dataclass(frozen=True)
class ReplayOutcome:
    receipts: list[dict[str, Any]] | None = None
    card_evidence: tuple[EvidenceRef, ...] = ()
    error: str | None = None


def verify_game(
    game: dict[str, Any], card_evidence_dir: Path | None
) -> tuple[dict[str, Any], EvidenceRef | None]:
    """Replay one game's Commands; with a directory, also retain its card evidence."""
    if card_evidence_dir is None:
        return replay_games([game]).to_dict(), None
    evidence = capture_verified_game(GameTrace(data=game))
    return dict(evidence.replay), retain_evidence(card_evidence_dir, evidence)


def replay_worker(
    connection: Connection, games: list[dict[str, Any]], card_evidence_dir: Path | None
) -> None:
    """Replay child entry; any failure makes the whole outcome an error."""
    try:
        verified = [verify_game(game, card_evidence_dir) for game in games]
        connection.send(
            ReplayOutcome(
                [receipt for receipt, _ in verified],
                tuple(ref for _, ref in verified if ref is not None),
            )
        )
    except BaseException as exc:
        connection.send(ReplayOutcome(error=f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def replay_retained(
    games: list[dict[str, Any]],
    seconds: float,
    *,
    card_evidence_dir: Path | None = None,
) -> ReplayOutcome:
    """Replay native evidence under the same killable allocation boundary."""
    if seconds <= 0:
        return ReplayOutcome(error="allocation deadline exhausted before replay")
    started = time.monotonic()
    process = ChildProcess(
        "manabot.sim.supervision:replay_worker", (games, card_evidence_dir)
    )
    receiver = process.connection
    try:
        while time.monotonic() - started < seconds:
            if not receiver.poll(
                min(0.1, max(0, seconds - (time.monotonic() - started)))
            ):
                if process.is_alive():
                    continue
                if not receiver.poll():
                    return ReplayOutcome(
                        error=f"replay worker exited {process.exitcode}"
                    )
            try:
                return receiver.recv()
            except EOFError:
                return ReplayOutcome(error="replay worker exited without a result")
        return ReplayOutcome(error="replay wall-clock budget exhausted")
    finally:
        process.close()
