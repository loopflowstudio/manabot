"""Killable game supervision shared by served and in-child players.

The pending Command is fsynced before acknowledgement. Each call owns one
native `ChildProcess` (game or replay); allocation slots bound concurrent
callbacks, including wait time. A served game's child holds its own connection
to the inference service, so decisions never pass through this process; the
child reports an inference failure, which fails the allocation. Torch-free.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import time
from typing import IO, Any

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
from manabot.sim.serving_protocol import InferenceFailure, InferenceRuntime
from manabot.sim.threads import REPLAY_BUDGET


def _failure(
    game: dict[str, Any], reason: str, message: str, player_id: str | None
) -> None:
    game.setdefault("integrity", {})["execution_failures"] = 1
    game.update(
        termination_reason=reason,
        failure=message,
        failed_player_id=player_id,
        winner=None,
        terminated=False,
        truncated=reason == "truncated",
    )


def _infrastructure_failure(game: dict[str, Any], message: str) -> dict[str, Any]:
    """Fail the game without blaming a player: the harness, not play, broke."""
    _failure(game, "crash", message, None)
    game["infrastructure_failure"] = True
    return game


def supervise_game(
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    game_seconds: float,
    evidence_path: Path | None = None,
    *,
    worker: str,
    inference: InferenceRuntime | None = None,
    keep_completed: bool = False,
) -> dict[str, Any]:
    """Retain attempted Commands before native code can hang or kill its worker.

    `worker` is the child's `module:function` target, called with the
    connection, game, registrations, checkpoint paths, Command cap, the
    descriptor of its inference connection (or None) and the game's wall-clock
    deadline in Unix seconds.
    Successful games discard their journal unless `keep_completed`; their
    returned trace holds every decision. Failed games always keep the journal.
    """
    slot = (
        nullcontext()
        if inference is None
        else inference.game_slot(game.get("match_id"))
    )
    with slot:
        started = time.monotonic()
        perf = EvalPerformance(
            str(game.get("match_id", "game")), game.get("instrumentation", True)
        )

        journal: IO[str] | None = None

        def retain(kind: str, value: Any) -> None:
            if journal is not None:
                journal.write(json.dumps([kind, value], allow_nan=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())

        game["execution_limits"] = {"commands": max_commands, "seconds": game_seconds}
        game["registrations"] = [
            registration.model_dump(mode="json") for registration in registrations
        ]
        deadline_unix = time.time() + game_seconds
        if inference is not None:
            deadline_unix = min(deadline_unix, inference.execution.deadline_unix)
        if evidence_path is not None:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.touch(mode=0o600, exist_ok=False)
            journal = evidence_path.open("a")
            retain("game", game)
        process: ChildProcess | None = None
        responsible = None
        try:
            client = None
            if inference is not None:
                try:
                    client = inference.connect()
                except InferenceFailure as exc:
                    return _infrastructure_failure(game, str(exc))
            try:
                descriptor = None if client is None else client.fileno()
                process = ChildProcess(
                    worker,
                    (
                        game,
                        registrations,
                        checkpoint_paths,
                        max_commands,
                        descriptor,
                        deadline_unix,
                    ),
                    pass_fds=() if descriptor is None else (descriptor,),
                )
            finally:
                # The child holds its own copy; the service answers only the child.
                if client is not None:
                    client.close()
            receiver = process.connection
            while True:
                remaining = game_seconds - (time.monotonic() - started)
                if inference is not None:
                    remaining = min(remaining, inference.execution.remaining())
                    try:
                        inference.check()
                    except InferenceFailure as exc:
                        return _infrastructure_failure(game, str(exc))
                if remaining <= 0:
                    _failure(game, "timeout", "game wall-clock budget exhausted", None)
                    return game
                if not receiver.poll(min(remaining, 0.1)):
                    if process.is_alive():
                        continue
                    # Exit may race the timed poll. Drain queued phase/Command
                    # events before deciding which player, if any, failed.
                    if not receiver.poll():
                        _failure(
                            game,
                            "crash",
                            f"worker exited {process.exitcode}",
                            responsible,
                        )
                        return game
                try:
                    kind, value = receiver.recv()
                except EOFError:
                    _failure(
                        game, "crash", "worker exited without a result", responsible
                    )
                    return game
                if kind == "inference_failure":
                    # A failed or late reply means the shared service cannot be
                    # trusted for any game; stop the allocation, as for a crash.
                    if inference is not None:
                        inference.fail(value)
                    return _infrastructure_failure(game, value)
                with perf.span("durable_evidence"):
                    retain(kind, value)
                game = apply_game_event(game, kind, value)
                if kind == "phase":
                    responsible = value
                elif kind == "pending":
                    try:
                        receiver.send("retained")
                    except (BrokenPipeError, ConnectionResetError):
                        _failure(
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
            if journal is not None:
                journal.close()
            if (
                evidence_path is not None
                and game["failure"] is None
                and not keep_completed
            ):
                evidence_path.unlink()


@dataclass(frozen=True)
class ReplayOutcome:
    receipts: list[dict[str, Any]] | None = None
    card_evidence: tuple[EvidenceRef, ...] = ()
    error: str | None = None
    # The replay child's `ProcessCensus` counters, when it reported.
    process: dict[str, int] = field(default_factory=dict)


def replay_worker(
    connection: Connection, games: list[dict[str, Any]], card_evidence_dir: Path | None
) -> None:
    """Replay child entry; over its thread budget, the outcome is an error."""
    try:
        if card_evidence_dir is None:
            outcome = ReplayOutcome([replay_games([game]).to_dict() for game in games])
        else:
            receipts = []
            references = []
            for game in games:
                evidence = capture_verified_game(GameTrace(data=game))
                references.append(retain_evidence(card_evidence_dir, evidence))
                receipts.append(dict(evidence.replay))
            outcome = ReplayOutcome(receipts, tuple(references))
        REPLAY_BUDGET.account(outcome.process)
        connection.send(outcome)
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
