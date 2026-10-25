"""Killable game supervision shared by scalar and resident-inference execution.

The pending Command is fsynced before acknowledgement. Each call owns one
native child; allocation slots bound concurrent callbacks, including wait time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
from pathlib import Path
import time
from typing import Any

from manabot.sim.card_evidence import (
    EvidenceRef,
    GameTrace,
    capture_verified_game,
    retain_evidence,
)
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import PlayerRegistration
from manabot.sim.replay import apply_game_event, replay_games
from manabot.sim.serving import InferenceFailure, InferenceRuntime


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


@contextmanager
def _game_slot(
    inference: InferenceRuntime | None, identity: str | None
) -> Iterator[None]:
    queued = time.monotonic()
    if inference is not None:
        while not inference.slots.acquire(
            timeout=min(0.1, inference.execution.remaining())
        ):
            inference.check()
        try:
            inference.check()
        except BaseException:
            inference.slots.release()
            raise
        inference.performance.record("game_admission", queued, request=identity)
    try:
        yield
    finally:
        if inference is not None:
            inference.slots.release()


def supervise_game(
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    game_seconds: float,
    evidence_path: Path | None = None,
    *,
    worker: Callable[..., None],
    inference: InferenceRuntime | None = None,
) -> dict[str, Any]:
    """Retain attempted Commands before native code can hang or kill its worker."""
    with _game_slot(inference, game.get("match_id")):
        started = time.monotonic()
        perf = EvalPerformance(
            str(game.get("match_id", "game")), game.get("instrumentation", True)
        )

        def retain(kind: str, value: Any) -> None:
            if evidence_path is not None:
                with evidence_path.open("a") as evidence:
                    evidence.write(json.dumps([kind, value], allow_nan=False) + "\n")
                    evidence.flush()
                    os.fsync(evidence.fileno())

        game["execution_limits"] = {"commands": max_commands, "seconds": game_seconds}
        game["registrations"] = [
            registration.model_dump(mode="json") for registration in registrations
        ]
        if evidence_path is not None:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.touch(mode=0o600, exist_ok=False)
            retain("game", game)
        context = mp.get_context("spawn")
        receiver, sender = context.Pipe(duplex=True)
        process = context.Process(
            target=worker,
            args=(
                sender,
                game,
                registrations,
                checkpoint_paths,
                max_commands,
            ),
        )
        responsible = None
        try:
            process.start()
            sender.close()
            while True:
                remaining = game_seconds - (time.monotonic() - started)
                if inference is not None:
                    remaining = min(remaining, inference.execution.remaining())
                    try:
                        inference.check()
                    except InferenceFailure as exc:
                        _failure(game, "crash", str(exc), None)
                        game["infrastructure_failure"] = True
                        return game
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
                if kind == "inference":
                    if inference is None:
                        raise InferenceFailure("no inference runtime attached")
                    # Inference waiting consumes the active game's wall allowance.
                    value = replace(
                        value,
                        deadline_unix=min(
                            value.deadline_unix,
                            time.time()
                            + max(0, game_seconds - (time.monotonic() - started)),
                        ),
                    )
                    try:
                        reply = inference.infer(value)
                        receiver.send(reply)
                    except InferenceFailure as exc:
                        _failure(game, "crash", str(exc), None)
                        game["infrastructure_failure"] = True
                        return game
                    continue
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
            if process.is_alive():
                process.kill()
            if process.pid is not None:
                process.join()
            sender.close()
            receiver.close()
            game["execution_seconds"] = time.monotonic() - started
            perf.record("worker_wall", started, request=game.get("match_id"))
            game["supervision_performance"] = perf.snapshot()
            retain("settled", game)


@dataclass(frozen=True)
class ReplayOutcome:
    receipts: list[dict[str, Any]] | None = None
    card_evidence: tuple[EvidenceRef, ...] = ()
    error: str | None = None


def _replay_worker(
    connection: Connection, games: list[dict[str, Any]], card_evidence_dir: Path | None
) -> None:
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
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe()
    process = context.Process(
        target=_replay_worker, args=(sender, games, card_evidence_dir)
    )
    started = time.monotonic()
    try:
        process.start()
        sender.close()
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
        if process.is_alive():
            process.kill()
        if process.pid is not None:
            process.join()
        sender.close()
        receiver.close()
