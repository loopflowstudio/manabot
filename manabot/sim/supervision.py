"""Killable game supervision shared by served and in-child players.

Ordinary games fsync the capture journal before native execution; scenario
Commands are fsynced here before acknowledgement. Each call owns one
native `ChildProcess` (game or replay); allocation slots bound concurrent
callbacks, including wait time. A served game's child holds its own connection
to the inference service, so decisions never pass through this process; the
child reports an inference failure, which fails the allocation. Torch-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
import io
from multiprocessing.connection import Connection
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import cloudpickle

from manabot.sim.card_evidence import (
    EvidenceRef,
    GameTrace,
    capture_verified_game,
    retain_evidence,
)
from manabot.sim.child import ChildProcess
from manabot.sim.evaluation_records import (
    audit_view,
    bind_record,
    finalize_game,
    stored_game,
)
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import PlayerRegistration
from manabot.sim.replay import apply_game_event, replay_game, replay_games
from manabot.sim.serving_protocol import InferenceFailure, InferenceRuntime
from manabot.sim.threads import REPLAY_BUDGET
import managym
from managym.journal import DurableJournal
from managym.recording import atomic_private_json, read_record
from managym.statistics import Reducer, Reduction


def _native_attribute(kind: str, member: str | None = None) -> object:
    native = getattr(managym, kind)
    return native if member is None else getattr(native, member)


class _ReducerPickler(cloudpickle.CloudPickler):
    """Resolve public PyO3 types/enums by their managym names.

    PyO3 publishes them with ``builtins`` as their module. Ordinary cloudpickle
    therefore tries to copy their native class internals when a script-local
    reducer refers to ActionEnum, instead of importing the shared public type.
    Native observation instances still require ordinary extracted Python facts.
    """

    def reducer_override(self, value: object) -> Any:
        # Pickle's reduction tuples are the local untyped library boundary.
        kind = value if isinstance(value, type) else type(value)
        if (
            kind.__module__ == "builtins"
            and getattr(managym, kind.__name__, None) is kind
        ):
            if isinstance(value, type):
                return _native_attribute, (kind.__name__,)
            for name, member in vars(kind).items():
                if member is value:
                    return _native_attribute, (kind.__name__, name)
        return super().reducer_override(value)


def _reducer_bytes(value: object) -> bytes:
    output = io.BytesIO()
    _ReducerPickler(output).dump(value)
    return output.getvalue()


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


def infrastructure_failure(game: dict[str, Any], message: str) -> dict[str, Any]:
    """Fail the game without blaming a player: the harness, not play, broke."""
    failure_record(game, "crash", message, None)
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
    canonical record holds every decision. Failed ordinary games retain a
    settled canonical snapshot of the journal, including an unknown pending call.
    Scenario roots keep the acknowledged event journal.
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

        journal: DurableJournal | None = None

        def retain(kind: str, value: Any) -> None:
            if journal is not None:
                journal.append([kind, value])

        game["execution_limits"] = {"commands": max_commands, "seconds": game_seconds}
        game["registrations"] = [
            registration.model_dump(mode="json") for registration in registrations
        ]
        # The game's own allowance, clamped to its allocation's deadline.
        seconds = game_seconds
        if inference is not None:
            seconds = min(seconds, inference.execution.remaining())
        deadline = started + seconds
        deadline_unix = time.time() + seconds
        canonical = "scenario_root" not in game
        if canonical and evidence_path is None:
            evidence_path = Path(".runs/game-failures") / f"{uuid4().hex}.jsonl"
        if canonical and evidence_path is not None:
            game["capture_path"] = str(evidence_path)
        if evidence_path is not None and not canonical:
            journal = DurableJournal(evidence_path)
            journal.append(["game", game], create=True)
        process: ChildProcess | None = None
        responsible = None
        try:
            client = None
            if inference is not None:
                try:
                    client = inference.connect()
                except InferenceFailure as exc:
                    return infrastructure_failure(game, str(exc))
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
                remaining = deadline - time.monotonic()
                if inference is not None:
                    try:
                        inference.check()
                    except InferenceFailure as exc:
                        return infrastructure_failure(game, str(exc))
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
                if kind == "inference_failure":
                    # A failed or late reply means the shared service cannot be
                    # trusted for any game; stop the allocation, as for a crash.
                    if inference is not None:
                        inference.fail(value)
                    return infrastructure_failure(game, value)
                with perf.span("durable_evidence"):
                    retain(kind, value)
                game = apply_game_event(game, kind, value)
                if kind == "phase":
                    responsible = value
                elif kind == "pending" and not canonical:
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
            if canonical:
                if (
                    "record" not in game
                    and evidence_path is not None
                    and evidence_path.exists()
                ):
                    bind_record(game, read_record(evidence_path))
                finalize_game(game)
                if "record" in game:
                    game.pop("pending_decision", None)
                    game.update(audit_view(game["record"]))
                    if evidence_path is not None and (
                        game.get("failure") or keep_completed
                    ):
                        atomic_private_json(evidence_path, game["record"])
            retain("settled", game)
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
    # Heterogeneous user row types cross this explicit extension boundary.
    statistics_rows: list[list[Any]] = field(default_factory=list)


def replay_worker(
    connection: Connection,
    games: list[dict[str, Any]],
    card_evidence_dir: Path | None,
    reducer_payload: bytes | None = None,
) -> None:
    """Replay child entry; over its thread budget, the outcome is an error."""
    try:
        reduction = (
            Reduction(cloudpickle.loads(reducer_payload))
            if reducer_payload is not None
            else None
        )
        if card_evidence_dir is None and reduction is None:
            outcome = ReplayOutcome([replay_games([game]).to_dict() for game in games])
        else:
            receipts = []
            references = []
            for game in games:
                if card_evidence_dir is not None:
                    evidence = capture_verified_game(
                        GameTrace(data=stored_game(game)), reduction=reduction
                    )
                    references.append(retain_evidence(card_evidence_dir, evidence))
                    receipts.append(dict(evidence.replay))
                else:
                    receipt, _ = replay_game(game, reduction=reduction)
                    receipts.append(receipt.to_dict())
            outcome = ReplayOutcome(
                receipts,
                tuple(references),
                statistics_rows=reduction.rows if reduction is not None else [],
            )
        REPLAY_BUDGET.account(outcome.process)
        # A user may define reducers/row dataclasses in their script. Cloudpickle
        # keeps that code portable without re-importing the parent's __main__.
        if reducer_payload is None:
            connection.send(outcome)
        else:
            connection.send_bytes(_reducer_bytes(outcome))
    except BaseException as exc:
        connection.send(ReplayOutcome(error=f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def replay_retained(
    games: list[dict[str, Any]],
    seconds: float,
    *,
    card_evidence_dir: Path | None = None,
    reducers: Sequence[Reducer[Any]] = (),
) -> ReplayOutcome:
    """Replay native evidence under the same killable allocation boundary."""
    if seconds <= 0:
        return ReplayOutcome(error="allocation deadline exhausted before replay")
    started = time.monotonic()
    payload = _reducer_bytes(tuple(reducers)) if reducers else None
    process = ChildProcess(
        "manabot.sim.supervision:replay_worker", (games, card_evidence_dir, payload)
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
