"""Killable standalone native inspection, retaining evidence before each step.

Managed Etude evaluations execute and replay in their owning game process.
This resource is for the independent scalar inspection API, not an allocation
or a model server. A child is always terminated and reaped on block exit.
Ordinary games fsync their capture journal before native execution; scenario
Commands are fsynced here before acknowledgement. Custom reducers cross into
the replay child by cloudpickle; in-process replay passes them directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
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
import managym
from managym import AgentError
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


def is_game_error(error: BaseException) -> bool:
    """Whether an error invalidates one game instead of ending the process.

    Only native failures are game outcomes. Policy, configuration and Python
    programming errors must reach the caller. A native panic arrives as
    PyO3's `PanicException`, a `BaseException`.
    """
    return isinstance(error, AgentError) or type(error).__name__ == "PanicException"


def supervise_game(
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    game_seconds: float,
    evidence_path: Path | None = None,
    *,
    worker: str,
    worker_args: tuple[object, ...] = (),
    keep_completed: bool = False,
) -> dict[str, Any]:
    """Retain attempted Commands before native code can hang or kill its worker.

    `worker` is the child's `module:function` target, called with the
    connection, game, registrations, checkpoint paths, Command cap, absolute
    Unix deadline and any additional worker arguments.
    Successful games discard their journal unless `keep_completed`; their
    canonical record holds every decision. Failed ordinary games retain a
    settled canonical snapshot of the journal, including an unknown pending call.
    Scenario roots keep the acknowledged event journal.
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
        process = ChildProcess(
            worker,
            (
                game,
                registrations,
                checkpoint_paths,
                max_commands,
                deadline_unix,
                *worker_args,
            ),
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
            settle_record(game, evidence_path, keep_completed=keep_completed)
        retain("settled", game)
        if evidence_path is not None and game["failure"] is None and not keep_completed:
            evidence_path.unlink()


@dataclass(frozen=True)
class ReplayOutcome:
    receipts: list[dict[str, Any]] | None = None
    card_evidence: tuple[EvidenceRef, ...] = ()
    error: str | None = None
    # Heterogeneous user row types cross this explicit extension boundary.
    statistics_rows: list[list[Any]] = field(default_factory=list)
    analysis_error: str | None = None


@dataclass(frozen=True)
class _ReplayMessage:
    outcome: ReplayOutcome
    rows: bytes | None


def settle_record(
    game: dict[str, Any], evidence_path: Path | None, *, keep_completed: bool
) -> None:
    """Freeze a settled ordinary game's record and its audit view in place.

    A failed game, or any game under `keep_completed`, keeps its canonical
    record at `evidence_path` in place of the raw capture journal.
    """
    finalize_game(game)
    if "record" in game:
        game.pop("pending_decision", None)
        game.update(audit_view(game["record"]))
        if evidence_path is not None and (game.get("failure") or keep_completed):
            atomic_private_json(evidence_path, game["record"])


def verify_game(
    game: dict[str, Any],
    card_evidence_dir: Path | None,
    reduction: Reduction | None = None,
) -> tuple[dict[str, Any], EvidenceRef | None]:
    """Replay one game; with a directory, also retain its card evidence.

    `reduction` sees the same verified traversal.
    """
    if card_evidence_dir is not None:
        evidence = capture_verified_game(
            GameTrace(data=stored_game(game)), reduction=reduction
        )
        return dict(evidence.replay), retain_evidence(card_evidence_dir, evidence)
    if reduction is not None:
        receipt, _ = replay_game(game, reduction=reduction)
        return receipt.to_dict(), None
    return replay_games([game]).to_dict(), None


def replay_worker(
    connection: Connection,
    games: list[dict[str, Any]],
    card_evidence_dir: Path | None,
    reducer_payload: bytes | None = None,
) -> None:
    """Replay child entry; any failure makes the whole outcome an error.

    A reducer failure is an `analysis_error`, never a replay failure.
    """
    try:
        reduction = None
        analysis_error = None
        try:
            if reducer_payload is not None:
                reduction = Reduction(cloudpickle.loads(reducer_payload))
        except Exception as exc:
            analysis_error = f"analysis setup: {type(exc).__name__}: {exc}"
        verified = [verify_game(game, card_evidence_dir, reduction) for game in games]
        outcome = ReplayOutcome(
            [receipt for receipt, _ in verified],
            tuple(ref for _, ref in verified if ref is not None),
            statistics_rows=reduction.rows if reduction is not None else [],
            analysis_error=analysis_error
            or (reduction.error if reduction is not None else None),
        )
        if reducer_payload is None:
            connection.send(outcome)
        else:
            # Keep user-row decoding separate from the completed verification receipt.
            rows = None
            try:
                rows = _reducer_bytes(outcome.statistics_rows)
            except Exception as exc:
                outcome = replace(
                    outcome,
                    analysis_error=f"analysis result transport: {type(exc).__name__}: {exc}",
                )
            connection.send(_ReplayMessage(replace(outcome, statistics_rows=[]), rows))
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
    """Replay native evidence in a killable child within `seconds`."""
    if seconds <= 0:
        return ReplayOutcome(error="allocation deadline exhausted before replay")
    started = time.monotonic()
    transport_error = None
    try:
        payload = _reducer_bytes(tuple(reducers)) if reducers else None
    except Exception as exc:
        payload = None
        transport_error = f"analysis setup transport: {type(exc).__name__}: {exc}"
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
                message = receiver.recv()
            except EOFError:
                return ReplayOutcome(error="replay worker exited without a result")
            outcome = message
            if isinstance(message, _ReplayMessage):
                outcome = message.outcome
                if message.rows is not None:
                    try:
                        outcome = replace(
                            outcome, statistics_rows=cloudpickle.loads(message.rows)
                        )
                    except Exception as exc:
                        outcome = replace(
                            outcome,
                            analysis_error=f"analysis result decoding: {type(exc).__name__}: {exc}",
                        )
            if transport_error is not None:
                outcome = replace(outcome, analysis_error=transport_error)
            return outcome
        return ReplayOutcome(error="replay wall-clock budget exhausted")
    finally:
        process.close()
