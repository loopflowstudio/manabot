"""Torch-free protocol and allocation supervisor for shared evaluation inference.

Every concurrent game in an allocation talks to one serving child
(`manabot.sim.serving`), the only process that loads served weights. The
allocation's `InferenceRuntime` starts that child, admits policies over its
control connection, and hands the service one end of a fresh socket per game
(`connect`). The game child holds the other end as an `InferenceClient`, so
decisions travel game → service → game without passing through the parent;
the parent keeps game lifecycle, deadlines, failures and the Command journal.

`PolicyKey` is the batching key: decisions that seat the same checkpoint under
the same observation/action ABI, world and numerics share one resident policy
and one batch, whatever their matchup. Matchup and content suite are not model
inputs (the card catalog is bound by the checkpoint's world); they remain part
of each registration, its admission and its evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import pickle
import queue
import socket
import struct
import threading
import time
from types import TracebackType
from typing import Any
from uuid import uuid4

import numpy as np

from manabot.infra.config import MatchConfig, ObservationConfig
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.sim.child import ChildProcess
from manabot.sim.execution import EvalExecution
from manabot.sim.performance import EvalPerformance, allocation_cost
from manabot.sim.registration import PlayerRegistration


class InferenceFailure(RuntimeError):
    """Infrastructure failure, never an attributed player forfeit."""


@dataclass(frozen=True)
class PolicyKey:
    checkpoint: str
    input_binding: str
    device: str
    dtype: str
    deterministic: bool
    implementation: str = "policy-inference-cpu-row-sampling-v1"

    @classmethod
    def from_registration(
        cls, player: PlayerRegistration, execution: EvalExecution
    ) -> PolicyKey:
        return cls(
            str(player.checkpoint_sha256),
            canonical_sha256(
                {
                    "observation": player.observation_abi_sha256,
                    "action": player.action_abi_sha256,
                    "world": player.world,
                }
            ),
            execution.device,
            execution.dtype,
            bool(player.player_spec["deterministic"]),
        )


@dataclass(frozen=True)
class DecisionRequest:
    request_id: str
    policy: PolicyKey
    seed: int
    observation: dict[str, np.ndarray]
    deadline_unix: float
    submitted: float


@dataclass(frozen=True)
class DecisionReply:
    """One decision's action; a game connection has one request in flight, so
    the request's unique `request_id` alone binds the reply to its decision."""

    request_id: str
    action: int
    batch_id: str

    def validate(self, request: DecisionRequest) -> None:
        if self.request_id != request.request_id:
            raise InferenceFailure("stale or mismatched inference reply")


@dataclass(frozen=True)
class PolicyAdmission:
    registration: PlayerRegistration
    path: str
    setups: tuple[MatchConfig, ...]


class InferenceClient:
    """A game's direct connection to the serving child, one request in flight.

    Owns `descriptor`, one end of a socket whose other end the runtime handed
    to the service, which reads it as a `multiprocessing` `Connection`. The
    client writes and reads that framing (a 4-byte big-endian length, then a
    pickle) itself, so the whole exchange, including a send to a stopped
    service, is bounded by the request's deadline. Every failure is an
    `InferenceFailure`.
    """

    def __init__(self, descriptor: int) -> None:
        self._socket = socket.socket(fileno=descriptor)

    def infer(self, request: DecisionRequest) -> DecisionReply:
        payload = ForkingPickler.dumps(request)
        try:
            self._socket.settimeout(_remaining(request))
            self._socket.sendall(struct.pack("!i", len(payload)) + payload)
            (size,) = struct.unpack("!i", self._read(4, request))
            reply, error = pickle.loads(self._read(size, request))
        except TimeoutError as exc:
            raise InferenceFailure("inference service request timed out") from exc
        except (OSError, EOFError) as exc:
            raise InferenceFailure(f"inference service exited: {exc}") from exc
        if error:
            raise InferenceFailure(error)
        if not isinstance(reply, DecisionReply):
            raise InferenceFailure("invalid inference reply")
        reply.validate(request)
        if time.time() >= request.deadline_unix:
            raise InferenceFailure("inference reply arrived after its deadline")
        return reply

    def _read(self, size: int, request: DecisionRequest) -> bytearray:
        buffer = bytearray(size)
        view = memoryview(buffer)
        received = 0
        while received < size:
            self._socket.settimeout(_remaining(request))
            count = self._socket.recv_into(view[received:])
            if count == 0:
                raise EOFError("connection closed")
            received += count
        return buffer

    def close(self) -> None:
        self._socket.close()


def _remaining(request: DecisionRequest) -> float:
    remaining = request.deadline_unix - time.time()
    if remaining <= 0:
        raise TimeoutError
    return remaining


class InferenceRuntime:
    """Allocation supervisor, shared by concurrent local Etude callbacks."""

    def __init__(self, execution: EvalExecution, directory: Path) -> None:
        if execution.remaining() <= 0:
            raise ValueError("allocation deadline has expired")
        self.execution = execution
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.identity = uuid4().hex
        self.performance = EvalPerformance(
            self.identity,
            execution.instrumentation,
            latency_boundary="no decisions pass through the allocation supervisor; game worker and serving receipts hold request latency",
        )
        self._slots = threading.BoundedSemaphore(execution.max_active_games)
        self._game_capacity = threading.BoundedSemaphore(execution.max_active_games * 2)
        self._games = ThreadPoolExecutor(max_workers=execution.max_active_games)
        self._lock = threading.RLock()
        self._futures: dict[str, Future[Any]] = {}
        self._error: str | None = None
        self._closed = False
        self._closing = False
        self._write_allocation()
        self._service_started = time.monotonic()
        # Game connections reach the service over this socket as descriptors.
        self._handoff, service_handoff = socket.socketpair()
        self._handoff.settimeout(execution.request_seconds)
        self._handoff_lock = threading.Lock()
        try:
            self.process = ChildProcess(
                "manabot.sim.serving:serve",
                (execution, directory, self.identity, service_handoff.fileno()),
                pass_fds=(service_handoff.fileno(),),
            )
        except BaseException as exc:
            self._handoff.close()
            self._games.shutdown(wait=True, cancel_futures=True)
            self.performance.count("failed_service_starts")
            self.performance.record(
                "failed_service_start", self._service_started, partitions_elapsed=False
            )
            self._write_allocation()
            self._write_execution(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            service_handoff.close()
        self._connection = self.process.connection
        self._reader = threading.Thread(target=self._receive, daemon=True)
        self._reader.start()
        # Only admissions, snapshots and close travel here, so the queue needs
        # no bound. Pipe.send can block while a service is stopped; keep it off
        # the futures lock and caller thread so call deadlines and fail() work.
        self._outgoing: queue.Queue[tuple[str, str, object] | None] = queue.Queue()
        self._sender = threading.Thread(target=self._send, daemon=True)
        self._sender.start()

    def check(self) -> None:
        if (
            self._error
            or self._closing
            or self.execution.remaining() <= 0
            or not self.process.is_alive()
        ):
            raise InferenceFailure(self._error or "inference allocation stopped")

    def _acquire(self, semaphore: threading.BoundedSemaphore) -> None:
        """Wait for `semaphore` while the allocation stays healthy."""
        while not semaphore.acquire(timeout=min(0.1, self.execution.remaining())):
            self.check()
        try:
            self.check()
        except BaseException:
            semaphore.release()
            raise

    def submit_game(self, run: Callable[[], dict[str, Any]]) -> Future[dict[str, Any]]:
        """Bound both threads and queued games across all callbacks."""
        self._acquire(self._game_capacity)
        try:
            future = self._games.submit(run)
        except BaseException:
            self._game_capacity.release()
            raise
        future.add_done_callback(lambda _: self._game_capacity.release())
        return future

    @contextmanager
    def game_slot(self, identity: str | None) -> Iterator[None]:
        """Hold one of `max_active_games` slots, including the wait for it."""
        queued = time.monotonic()
        self._acquire(self._slots)
        try:
            self.performance.record(
                "game_admission", queued, request=identity, partitions_elapsed=False
            )
            yield
        finally:
            self._slots.release()

    def connect(self) -> socket.socket:
        """Open one game's direct connection; the caller owns the returned end.

        The service receives the other end over the handoff socket and answers
        that game's requests on it. A failed handoff fails the allocation.
        """
        self.check()
        client, service = socket.socketpair()
        try:
            with self._handoff_lock:
                socket.send_fds(self._handoff, [b"c"], [service.fileno()])
                # Closing an in-flight socket's last descriptor disconnects it
                # on macOS, so keep ours until the service has its own.
                if self._handoff.recv(1) != b"a":
                    raise OSError("service closed the connection handoff")
        except OSError as exc:
            client.close()
            message = f"inference connection handoff failed: {exc}"
            self.fail(message)
            raise InferenceFailure(message) from exc
        finally:
            service.close()
        self.performance.count("client_connections")
        return client

    def _send(self) -> None:
        try:
            while (message := self._outgoing.get()) is not None:
                self._connection.send(message)
        except (OSError, EOFError) as exc:
            self.fail(f"inference send failed: {exc}")

    def _receive(self) -> None:
        try:
            while True:
                identity, result, error = self._connection.recv()
                if identity == "ready":
                    self.performance.record(
                        "service_start_to_ready",
                        self._service_started,
                        partitions_elapsed=False,
                    )
                    self._write_allocation()
                    continue
                with self._lock:
                    future = self._futures.pop(identity, None)
                if future is not None:
                    if error:
                        future.set_exception(InferenceFailure(error))
                    else:
                        future.set_result(result)
        except (EOFError, OSError) as exc:
            if not self._closing:
                self.fail(f"inference service exited: {exc}")

    def fail(self, message: str) -> None:
        with self._lock:
            self._error = message
            for future in self._futures.values():
                if not future.done():
                    future.set_exception(InferenceFailure(message))
            self._futures.clear()

    def _call(self, kind: str, payload: object) -> Any:
        timeout = min(self.execution.request_seconds, self.execution.remaining())
        if timeout <= 0:
            raise InferenceFailure("inference request deadline exhausted")
        self.check()
        identity = uuid4().hex
        future: Future[Any] = Future()
        with self._lock:
            self._futures[identity] = future
            self._outgoing.put((identity, kind, payload))
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:
            self.fail("inference service request timed out")
            raise InferenceFailure("inference service request timed out") from exc

    def admit(self, admission: PolicyAdmission) -> ObservationConfig:
        started = time.monotonic()
        try:
            result = self._call("admit", admission)
            if not isinstance(result, ObservationConfig):
                raise InferenceFailure("invalid observation admission reply")
            self.performance.count("completed_admissions")
            return result
        except BaseException:
            self.performance.count("failed_admissions")
            raise
        finally:
            # Includes startup when admission is submitted before service ready.
            self.performance.record(
                "admission_round_trip", started, partitions_elapsed=False
            )
            self._write_allocation()

    def snapshot(self) -> dict[str, Any]:
        result = self._call("snapshot", None)
        if not isinstance(result, dict):
            raise InferenceFailure("invalid serving receipt")
        return result

    def __enter__(self) -> InferenceRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        cleanup_started = time.monotonic()
        if self._futures:
            self.fail("allocation closed with pending requests")
        self._games.shutdown(wait=True, cancel_futures=True)
        self._outgoing.put(("close", "close", None))
        self.process.join(timeout=min(5, self.execution.remaining()))
        self.process.kill()
        self.process.join()
        self._reader.join(timeout=1)
        self._outgoing.put(None)
        self._sender.join(timeout=1)
        self._closed = True
        self._connection.close()
        self._handoff.close()
        self.performance.record("cleanup", cleanup_started, partitions_elapsed=False)
        self._write_allocation()

        self._write_execution(self._error)

    def _write_allocation(self) -> None:
        self.performance.write(self.directory / "allocation.json")

    def _write_execution(self, error: str | None) -> None:
        """Retain the same allocation identity and cost boundary on every exit."""
        atomic_json(
            self.directory / "execution.json",
            {
                "allocation_id": self.identity,
                "execution": self.execution.model_dump(),
                "cost": allocation_cost(time.monotonic() - self.performance.started),
                "error": error,
            },
        )
