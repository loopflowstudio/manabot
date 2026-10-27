"""Torch-free protocol and parent supervisor for shared evaluation inference.

Game children build `DecisionRequest`s and validate `DecisionReply`s; the
parent's `InferenceRuntime` multiplexes them, from every concurrent game in an
allocation, to one serving child (`manabot.sim.serving`), the only process that
loads served weights.

`PolicyKey` is the batching key: decisions that seat the same checkpoint under
the same observation/action ABI, world and numerics share one resident policy
and one batch, whatever their matchup. Matchup and content suite are not model
inputs (the card catalog is bound by the checkpoint's world); they remain part
of each registration, its admission and its evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
import queue
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
    game_id: str
    actor: int
    revision: int
    policy: PolicyKey
    ordinal: int
    seed: int
    observation: dict[str, np.ndarray]
    deadline_unix: float
    submitted: float


@dataclass(frozen=True)
class DecisionReply:
    request_id: str
    game_id: str
    actor: int
    revision: int
    ordinal: int
    action: int
    batch_id: str

    def validate(self, request: DecisionRequest) -> None:
        if (self.request_id, self.game_id, self.actor, self.revision, self.ordinal) != (
            request.request_id,
            request.game_id,
            request.actor,
            request.revision,
            request.ordinal,
        ):
            raise InferenceFailure("stale or mismatched inference reply")


@dataclass(frozen=True)
class PolicyAdmission:
    registration: PlayerRegistration
    path: str
    setups: tuple[MatchConfig, ...]


class InferenceRuntime:
    """Allocation supervisor, shared by concurrent local Etude callbacks."""

    def __init__(self, execution: EvalExecution, directory: Path) -> None:
        if execution.remaining() <= 0:
            raise ValueError("allocation deadline has expired")
        self.execution = execution
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.identity = uuid4().hex
        self.performance = EvalPerformance(self.identity, execution.instrumentation)
        self.slots = threading.BoundedSemaphore(execution.max_active_games)
        self._game_capacity = threading.BoundedSemaphore(execution.max_active_games * 2)
        self._games = ThreadPoolExecutor(max_workers=execution.max_active_games)
        self._capacity = threading.BoundedSemaphore(execution.max_active_games + 1)
        self._lock = threading.RLock()
        self._futures: dict[str, Future[Any]] = {}
        self._error: str | None = None
        self._closed = False
        self._closing = False
        self.process = ChildProcess(
            "manabot.sim.serving:serve", (execution, directory, self.identity)
        )
        self._connection = self.process.connection
        self._reader = threading.Thread(target=self._receive, daemon=True)
        self._reader.start()
        self._outgoing: queue.Queue[tuple[str, str, object] | None] = queue.Queue(
            execution.max_active_games + 2
        )
        # Pipe.send can block while a service is stopped. Keep it off the
        # futures lock and caller thread so request deadlines and fail() work.
        self._sender = threading.Thread(target=self._send, daemon=True)
        self._sender.start()

    def check(self) -> None:
        if (
            self._error
            or self._closed
            or self._closing
            or self.execution.remaining() <= 0
            or not self.process.is_alive()
        ):
            raise InferenceFailure(self._error or "inference allocation stopped")

    def submit_game(self, run: Callable[[], dict[str, Any]]) -> Future[dict[str, Any]]:
        """Bound both threads and queued games across all callbacks."""
        while not self._game_capacity.acquire(
            timeout=min(0.1, self.execution.remaining())
        ):
            self.check()
        try:
            self.check()
            future = self._games.submit(run)
        except BaseException:
            self._game_capacity.release()
            raise
        future.add_done_callback(lambda _: self._game_capacity.release())
        return future

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

    def _call(self, kind: str, payload: object, timeout: float | None = None) -> Any:
        timeout = min(
            self.execution.request_seconds,
            self.execution.request_seconds if timeout is None else timeout,
            self.execution.remaining(),
        )
        if timeout <= 0:
            raise InferenceFailure("inference request deadline exhausted")
        deadline = time.monotonic() + timeout
        if not self._capacity.acquire(timeout=timeout):
            raise InferenceFailure("inference backpressure deadline exhausted")
        try:
            self.check()
            identity = uuid4().hex
            future: Future[Any] = Future()
            with self._lock:
                self._futures[identity] = future
                self._outgoing.put_nowait((identity, kind, payload))
            try:
                return future.result(timeout=max(0, deadline - time.monotonic()))
            except TimeoutError as exc:
                self.fail("inference service request timed out")
                raise InferenceFailure("inference service request timed out") from exc
        finally:
            self._capacity.release()

    def admit(self, admission: PolicyAdmission) -> ObservationConfig:
        result = self._call("admit", admission)
        if not isinstance(result, ObservationConfig):
            raise InferenceFailure("invalid observation admission reply")
        return result

    def infer(self, request: DecisionRequest) -> DecisionReply:
        try:
            result = self._call("infer", request, request.deadline_unix - time.time())
            if not isinstance(result, DecisionReply):
                raise InferenceFailure("invalid inference reply")
            result.validate(request)
            if time.time() >= request.deadline_unix:
                raise InferenceFailure("inference reply arrived after its deadline")
            return result
        except InferenceFailure as exc:
            self.fail(str(exc))
            raise

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
        if self._futures:
            self.fail("allocation closed with pending requests")
        self._games.shutdown(wait=True, cancel_futures=True)
        try:
            self._outgoing.put_nowait(("close", "close", None))
        except queue.Full:
            self.fail("inference shutdown queue saturated")
        self.process.join(timeout=min(5, self.execution.remaining()))
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self._reader.join(timeout=1)
        try:
            self._outgoing.put_nowait(None)
        except queue.Full:
            pass
        self._sender.join(timeout=1)
        self._closed = True
        self._connection.close()
        self.performance.write(self.directory / "allocation.json")

        atomic_json(
            self.directory / "execution.json",
            {
                "allocation_id": self.identity,
                "execution": self.execution.model_dump(),
                "cost": allocation_cost(time.monotonic() - self.performance.started),
                "error": self._error,
            },
        )
