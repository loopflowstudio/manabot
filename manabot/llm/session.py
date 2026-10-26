"""Keep one pinned runtime loaded while supervising sequential decisions.

The parent owns each wall deadline and retains progress before terminating a
failed worker. Loading has its own deadline; decision timings exclude loading.
No game state crosses this boundary, and an interrupted session cannot resume.
"""

from dataclasses import dataclass
import multiprocessing
from multiprocessing.connection import Connection
import time
from typing import Callable

from manabot.llm.schema import (
    DecisionInput,
    Failure,
    InferenceConfig,
    RuntimeIdentity,
    SelectionResult,
)
from manabot.llm.selection import Runtime, select


@dataclass(frozen=True)
class _Ready:
    identity: RuntimeIdentity


@dataclass(frozen=True)
class _LoadFailure:
    message: str


def _runtime(config: InferenceConfig) -> Runtime:
    from manabot.llm.transformers_runtime import TransformersRuntime

    return TransformersRuntime(config)


def _serve(
    connection: Connection,
    config: InferenceConfig,
    factory: Callable[[InferenceConfig], Runtime],
) -> None:
    with connection:
        try:
            runtime = factory(config)
        except Exception as error:
            connection.send(_LoadFailure(f"{type(error).__name__}: {error}"))
            return
        connection.send(_Ready(runtime.identity))
        while True:
            try:
                decision = connection.recv()
            except EOFError:
                return
            if decision is None:
                return
            if not isinstance(decision, DecisionInput):
                raise TypeError("expected DecisionInput")
            result = select(decision, config, runtime, progress=connection.send)
            if result.failure is not None:
                return


class InferenceSession:
    """One loaded worker, one outstanding decision, no automatic restart."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        load_seconds: float = 120,
        factory: Callable[[InferenceConfig], Runtime] = _runtime,
    ) -> None:
        self.config = config
        self.closed = False
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child, config, factory))
        started = time.monotonic()
        try:
            self.process.start()
            child.close()
            if not self.connection.poll(load_seconds):
                raise TimeoutError("model loading wall deadline exhausted")
            message = self.connection.recv()
            if isinstance(message, _LoadFailure):
                raise RuntimeError(message.message)
            if not isinstance(message, _Ready):
                raise RuntimeError("worker did not report runtime identity")
            self.identity = message.identity
            self.load_seconds = time.monotonic() - started
        except BaseException:
            child.close()
            self.close()
            raise

    def decide(
        self,
        decision: DecisionInput,
        *,
        progress: Callable[[SelectionResult], None] | None = None,
    ) -> SelectionResult:
        if self.closed:
            raise RuntimeError("inference session is closed")
        started = time.monotonic()
        deadline = started + self.config.deadline_seconds
        result = SelectionResult(
            decision=decision, config=self.config, runtime=self.identity
        )
        try:
            self.connection.send(decision)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.connection.poll(remaining):
                    raise TimeoutError("decision wall deadline exhausted")
                message = self.connection.recv()
                if not isinstance(message, SelectionResult):
                    raise RuntimeError("worker did not report selection progress")
                result = message
                if progress is not None:
                    progress(result)
                if time.monotonic() >= deadline:
                    raise TimeoutError("decision wall deadline exhausted")
                if result.failure is not None or result.origin is not None:
                    break
        except Exception as error:
            result = result.model_copy(
                update={
                    "selected_reference": None,
                    "origin": None,
                    "fallback": None,
                    "failure": Failure(
                        kind="deadline"
                        if isinstance(error, TimeoutError)
                        else "runtime",
                        message=f"{type(error).__name__}: {error}",
                    ),
                }
            )
        if result.failure is not None:
            self.close()
        return result.model_copy(update={"seconds": time.monotonic() - started})

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.connection.close()
        if self.process.pid is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
            self.process.close()

    def __enter__(self) -> "InferenceSession":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
