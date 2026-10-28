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
from uuid import uuid4

from manabot.llm.schema import (
    DecisionInput,
    EncodedPrompt,
    Failure,
    InferenceConfig,
    RuntimeIdentity,
    SelectionResult,
)
from manabot.llm.selection import Runtime, select


@dataclass(frozen=True)
class _DecisionRequest:
    request_id: str
    behavior_version: str
    decision: DecisionInput
    expected_prompt: EncodedPrompt | None


@dataclass(frozen=True)
class _Ready:
    identity: RuntimeIdentity
    behavior_version: str


@dataclass(frozen=True)
class _Progress:
    request_id: str
    result: SelectionResult


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
        version = runtime.identity.behavior_version(config)
        connection.send(_Ready(runtime.identity, version))
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            if request is None:
                return
            if not isinstance(request, _DecisionRequest):
                raise TypeError("expected decision request")
            if request.behavior_version != version:
                raise ValueError("request differs from admitted behavior version")

            def publish(result: SelectionResult) -> None:
                connection.send(_Progress(request.request_id, result))

            result = select(
                request.decision,
                config,
                runtime,
                progress=publish,
                expected_prompt=request.expected_prompt,
            )
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
            self.behavior_version = self.identity.behavior_version(config)
            if message.behavior_version != self.behavior_version:
                raise RuntimeError("worker admitted a different behavior version")
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
        expected_prompt: EncodedPrompt | None = None,
    ) -> SelectionResult:
        if self.closed:
            raise RuntimeError("inference session is closed")
        started = time.monotonic()
        deadline = started + self.config.deadline_seconds
        result = SelectionResult(
            decision=decision,
            config=self.config,
            runtime=self.identity,
            behavior_version=self.behavior_version,
        )
        try:
            request_id = uuid4().hex
            self.connection.send(
                _DecisionRequest(
                    request_id, self.behavior_version, decision, expected_prompt
                )
            )
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.connection.poll(remaining):
                    raise TimeoutError("decision wall deadline exhausted")
                message = self.connection.recv()
                if not isinstance(message, _Progress):
                    raise RuntimeError("worker did not report selection progress")
                received = message.result
                if (
                    message.request_id != request_id
                    or received.behavior_version != self.behavior_version
                    or received.config != self.config
                    or received.runtime != self.identity
                    or received.decision != decision
                ):
                    raise RuntimeError("stale or mismatched behavior reply")
                if time.monotonic() >= deadline:
                    raise TimeoutError("decision wall deadline exhausted")
                result = received
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
