"""One-shot inference supervision with a hard wall deadline and retained progress.

Each invocation loads one model in a fresh child process. The parent owns the
complete decision deadline, including imports, downloads, loading and repair.
Completed prompt/response snapshots survive runtime crashes and forced stops.
"""

from multiprocessing.connection import Connection
import time

from manabot.llm.schema import DecisionInput, InferenceConfig, SelectionResult
from manabot.llm.selection import select
from manabot.sim.child import ChildProcess


def _worker(
    connection: Connection,
    decision: DecisionInput,
    config: InferenceConfig,
    started: float,
) -> None:
    with connection:
        try:
            # Optional dependency: core imports and protocol tests need no Transformers.
            from manabot.llm.transformers_runtime import TransformersRuntime

            runtime = TransformersRuntime(config)
        except Exception as error:
            connection.send(
                SelectionResult(
                    decision=decision,
                    config=config,
                    seconds=time.monotonic() - started,
                ).invalidate(error, kind="runtime")
            )
            return
        # Selection owns inference failures; the parent retains every snapshot
        # even if this worker exits unexpectedly. No second evidence copy is needed.
        select(decision, config, runtime, started=started, progress=connection.send)


def run_decision(decision: DecisionInput, config: InferenceConfig) -> SelectionResult:
    """Load a pinned model and select once; no runtime failure permits fallback."""
    started = time.monotonic()
    deadline = started + config.deadline_seconds
    result = SelectionResult(decision=decision, config=config)
    process: ChildProcess | None = None
    try:
        process = ChildProcess(
            "manabot.llm.runner:_worker", (decision, config, started)
        )
        receiver = process.connection
        while True:
            remaining = deadline - time.monotonic()
            # The worker owns the only remaining sender. Its exit makes the pipe
            # readable at EOF, after any queued evidence; no liveness polling is needed.
            if remaining <= 0 or not receiver.poll(remaining):
                raise TimeoutError("decision wall deadline exhausted")
            try:
                message = receiver.recv()
            except EOFError:
                break
            if not isinstance(message, SelectionResult):
                raise RuntimeError("worker did not report selection progress")
            result = message
            if time.monotonic() >= deadline:
                raise TimeoutError("decision wall deadline exhausted")
            if result.failure is not None or result.origin is not None:
                break
        if result.origin is None and result.failure is None:
            raise RuntimeError("inference worker exited without a result")
    except Exception as error:
        # Parent failures invalidate even a selection received after the deadline.
        result = result.invalidate(error)
    finally:
        if process is not None:
            # Let a completed worker release tokenizer/download resources normally.
            process.join(timeout=min(1, max(0, deadline - time.monotonic())))
            process.close()
    return result.model_copy(update={"seconds": time.monotonic() - started})
