"""One-shot inference supervision with a hard wall deadline and retained progress.

Each invocation loads one model in a fresh spawned process. The parent owns the
complete decision deadline, including imports, downloads, loading and repair.
Completed prompt/response snapshots survive runtime crashes and forced stops.
"""

import multiprocessing
from multiprocessing.connection import Connection
import time

from manabot.llm.schema import DecisionInput, Failure, InferenceConfig, SelectionResult
from manabot.llm.selection import select


def _worker(
    connection: Connection,
    decision: DecisionInput,
    config: InferenceConfig,
    started: float,
) -> None:
    def publish(snapshot: SelectionResult) -> None:
        connection.send(snapshot.model_dump_json())

    with connection:
        try:
            # Optional dependency: core imports and protocol tests need no Transformers.
            from manabot.llm.transformers_runtime import TransformersRuntime

            runtime = TransformersRuntime(config)
        except Exception as error:
            publish(
                SelectionResult(
                    decision=decision,
                    config=config,
                    failure=Failure(
                        kind="runtime", message=f"{type(error).__name__}: {error}"
                    ),
                    seconds=time.monotonic() - started,
                )
            )
            return
        # Selection owns inference failures; the parent retains every snapshot
        # even if this worker exits unexpectedly. No second evidence copy is needed.
        select(decision, config, runtime, started=started, progress=publish)


def run_decision(decision: DecisionInput, config: InferenceConfig) -> SelectionResult:
    """Load a pinned model and select once; no runtime failure permits fallback."""
    started = time.monotonic()
    deadline = started + config.deadline_seconds
    result = SelectionResult(decision=decision, config=config)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, decision, config, started))
    try:
        process.start()
        sender.close()
        while True:
            remaining = deadline - time.monotonic()
            # The worker owns the only remaining sender. Its exit makes the pipe
            # readable at EOF, after any queued evidence; no liveness polling is needed.
            if remaining <= 0 or not receiver.poll(remaining):
                raise TimeoutError("decision wall deadline exhausted")
            try:
                result = SelectionResult.model_validate_json(receiver.recv())
            except EOFError:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("decision wall deadline exhausted")
            if result.failure is not None or result.origin is not None:
                break
        if result.origin is None and result.failure is None:
            raise RuntimeError("inference worker exited without a result")
    except Exception as error:
        # Parent failures invalidate even a selection received after the deadline.
        result = result.model_copy(
            update={
                "selected_reference": None,
                "origin": None,
                "fallback": None,
                "failure": Failure(
                    kind="deadline" if isinstance(error, TimeoutError) else "runtime",
                    message=f"{type(error).__name__}: {error}",
                ),
            }
        )
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            # Let a completed worker release tokenizer/download resources normally.
            process.join(timeout=min(1, max(0, deadline - time.monotonic())))
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            process.close()
    return result.model_copy(update={"seconds": time.monotonic() - started})
