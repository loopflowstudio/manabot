"""Bounded experiment child ownership without model or native imports.

The caller retains progress on disk. This supervisor owns one process group,
measures parent wall time (never summed worker time), and reaps children even
when the worker exits before its descendants. No retry is performed.
"""

from collections.abc import Callable
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
import signal
import time

import psutil
from pydantic import JsonValue, TypeAdapter


@dataclass(frozen=True)
class ProcessOutcome:
    payload: dict[str, JsonValue] | None
    error: str | None
    seconds: float
    cleanup_seconds: float
    peak_process_tree_rss_bytes: int
    exitcode: int | None
    cleanup_complete: bool


def _entry(
    target: Callable[..., None], sender: Connection, args: tuple[object, ...]
) -> None:
    if hasattr(os, "setsid"):
        os.setsid()
    try:
        target(sender, *args)
    finally:
        sender.close()


def bounded_call(
    target: Callable[..., None],
    args: tuple[object, ...],
    *,
    deadline_unix: float,
    process_tree_bytes: int,
    on_progress: Callable[[dict[str, JsonValue]], None] | None = None,
) -> ProcessOutcome:
    """Reserve two seconds of the supplied deadline for terminate/kill cleanup.

    With a progress consumer, messages containing only ``progress`` are small
    observations, not completion. Consumers must retain them without blocking;
    the next poll still uses the original deadline and RSS ceiling.
    """
    started = time.monotonic()
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_entry, args=(target, sender, args))
    descendants: dict[int, psutil.Process] = {}
    peak = 0
    payload = None
    error = None
    cleanup_complete = True
    exitcode = None
    message_adapter = TypeAdapter(dict[str, JsonValue])
    try:
        if time.time() >= deadline_unix - 2:
            raise TimeoutError("no execution budget remains before cleanup reserve")
        process.start()
        sender.close()
        while True:
            remaining = deadline_unix - 2 - time.time()
            if remaining <= 0:
                raise TimeoutError("attempt wall deadline exhausted")
            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                descendants.update((child.pid, child) for child in children)
                rss = sum(p.memory_info().rss for p in [parent, *children])
                peak = max(peak, rss)
                if rss > process_tree_bytes:
                    raise MemoryError("process tree RSS limit exceeded")
            except psutil.NoSuchProcess:
                pass
            if receiver.poll(min(0.1, remaining)):
                message = message_adapter.validate_python(receiver.recv())
                if on_progress is not None and set(message) == {"progress"}:
                    progress = message_adapter.validate_python(message["progress"])
                    on_progress(progress)
                    continue
                payload = message
                break
            if not process.is_alive():
                raise RuntimeError(
                    f"worker exited without a result: {process.exitcode}"
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_start = time.monotonic()
        sender.close()
        if process.pid is not None:
            try:
                descendants.update(
                    (p.pid, p)
                    for p in psutil.Process(process.pid).children(recursive=True)
                )
            except psutil.NoSuchProcess:
                pass
            # Allow the result sender to finish its normal finally blocks first.
            process.join(timeout=0.1)
            group = hasattr(os, "killpg")
            if group:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            elif process.is_alive():
                process.terminate()
            for child in descendants.values():
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            process.join(timeout=0.5)
            if group:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            if process.is_alive():
                process.kill()
            for child in descendants.values():
                try:
                    if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.join(timeout=0.5)
            _, alive = psutil.wait_procs(list(descendants.values()), timeout=0.5)
            cleanup_complete = not process.is_alive()
            for child in alive:
                try:
                    cleanup_complete &= child.status() == psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    pass
            exitcode = process.exitcode
            if not process.is_alive():
                process.close()
        receiver.close()
    return ProcessOutcome(
        payload,
        error,
        time.monotonic() - started,
        time.monotonic() - cleanup_start,
        peak,
        exitcode,
        cleanup_complete,
    )
