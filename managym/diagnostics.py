"""Runner phases and adaptive diagnostics over the engine's recording boundary.

Diagnostics owns the watchdog and signal handler; callers own its lifetime and
supply phase names. Escalation snapshots cached inputs without reading native
state, so it can run while a buffered call has released the interpreter lock.
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
import faulthandler
import json
from pathlib import Path
import signal
import threading
import time
from typing import BinaryIO, Literal, Protocol
import weakref

from managym.recording import GameRecord, atomic_private_json

type Level = Literal["off", "standard", "detailed"]

_SIGNAL_OWNER = threading.Lock()


class Capture(Protocol):
    def snapshot(self) -> GameRecord: ...
    def enable_journal(self, path: Path) -> None: ...


class Diagnostics:
    def __init__(
        self,
        directory: Path,
        *,
        captures: Sequence[Capture] = (),
        level: Level = "standard",
        stall_seconds: float = 600,
        slow_fraction: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
        context_provider: Callable[[], dict[str, object]] = dict,
    ) -> None:
        if level not in {"off", "standard", "detailed"}:
            raise ValueError("unknown diagnostics level")
        if stall_seconds <= 0 or not 0 < slow_fraction < 1:
            raise ValueError(
                "stall_seconds must be positive and slow_fraction in (0, 1)"
            )
        self.directory = directory
        self.level = level
        self.stall_seconds = stall_seconds
        self.slow_fraction = slow_fraction
        self._context_provider = context_provider
        self._clock = clock
        self._captures = [weakref.ref(capture) for capture in captures]
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._operations: list[tuple[str, float]] = []
        self._sequence = 0
        self._rate: float | None = None
        self._progress_at = clock()
        self._stalled_sequence: int | None = None
        self._stacks: BinaryIO | None = None
        self._signal_installed = False
        self._thread: threading.Thread | None = None
        if level == "off":
            return
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._stacks = (directory / "stacks.txt").open("ab")
        (directory / "stacks.txt").chmod(0o600)
        # Own SIGUSR1 while open, leaving a pre-existing Python handler alone.
        if hasattr(signal, "SIGUSR1") and not faulthandler.is_enabled():
            faulthandler.enable()
        if (
            hasattr(signal, "SIGUSR1")
            and signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL
            and _SIGNAL_OWNER.acquire(blocking=False)
        ):
            faulthandler.register(
                signal.SIGUSR1, file=self._stacks, all_threads=True, chain=False
            )
            self._signal_installed = True
        if level == "detailed":
            self.escalate("started with detailed diagnostics")
        self._thread = threading.Thread(
            target=self._watch, name="managym-watchdog", daemon=True
        )
        self._thread.start()

    def add_capture(self, capture: Capture) -> None:
        with self._lock:
            self._captures.append(weakref.ref(capture))
            if self.level == "detailed":
                capture.enable_journal(
                    self.directory / f"{len(self._captures) - 1}.jsonl"
                )

    def _mark(self, name: str, state: str) -> None:
        self._sequence += 1
        atomic_private_json(
            self.directory / "operation.json",
            {
                "operation": name,
                "state": state,
                "sequence": self._sequence,
                "timestamp": time.time(),
                "context": self._context_provider(),
            },
        )

    @contextmanager
    def operation(self, name: str) -> Iterator[None]:
        if self.level == "off":
            yield
            return
        with self._lock:
            self._operations.append((name, self._clock()))
            if name == "collection":
                self._progress_at = self._clock()
            self._mark(name, "pending")
        try:
            yield
        except BaseException:
            with self._lock:
                self._operations.pop()
            raise
        else:
            with self._lock:
                self._operations.pop()
                self._mark(name, "returned")
                if self._operations:
                    self._mark(self._operations[-1][0], "pending")

    def progress(self, decisions: int) -> None:
        if self.level == "off":
            return
        with self._lock:
            now = self._clock()
            elapsed = now - self._progress_at
            self._progress_at = now
            if elapsed <= 0 or decisions <= 0:
                return
            rate = decisions / elapsed
            if self._rate is not None and rate < self.slow_fraction * self._rate:
                self.escalate(
                    f"collection rate {rate:.6g} below {self.slow_fraction:g} of trailing {self._rate:.6g}"
                )
            self._rate = rate if self._rate is None else 0.8 * self._rate + 0.2 * rate

    def escalate(self, reason: str) -> None:
        if self.level == "off":
            return
        with self._lock:
            self.level = "detailed"
            atomic_private_json(
                self.directory / "stall.json",
                {
                    "reason": reason,
                    "timestamp": time.time(),
                    "operation": self._operations[-1][0] if self._operations else None,
                },
            )
            for index, reference in enumerate(self._captures):
                capture = reference()
                if capture is None:
                    continue
                atomic_private_json(
                    self.directory / f"{index}.snapshot.jsonl",
                    capture.snapshot().model_dump(mode="json", by_alias=True),
                )
                capture.enable_journal(self.directory / f"{index}.jsonl")
            if self._stacks is not None:
                faulthandler.dump_traceback(file=self._stacks, all_threads=True)

    def check(self) -> None:
        """One watchdog tick, also usable with an injected clock."""
        if self.level == "off":
            return
        with self._lock:
            if (
                self._operations
                and self._clock() - self._operations[-1][1] >= self.stall_seconds
            ):
                if self._stalled_sequence != self._sequence:
                    self._stalled_sequence = self._sequence
                    self.escalate(
                        f"operation has not returned for {self.stall_seconds:g} seconds"
                    )
            request = self.directory.parent / "diagnostics.request"
            if request.exists():
                request.unlink()
                self.escalate("operator requested diagnostics")

    def _watch(self) -> None:
        while not self._stop.wait(min(1.0, self.stall_seconds / 4)):
            try:
                self.check()
            except (OSError, ValueError) as error:
                # Preserve watchdog failures as evidence without terminating the runner.
                if self._stacks is not None:
                    self._stacks.write(
                        (json.dumps({"diagnostics_error": str(error)}) + "\n").encode()
                    )
                    self._stacks.flush()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._signal_installed:
            faulthandler.unregister(signal.SIGUSR1)
            self._signal_installed = False
            _SIGNAL_OWNER.release()
        if self._stacks is not None:
            self._stacks.close()
            self._stacks = None

    def __enter__(self) -> "Diagnostics":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
