"""Ephemeral runner work, separate from Command evidence and native authority.

A runner owns one operation at a time. Readers copy its public status under a
small lock; they never touch the Env or initiate work. Stage callbacks run at
safe worker boundaries, before the next mutation or provider call.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from threading import Lock
from time import time
from typing import Literal
from uuid import uuid4

WorkKind = Literal["load", "choose", "search", "advance"]


@dataclass(frozen=True)
class WorkStatus:
    operation_id: str
    sequence: int
    kind: WorkKind
    started_at: float
    match_id: str | None
    revision: int | None
    chooser: int | None
    state: Literal["running", "complete", "failed"] = "running"


class Work:
    def __init__(self) -> None:
        self._lock = Lock()
        self._status: WorkStatus | None = None
        self._sequence = 0
        self.publish: Callable[[WorkStatus], None] | None = None

    def read(self) -> WorkStatus | None:
        with self._lock:
            return self._status

    def stage(
        self,
        kind: WorkKind,
        match_id: str | None = None,
        revision: int | None = None,
        chooser: int | None = None,
    ) -> None:
        with self._lock:
            self._sequence += 1
            status = WorkStatus(
                uuid4().hex, self._sequence, kind, time(), match_id, revision, chooser
            )
            self._status = status
        if self.publish is not None:
            self.publish(status)

    def finish(self, *, failed: bool = False) -> None:
        with self._lock:
            if self._status is None:
                return
            status = replace(self._status, state="failed" if failed else "complete")
            self._status = status
        if self.publish is not None:
            self.publish(status)

    def payload(self) -> dict[str, str | int | float | None]:
        status = self.read()
        return asdict(status) if status is not None else {}
