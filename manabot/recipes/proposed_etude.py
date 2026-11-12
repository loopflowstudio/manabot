"""A local stand-in for the Etude run API the recipes are written against.

A recipe writes `with track(...) as run:` and uses `run.out` (the run's
directory, kept whole across attempts), `run.log`, `run.log_table`,
`run.checkpoint` and `run.stopping()`. Leaving the block finishes the run, as
failed when it raised. Etude's own `Run` (`etude.train.run`) offers all of
this except `out` and `stopping`; `TrackedRun` adds those two over it so the
recipes run today, and this module is deleted when Etude ships them (then
recipes import `track` from `etude.train.run`).

What each addition stands in for:

- `out` is `ETUDE_OUT`, otherwise `.runs/<experiment>`. A recipe names no
  experiment: the launch does (`ETUDE_EXPERIMENT`). The Etude metric run lives
  in `out/etude/` under the out directory's name, so running again resumes the
  same run.
- `stopping()` is true once `ETUDE_STOP_FILE` exists, or once `ETUDE_DEADLINE`
  (ISO time) is nearer than the longest interval between two `stopping()`
  calls so far, which for a loop that asks once per update is its slowest
  update.
- `local_launch(out=, experiment=)` sets what a launch would for the block, so
  a command line can call a recipe as Etude does.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import time
from types import TracebackType
from typing import Self

from etude.train.records import Step
from etude.train.run import Run, track as etude_track

__all__ = ["Step", "TrackedRun", "local_launch", "track"]


@dataclass
class TrackedRun:
    """Etude's `Run` plus the run directory and the stop signal."""

    etude: Run
    out: Path
    # Monotonic time of the last `stopping()` call and the longest gap between two.
    asked: float | None = None
    slowest: float = 0.0

    def log(self, values: Mapping[str, object], /, **clocks: float) -> None:
        self.etude.log(dict(values), **clocks)

    def log_table(
        self, name: str, rows: Sequence[Mapping[str, object]], /, **clocks: float
    ) -> object:
        return self.etude.log_table(name, rows, **clocks)

    def checkpoint(
        self, path: Path, /, *, tags: Sequence[str] = (), **clocks: float
    ) -> object:
        return self.etude.checkpoint(path, tags=tags, **clocks)

    def stopping(self) -> bool:
        now = time.monotonic()
        if self.asked is not None:
            self.slowest = max(self.slowest, now - self.asked)
        self.asked = now
        return _stop_requested() or _seconds_left() < self.slowest

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.etude.finish("failed" if error is not None else "finished")


def _stop_requested() -> bool:
    stop = os.environ.get("ETUDE_STOP_FILE")
    return bool(stop) and Path(stop).exists()


def _seconds_left() -> float:
    deadline = os.environ.get("ETUDE_DEADLINE")
    if not deadline:
        return float("inf")
    return datetime.fromisoformat(deadline).timestamp() - time.time()


def track(
    *,
    clocks: Sequence[Step],
    seed: int | None = None,
    config: Mapping[str, object] | None = None,
) -> TrackedRun:
    """Declare or resume the run whose directory is `ETUDE_OUT`."""
    experiment = os.environ.get("ETUDE_EXPERIMENT")
    if experiment is None:
        raise RuntimeError(
            "no experiment names this run: launch it with Etude or "
            "`manabot train --recipe`, or set ETUDE_EXPERIMENT"
        )
    out = Path(os.environ.get("ETUDE_OUT") or Path(".runs") / experiment).resolve()
    out.mkdir(parents=True, exist_ok=True)
    name = os.environ.get("ETUDE_RUN") or re.sub(r"[^A-Za-z0-9_.-]", "-", out.name)
    run = etude_track(
        clocks=clocks,
        run=name,
        seed=seed,
        config=config,
        directory=None if os.environ.get("ETUDE_TRACK_DIR") else out / "etude",
        strict=True,
    )
    return TrackedRun(run, out)


@contextmanager
def local_launch(*, out: Path | str, experiment: str) -> Iterator[None]:
    """Name the run directory and experiment for recipes called in the block."""
    launch = {"ETUDE_OUT": str(out), "ETUDE_EXPERIMENT": experiment}
    previous = {key: os.environ.get(key) for key in launch}
    os.environ.update(launch)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
