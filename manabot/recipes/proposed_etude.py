"""A local stand-in for the Etude run API the recipes are written against.

Recipes call `track(...)` and use the run it returns: `run.out` (the run's
directory, kept whole across attempts), `run.log`, `run.log_table`,
`run.checkpoint`, `run.stopping()` and `run.finish()`. Etude's own `Run`
(`etude.train.run`) offers everything here except `out` and `stopping`; this
module adds those two over it so the recipes run today, and it is deleted when
Etude ships them (then recipes import `track` from `etude.train.run`).

What each addition stands in for:

- `out` is `ETUDE_OUT` when the launcher (or `manabot train --out`) sets it,
  otherwise `.runs/<experiment>`. A recipe names no experiment: the launch
  does (`ETUDE_EXPERIMENT`), and `manabot train` sets it to the recipe's name. The Etude metric run lives in `out/etude/`
  under the out directory's name, so running again resumes the same run.
- `stopping()` is true once `ETUDE_STOP_FILE` exists, or once `ETUDE_DEADLINE`
  (ISO time) less the slowest interval between logged steps has passed.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
import os
from pathlib import Path
import re
import time
from typing import Literal

from etude.train.records import Step
from etude.train.run import Run, track as etude_track

__all__ = ["Step", "TrackedRun", "track"]


class TrackedRun:
    """Etude's `Run` plus the run directory and the stop signal."""

    def __init__(self, run: Run, out: Path) -> None:
        self.etude = run
        self.out = out
        self._last_step: float | None = None
        self._slowest = 0.0

    def log(self, values: Mapping[str, object], /, **clocks: float) -> None:
        now = time.monotonic()
        if self._last_step is not None:
            self._slowest = max(self._slowest, now - self._last_step)
        self._last_step = now
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
        stop = os.environ.get("ETUDE_STOP_FILE")
        if stop and Path(stop).exists():
            return True
        deadline = os.environ.get("ETUDE_DEADLINE")
        if not deadline:
            return False
        left = datetime.fromisoformat(deadline).timestamp() - time.time()
        return left < self._slowest

    def finish(self, state: Literal["finished", "failed"] = "finished") -> None:
        self.etude.finish(state)


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
