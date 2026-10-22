"""Training runtime setup and the public Etude tracking client.

Experiment owns seeding and the learner's tracking lifetime, not scheduling or
scientific evaluation. Etude owns metric storage, checkpoint copies and reports.
Legacy W&B configuration fields remain readable in frozen recipes but do not
select a second live tracker. Historical W&B readers live elsewhere.
"""

from collections.abc import Mapping
import logging
import os
from pathlib import Path
import random
from types import TracebackType
from typing import Literal, Self

from etude.run.records import ArtifactRef
from etude.train.records import Step
from etude.train.run import Run, track
import numpy as np
import torch

import manabot.infra.log
from manabot.infra.profiler import Profiler

from .hypers import ExperimentHypers, Hypers


def seed_random(seed: int, *, torch_deterministic: bool = True) -> None:
    """Reset training RNGs without constructing a tracking run.

    Call at the declared sampling boundary: evaluation and checkpoint admission
    can consume RNG state before collection begins.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def flatten_config(
    cfg: Mapping[str, object], parent_key: str = "", sep: str = "/"
) -> dict[str, object]:
    """Flatten named configuration for historical reporting consumers."""
    items: dict[str, object] = {}
    for key, value in cfg.items():
        name = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_config(value, name, sep))
        else:
            items[name] = value
    return items


class Experiment:
    """Seed the learner and record locally or into an Etude-launched run.

    A standalone invocation writes below runs_dir; under the SDK launcher its
    run identity and tracking directory come from ETUDE_RUN/ETUDE_TRACK_DIR.
    SDK admission failures are not silently converted to untracked training.
    """

    def __init__(
        self,
        experiment_hypers: ExperimentHypers | None = None,
        full_hypers: Hypers | None = None,
        *,
        resolved_config: Mapping[str, object] | None = None,
    ) -> None:
        self.experiment_hypers = experiment_hypers or ExperimentHypers()
        self.full_hypers = full_hypers or Hypers()
        config = self.experiment_hypers
        self.exp_name = config.exp_name
        self.seed = config.seed
        self.torch_deterministic = config.torch_deterministic
        self.device = config.device
        self.runs_dir = config.runs_dir / self.exp_name
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.profiler = Profiler(enabled=config.profiler_enabled)
        level = getattr(logging, config.log_level.upper(), logging.INFO)
        manabot.infra.log.setGlobalLogLevel(level)
        self.logger = manabot.infra.log.getLogger(__name__)
        seed_random(self.seed, torch_deterministic=self.torch_deterministic)
        self.run: Run = track(
            experiment=self.exp_name,
            clocks=[
                Step("learner_transitions", "collected learner transitions"),
                Step("updates", "completed collect/update iterations"),
            ],
            seed=self.seed,
            config={
                "training": dict(resolved_config)
                if resolved_config is not None
                else full_hypers.model_dump(mode="json")
                if full_hypers is not None
                else None,
                "runtime": config.model_dump(mode="json"),
            },
            directory=None
            if os.environ.get("ETUDE_TRACK_DIR")
            else self.runs_dir / "etude",
            strict=True,
        )
        self._closed = False
        self._updates = 0
        self._transitions = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close("failed" if exc_type else "finished")

    def log(self, metrics: Mapping[str, object], step: int | None = None) -> None:
        """Record metrics; the legacy step argument counts learner transitions."""
        if self._closed:
            raise RuntimeError("cannot log to a closed experiment")
        if step is not None:
            self._transitions = step
        self.run.log(
            metrics, updates=self._updates, learner_transitions=self._transitions
        )

    def log_update(
        self, metrics: Mapping[str, object], *, updates: int, transitions: int
    ) -> None:
        """Record completed work without conflating iterations and samples."""
        if self._closed:
            raise RuntimeError("cannot log to a closed experiment")
        self._updates = updates
        self._transitions = transitions
        self.run.log(metrics, updates=updates, learner_transitions=transitions)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.log({tag: value}, step=step)

    def checkpoint(self, path: Path, *, transitions: int) -> ArtifactRef:
        """Retain exact exported bytes; this is not policy or resume admission."""
        if self._closed:
            raise RuntimeError("cannot checkpoint a closed experiment")
        return self.run.checkpoint(
            path, updates=self._updates, learner_transitions=transitions
        )

    def close(self, state: Literal["finished", "failed"] = "finished") -> None:
        """Flush tracking once; callers must mark failed training explicitly."""
        if not self._closed:
            self.run.finish(state)
            self._closed = True

    def log_performance(self, step: int | None = None) -> None:
        """Record profiler fractions, not a separately reconstructed time clock."""
        self.log(
            {
                f"performance/{name}/pct_of_total": data["pct_of_total"]
                for name, data in self.profiler.get_stats().items()
            },
            step=step,
        )
