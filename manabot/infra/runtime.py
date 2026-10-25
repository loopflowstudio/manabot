"""Training runtime setup and the public Etude tracking client.

RunContext owns seeding and the learner's tracking lifetime, not scheduling or
scientific evaluation. Etude owns metric storage, checkpoint copies and reports.
Experiment definitions live in experiment.config; this module owns only
per-run setup and tracking.
"""

from collections.abc import Mapping
import logging
import os
from pathlib import Path
import random
from types import TracebackType
from typing import Literal, Self

from etude.run.records import ArtifactRef, RunFile
from etude.train.records import Step
from etude.train.run import Run, track
import numpy as np
import torch

import manabot.infra.log

from .config import RuntimeConfig


def seed_random(seed: int, *, torch_deterministic: bool = True) -> None:
    """Reset training RNGs without constructing a tracking run.

    Call at the declared sampling boundary: evaluation and checkpoint admission
    can consume RNG state before collection begins.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


class RunContext:
    """Seed the learner and record locally or into an Etude-launched run.

    A standalone invocation writes below runs_dir, or continues the existing
    run folder `resume` (a recovery attempt resumes its parent's metrics and
    seals rows the parent left unsealed). `resumed_at` is where this process
    stands in that run, a restored snapshot or a stage boundary, which need not
    be the run's latest Etude checkpoint; earlier rows after it are superseded. Under the SDK launcher the run
    identity and tracking directory come from ETUDE_RUN/ETUDE_TRACK_DIR.
    SDK admission failures are not silently converted to untracked training.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        run: str | None = None,
        resume: Path | None = None,
        resumed_at: Mapping[str, float] | None = None,
        resolved_config: Mapping[str, object] | None = None,
    ) -> None:
        self.device = config.device
        self.runs_dir = config.runs_dir / config.name
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        level = getattr(logging, config.log_level.upper(), logging.INFO)
        manabot.infra.log.setGlobalLogLevel(level)
        seed_random(config.seed, torch_deterministic=config.torch_deterministic)
        self.run: Run = track(
            # A resumed run keeps the experiment it was declared under, which
            # names the stage that started it, not necessarily this one.
            experiment=RunFile.model_validate_json(
                (resume / "run.json").read_bytes()
            ).experiment
            if resume
            else config.name,
            # The launcher's identity wins; it owns continuation of its runs.
            run=os.environ.get("ETUDE_RUN") or (resume.name if resume else run),
            clocks=[
                Step("learner_transitions", "collected learner transitions"),
                Step("updates", "completed collect/update iterations"),
            ],
            seed=config.seed,
            config={
                "training": dict(resolved_config)
                if resolved_config is not None
                else None,
                "runtime": config.model_dump(mode="json"),
            },
            directory=None
            if os.environ.get("ETUDE_TRACK_DIR")
            else resume.parent
            if resume
            else self.runs_dir / "etude",
            strict=True,
            resumed_at=resumed_at,
        )
        # A launcher retry restores the run's checkpoints and expects the job
        # to resume from the latest, but a policy export cannot restore
        # collector or optimizer state.
        if resume is None and self.run.latest_checkpoint() is not None:
            self.run.finish("failed")
            raise ValueError("policy artifacts cannot resume collector/optimizer state")
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

    def log_update(
        self, metrics: Mapping[str, object], *, updates: int, transitions: int
    ) -> None:
        """Record completed work without conflating iterations and samples."""
        if self._closed:
            raise RuntimeError("cannot log to a closed run context")
        self._updates = updates
        self._transitions = transitions
        self.run.log(metrics, updates=updates, learner_transitions=transitions)

    def checkpoint(self, path: Path, *, transitions: int) -> ArtifactRef:
        """Retain exact exported bytes; this is not policy or resume admission."""
        if self._closed:
            raise RuntimeError("cannot checkpoint a closed run context")
        return self.run.checkpoint(
            path, updates=self._updates, learner_transitions=transitions
        )

    def close(self, state: Literal["finished", "failed"] = "finished") -> None:
        """Flush tracking once; callers must mark failed training explicitly."""
        if not self._closed:
            self.run.finish(state)
            self._closed = True
