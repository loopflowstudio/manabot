"""Expose ordinary TrainingRun progress through Etude's existing result contract.

The store remains authoritative. Publication observes completed persistence calls
on the learner's thread; it never reads a live model or changes its RNG. Exported
policy checkpoints support evaluation, not exact optimizer/collector resumption.
"""

import json
from pathlib import Path
import time

from manabot.training.execution import atomic_json
from manabot.training.models import ArtifactReference, TrainingRun
from manabot.training.monitor_evaluation import stage_checkpoint
from manabot.verify.store import VerifyStore


class ProgressStore(VerifyStore):
    def __init__(self, path: Path, output: Path | None) -> None:
        super().__init__(path)
        self.output = output.resolve() if output else None
        self._published_at = float("-inf")
        self._checkpoint: str | None = None

    def save_training_run(self, run: TrainingRun) -> None:
        super().save_training_run(run)
        if self.output is None:
            return
        candidates: list[ArtifactReference] = []
        for stage in run.stages:
            if "initial_raw" in stage.artifacts:
                candidates.append(stage.artifacts["initial_raw"])
        candidates += [
            receipt.artifact
            for receipt in run.monitoring_checkpoints
            if receipt.artifact is not None
        ]
        if run.status == "completed":
            for stage in run.stages:
                final = stage_checkpoint(run, stage.id)
                if final is not None:
                    candidates.append(final.artifact)
        checkpoint = self._checkpoint
        for artifact in candidates:
            path = Path(artifact["path"]).resolve()
            if path.suffix == ".pt" and path.is_file():
                if not path.is_relative_to(self.output):
                    raise ValueError("Published checkpoint must be in Etude outputs")
                checkpoint = path.relative_to(self.output).as_posix()
        now = time.monotonic()
        if (
            now - self._published_at < 10
            and checkpoint == self._checkpoint
            and run.status not in {"completed", "failed", "interrupted", "paused"}
        ):
            return
        clocks = {
            "updates": run.updates_through(),
            "training_seconds": run.seconds,
            "learner_transitions": sum(s.learner_transitions for s in run.stages),
            "observed_unix_seconds": time.time(),
        }
        atomic_json(
            self.output / "etude-result.json",
            {"clocks": clocks, "checkpoint": checkpoint},
        )
        print(
            json.dumps({"phase": "training", "status": run.status, **clocks}),
            flush=True,
        )
        self._published_at, self._checkpoint = now, checkpoint
