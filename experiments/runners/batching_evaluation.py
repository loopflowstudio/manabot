"""Evaluate retained grid policies at exact, observed experience coordinates.

The caller freezes the study and its MonitorProtocol before scoring. This runner
selects initialization, retained milestones and the endpoint from TrainingRun
records, verifies relocated artifact bytes, and delegates games and replay checks
to the ordinary arena evaluator. It never rounds a requested transition count,
substitutes an endpoint, trains, or retries an incomplete evaluation.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from manabot.arena.models import file_sha256
from manabot.training.execution import atomic_json
from manabot.training.models import ArtifactReference, TrainingCoordinates, TrainingRun
from manabot.training.monitor_evaluation import (
    Checkpoint,
    MonitorProtocol,
    evaluate_checkpoint,
    stage_checkpoint,
)


@dataclass(frozen=True)
class CheckpointSelection:
    run: TrainingRun
    points: tuple[Checkpoint, ...]


def select_checkpoints(
    directory: Path, transitions: tuple[int, ...]
) -> CheckpointSelection:
    """Admit exact raw exports from a completed, fresh single-stage trajectory.

    Relocation changes only the artifact path. Original byte counts, digests and
    observed learning coordinates remain binding, including for initialization.
    Validate the entire panel before its first game can consume an allocation.
    """
    if (
        not transitions
        or tuple(sorted(set(transitions))) != transitions
        or transitions[0] < 0
    ):
        raise ValueError("transition milestones must be nonnegative and increase")
    run = TrainingRun.model_validate_json((directory / "run.json").read_text())
    if (
        run.status != "completed"
        or len(run.stages) != 1
        or run.parent_run_id is not None
        or run.stages[0].learning_state_origin is not None
        or run.regime.stages[0].operation != "train_self_play"
    ):
        raise ValueError("evaluation requires a completed fresh single self-play stage")
    stage = run.stages[0]
    initial = TypeAdapter(ArtifactReference).validate_python(
        stage.artifacts["initial_raw"]
    )
    available = [
        Checkpoint(
            artifact=initial,
            coordinates=TrainingCoordinates(
                stage_id=stage.id,
                updates=0,
                training_seconds=0,
                active_training_seconds=0,
                optimizer_updates=0,
            ),
        )
    ]
    for saved in run.monitoring_checkpoints:
        if saved.artifact is None or saved.error is not None:
            continue
        available.append(
            Checkpoint(
                artifact=saved.artifact,
                coordinates=TrainingCoordinates.model_validate(
                    saved.model_dump(exclude={"ordinal", "artifact", "error"})
                ),
            )
        )
    endpoint = stage_checkpoint(run, stage.id)
    if endpoint is None:
        raise ValueError("completed run has no admitted final checkpoint")
    available.append(endpoint)
    points: list[Checkpoint] = []
    for count in transitions:
        matches = [p for p in available if p.coordinates.learner_transitions == count]
        if len(matches) != 1:
            raise ValueError(f"expected one retained checkpoint at {count} transitions")
        point = matches[0]
        path = directory / Path(point.artifact["path"]).name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("checkpoint resolves outside the retained run")
        if (
            path.stat().st_size != point.artifact["bytes"]
            or file_sha256(path) != point.artifact["sha256"]
        ):
            raise ValueError(f"checkpoint bytes differ at {count} transitions")
        points.append(
            point.model_copy(
                update={
                    "artifact": {
                        **point.artifact,
                        "path": str(path.resolve()),
                    }
                }
            )
        )
    return CheckpointSelection(run, tuple(points))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--transitions", type=int, nargs="+", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = MonitorProtocol.model_validate_json(args.protocol.read_text())
    selected = select_checkpoints(args.run, tuple(args.transitions))
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(
        args.out / "selection.json",
        {
            "run_sha256": file_sha256(args.run / "run.json"),
            "run_id": selected.run.id,
            "seed": selected.run.seed,
            "protocol": protocol.model_dump(mode="json"),
            "checkpoints": [p.model_dump(mode="json") for p in selected.points],
        },
    )
    for point in selected.points:
        count = point.coordinates.learner_transitions
        result = evaluate_checkpoint(
            selected.run,
            point.artifact,
            point.coordinates,
            args.out / f"transitions-{count}",
            protocol=protocol,
            concurrent_activity="post-training panel; other host activity unmeasured",
        )
        # The evaluator validates cohort completeness and replay before marking
        # its result completed; an incomplete cohort must stop this panel.
        if result.status != "completed":
            raise RuntimeError(
                f"evaluation incomplete at {count} transitions: {result.error}"
            )


if __name__ == "__main__":
    main()
