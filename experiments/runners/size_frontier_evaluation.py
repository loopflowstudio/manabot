"""Evaluate the frozen size study through the ordinary replay-validated arena.

The input index contains verified recovered TrainingRun directories. Training
coordinates and checkpoint hashes are retained independently from arena cost.
A dedicated training seed supplies the fixed reference, never a selected winner.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from pydantic import BaseModel

from experiments.runners.head_to_head import (
    GREEDY,
    Entrant,
    Plan,
    Tournament,
    render,
    report,
    run,
)
from manabot.arena.models import file_sha256
from manabot.training.execution import atomic_json
from manabot.training.models import ArtifactReference, TrainingCoordinates, TrainingRun
from manabot.training.monitor_evaluation import stage_checkpoint


class RecoveredRun(BaseModel):
    name: str
    size: str
    seed: int
    directory: Path
    source: str


def admit_runs(
    index: Path, destination: Path, *, smoke: bool = False
) -> tuple[list[Entrant], list[Entrant]]:
    records = [
        RecoveredRun.model_validate(row) for row in json.loads(index.read_text())
    ]
    expected = {("small", 13560)} | {
        (size, seed)
        for size in ("small", "middle", "large")
        for seed in (13561, 13562, 13563)
    }
    if not smoke and {(r.size, r.seed) for r in records} != expected:
        raise ValueError("Incomplete scientific cohort; no comparison over survivors")
    if len({r.source for r in records}) != 1:
        raise ValueError("Training source differs within the cohort")
    models = destination / "models"
    models.mkdir(parents=True, exist_ok=False)
    monitoring, endpoint = [], []
    coordinates: dict[str, object] = {}
    identities: list[dict[str, str]] = []
    for record in records:
        found = TrainingRun.model_validate_json(
            (record.directory / "run.json").read_text()
        )
        if found.seed != record.seed or found.status != "completed":
            raise ValueError(f"Invalid completed run: {record.name}")
        if not smoke and found.updates_through() != 2048:
            raise ValueError("Training horizon differs")
        identities.append(
            {
                key: found.identities[key]
                for key in (
                    "engine_source_sha256",
                    "content_manifest_sha256",
                    "observation_abi_sha256",
                    "action_abi_sha256",
                    "matchup_sha256",
                )
            }
        )
        final = stage_checkpoint(found, "policy")
        if final is None:
            raise ValueError("Missing endpoint")
        checkpoints: list[tuple[ArtifactReference, TrainingCoordinates]] = [
            (final.artifact, final.coordinates)
        ]
        if record.seed != 13560:
            checkpoints += [
                (
                    found.stages[0].artifacts["initial_raw"],
                    TrainingCoordinates(
                        stage_id="policy", updates=0, training_seconds=0
                    ),
                )
            ]
            checkpoints += [
                (
                    c.artifact,
                    TrainingCoordinates.model_validate(
                        c.model_dump(exclude={"ordinal", "artifact", "error"})
                    ),
                )
                for c in found.monitoring_checkpoints
                if c.artifact is not None
            ]
            if not smoke and {c.updates for _, c in checkpoints} != {
                0,
                128,
                512,
                1024,
                2048,
            }:
                raise ValueError("Checkpoint schedule incomplete or changed")
        for artifact, coordinate in checkpoints:
            path = record.directory / Path(artifact["path"]).name
            if (
                path.stat().st_size != artifact["bytes"]
                or file_sha256(path) != artifact["sha256"]
            ):
                raise ValueError("Recovered checkpoint differs from TrainingRun")
            identifier = (
                "fixed-reference"
                if record.seed == 13560
                else f"{record.size}-{record.seed}-u{coordinate.updates}"
            )
            name = f"{identifier}.pt"
            shutil.copyfile(path, models / name)
            entrant = Entrant(
                identifier,
                identifier,
                record.name,
                name,
                artifact["sha256"],
                artifact["bytes"],
                record.seed,
                coordinate.updates,
            )
            coordinates[identifier] = {
                "coordinates": coordinate.model_dump(mode="json"),
                "run_id": found.id,
                "source": record.source,
                "size": record.size,
                "artifact": artifact,
            }
            if record.seed == 13560:
                monitoring.append(entrant)
                endpoint.append(entrant)
            elif coordinate.updates == found.updates_through():
                endpoint.append(entrant)
            else:
                monitoring.append(entrant)
    if any(value != identities[0] for value in identities):
        raise ValueError("World, inputs or matchup differ across training runs")
    atomic_json(destination / "training-coordinates.json", coordinates)
    return monitoring, endpoint


def evaluate(index: Path, destination: Path, *, smoke: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    monitoring, endpoints = admit_runs(index, destination, smoke=smoke)
    for name, entrants, start, deals in (
        ("monitoring", monitoring, 1916135000, 8),
        ("endpoint", endpoints, 1916136000, 25),
    ):
        reference = "fixed-reference"
        all_entrants = (Entrant(GREEDY, "Scripted greedy", "frozen source"), *entrants)
        extra = tuple((e.id, reference) for e in entrants if e.id != reference)
        spec = Tournament(
            entrants=all_entrants,
            core=(),
            extra=extra,
            contrasts=extra,
            reference=reference,
            training_caveat="Three paired training seeds; per-pair intervals resample deals, not independent training seeds. Fixed 2048-update horizon is not a skill ceiling.",
        )
        atomic_json(
            destination / f"{name}-tournament.json", spec.model_dump(mode="json")
        )
        plan = Plan(
            all_entrants,
            spec.pairings(),
            tuple(range(start, start + (1 if smoke else deals))),
            tournament=spec,
        )
        status = run(
            destination / name,
            destination / "models",
            plan=plan,
            workers=2,
            wall_seconds=180 if smoke else 6 * 3600,
        )
        atomic_json(destination / f"{name}-status.json", asdict(status))
        findings = report(destination / name)
        atomic_json(destination / f"{name}-report.json", asdict(findings))
        (destination / f"{name}-report.md").write_text(render(findings))
        if status.state != "completed":
            raise RuntimeError(f"Incomplete {name} evaluation: {status.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    evaluate(args.index, args.out, smoke=args.smoke)


if __name__ == "__main__":
    main()
