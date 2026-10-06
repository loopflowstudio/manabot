"""Declare capacity studies without executing them; the ordinary study runner executes.

Scientific counts come from a separately reviewed CPU calibration. The authoring
receipt and ResolvedStudy retain configuration and cohort authority respectively.
"""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from experiments.runners.model_capacity import experiment
from experiments.runners.training_protocol import EvaluationProtocol, ResolvedStudy
from manabot.arena.models import file_sha256
from manabot.training.experiments import Baseline, Case, Experiment, Pipeline, Resources
from manabot.training.models import Execution, TrainSelfPlay
from manabot.training.presets import ataraxos_mtg_v1


class CapacityWorkload(BaseModel):
    """Reviewed counts per five-minute unit, one per ordered capacity arm."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    updates_per_unit: tuple[int, int, int]
    calibration_path: Path
    calibration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    prior_campaign_seconds: float = Field(ge=0)
    runtime_identities: dict[str, str]
    projected_disk_bytes: int = Field(gt=0)
    projected_evaluation_seconds: float = Field(gt=0)


def declaration(workload: CapacityWorkload | None = None) -> Experiment:
    base = ataraxos_mtg_v1().regime()
    ladder = experiment(base)
    cases = []
    for index, case in enumerate(ladder.cases):
        units = workload.updates_per_unit[index] if workload else 1
        if units <= 0:
            raise ValueError("calibrated update counts must be positive")
        source = base.stages[0]
        assert isinstance(source, TrainSelfPlay)
        stages = tuple(
            source.model_copy(
                update={
                    "id": f"checkpoint-{i + 1}",
                    "initial": None if i == 0 else f"checkpoint-{i}",
                    "updates": units * factor,
                    "execution": Execution(
                        wall_seconds=300 * factor if workload else 30, threads=1
                    ),
                }
            )
            for i, factor in enumerate((1, 1, 2, 8) if workload else (1, 1))
        )
        cases.append(
            Case(case.name, (*case.overrides, Pipeline(stages=stages)), case.label)
        )
    return Experiment(
        name="capacity",
        baseline=Baseline.capture("ataraxos-capacity-v1", base),
        cases=tuple(cases),
        overrides=(Resources(wall_seconds=3660 if workload else 75),),
    )


def plan(workload: CapacityWorkload | None = None) -> ResolvedStudy:
    if (
        workload
        and file_sha256(workload.calibration_path) != workload.calibration_sha256
    ):
        raise ValueError("calibration evidence digest mismatch")
    if workload:
        required = {
            "engine_extension_sha256",
            "engine_source_sha256",
            "training_source_sha256",
            "content_manifest_sha256",
            "observation_abi_sha256",
            "action_abi_sha256",
            "matchup_sha256",
        }
        if set(workload.runtime_identities) != required or any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in workload.runtime_identities.values()
        ):
            raise ValueError(
                "calibration must bind all seven runtime/source identities"
            )
        if workload.projected_evaluation_seconds + 9 * 3660 + 1800 > 24 * 3600:
            raise ValueError(
                "projected evaluation exceeds the bounded allocation including report reserve"
            )
    cells = declaration(workload).resolve()
    scientific = workload is not None
    seeds = (1031, 1032, 1033) if scientific else (1031,)
    protocol = EvaluationProtocol(
        study="model-capacity",
        purpose="scientific" if scientific else "workflow-smoke",
        regime_digests=cells.digests,
        training_seeds=seeds,
        checkpoint_count=4 if scientific else 2,
        cost_cutoffs_seconds=(300, 600, 1200, 3600) if scientific else (),
        paired_deals=tuple(range(930100, 930125)) if scientific else (930100,),
        anchor_deals=tuple(range(931100, 931125)) if scientific else (931100,),
        endpoint_paired_deals=tuple(range(932100, 932125)) if scientific else (),
        endpoint_anchor_deals=tuple(range(933100, 933125)) if scientific else (),
        endpoint_seed_pairs=tuple((s, s) for s in seeds) if scientific else (),
        anchors=("random", "scripted-greedy", "puct-64") if scientific else ("random",),
        uncertainty="paired-seed-descriptive"
        if scientific
        else "cross-seed-unavailable",
        process_seconds=24 * 3600 if scientific else 900,
        early_progress_seconds=1200,
        progress_score=0.6,
    )
    return ResolvedStudy(
        protocol=protocol,
        recipes=tuple(r.model_dump(mode="json") for r in cells.regimes.values()),
        allocation_seconds=protocol.process_seconds,
        prior_campaign_seconds=workload.prior_campaign_seconds if workload else 0,
        runtime_identities=workload.runtime_identities if workload else {},
        projected_disk_bytes=workload.projected_disk_bytes if workload else 0,
        calibration_evidence=(
            workload.model_dump_json()
            if workload
            else "Unexecuted software plan; no calibration or scientific allocation"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-plan", type=Path, required=True)
    parser.add_argument("--workload", type=Path)
    args = parser.parse_args()
    workload = (
        CapacityWorkload.model_validate_json(args.workload.read_text())
        if args.workload
        else None
    )
    resolved = plan(workload)
    receipt = args.write_plan.with_suffix(".provenance.json")
    if args.write_plan.exists() or receipt.exists():
        raise FileExistsError("plan and provenance destinations must be new")
    args.write_plan.write_text(resolved.model_dump_json(indent=2) + "\n")
    receipt.write_text(
        json.dumps(declaration(workload).resolve().receipt(), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
