"""Admission and bounded rentals for ETU-103's two-card performance calibration."""

import argparse
import json
from pathlib import Path
import time

from manabot.remote.deploy import Receipt, current_source, deploy
from manabot.remote.plan import JobSpec, compile_plan
from manabot.remote.provider import RunPod
from manabot.remote.transport import REPO_DIR, Transport
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRegime, TrainSelfPlay


def launch(root: Path, gpu: str) -> None:
    root = root.resolve()
    original = Receipt.model_validate_json(
        (root / "calibration-small/deployment.json").read_text()
    )
    receipts = [
        Receipt.model_validate_json(p.read_text())
        for p in root.glob("calibration-*/deployment.json")
    ]
    if any(r.phase != "deleted" or r.estimated_dollars is None for r in receipts):
        raise ValueError("prior rental deletion/cost unresolved")
    spent = sum(r.estimated_dollars or 0 for r in receipts)
    remaining = 7200 - (time.time() - original.started)
    if remaining < 2100:
        raise ValueError(
            f"only {remaining:.1f} calibration seconds remain; this job requires 2100"
        )
    if spent + 0.62 * 2100 / 3600 + 1 > 15:
        raise ValueError(
            "sweep would exceed initial all-in ceiling with storage/report reserve"
        )
    source = current_source(Path.cwd())
    base = TrainingRegime.model_validate_json(
        (root / "inputs/cuda-calibration-small.json").read_text()
    )
    base.stages = base.stages[:1]
    base.schedule_clock = "iteration_fraction"
    stage = base.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates, stage.transitions, stage.execution.wall_seconds = 2, 64, 90
    base.wall_seconds = 100
    base = TrainingRegime.model_validate(base.model_dump())
    template = JobSpec.model_validate_json((root / "inputs/job.json").read_text())
    spec = JobSpec.model_validate(
        template.model_dump()
        | {
            "machine": template.machine.model_dump() | {"gpu_types": [gpu]},
            "lifetime_hours": 2100 / 3600,
            "setup_seconds": 420,
            "checkpoint_seconds": 120,
            "upload_seconds": 360,
            "cleanup_seconds": 120,
        }
    )
    plan = compile_plan(base.model_dump_json(), spec, source, 10349)
    suffix = "l4" if gpu == "NVIDIA L4" else "a40"
    out = root / f"calibration-{suffix}-sweep"
    quote = RunPod().prices().get(gpu)
    atomic_json(
        root / f"{suffix}-sweep-admission.json",
        {
            "observed_unix": time.time(),
            "calibration_remaining_seconds": remaining,
            "prior_rental_estimate_dollars": spent,
            "quote_dollars_per_hour": quote,
            "projection_dollars": spec.projected_dollars,
            "source": source.model_dump(mode="json"),
            "cell_deadline_seconds": 70,
            "sweep_deadline_seconds": 900,
            "input_manifest": json.loads(
                (root / "inputs/observations.json").read_text()
            ),
        },
    )

    def probe(transport: Transport) -> None:
        transport.shell("mkdir -p /workspace/evidence/performance")
        transport.put(
            root / "inputs/observations.pt",
            "/workspace/evidence/performance/observations.pt",
        )
        baseline = root / f"{suffix}-sweep-baseline.json"
        baseline.write_text(base.model_dump_json(indent=2))
        transport.put(baseline, "/workspace/evidence/performance/baseline.json")
        transport.shell(
            f"export PATH=/root/.local/bin:/root/.cargo/bin:$PATH\ncd {REPO_DIR}\nuv run --no-sync python -m experiments.runners.cuda_performance --root /workspace/evidence/performance --seconds 900"
        )

    result = deploy(plan, out, Path.cwd(), after_training=probe)
    print(
        f"{gpu}: {result.phase}; complete={result.complete}; estimated dollars={result.estimated_dollars}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", choices=("NVIDIA L4", "NVIDIA A40"), required=True)
    args = parser.parse_args()
    launch(args.root, args.gpu)


if __name__ == "__main__":
    main()
