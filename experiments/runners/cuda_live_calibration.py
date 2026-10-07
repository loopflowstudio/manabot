"""Bounded live-export/CPU-arena proof inside the original CUDA calibration clock.

This is timing and integration evidence with one four-leg deal, never scientific
capacity scoring. The original policy/world/setup admission and rental lifecycle
remain authoritative. No scores select the subsequent workload.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Literal

from pydantic import BaseModel, Field

from manabot.remote.deploy import Receipt, current_source, deploy
from manabot.remote.plan import JobSpec, compile_plan
from manabot.remote.progress import LiveExports
from manabot.remote.transport import REPO_DIR, Transport
from manabot.training.checkpoint_queue import CheckpointQueue, MonitoringBudget
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRegime, TrainingRun, TrainSelfPlay
from manabot.training.monitor_evaluation import MonitorProtocol


class LiveObservation(BaseModel):
    at: float
    run_status: str
    updates: int
    attempts: int


class LiveCalibrationReceipt(BaseModel):
    status: Literal["completed"]
    seconds: float = Field(gt=0, le=1250)
    evaluator_seconds: float = Field(gt=0, le=125)
    rental_estimated_dollars: float = Field(ge=0)
    observations: list[LiveObservation]
    purpose: str


def launch(root: Path, gpu: str, streams: int, batch: int) -> None:
    root = root.resolve()
    receipts = [
        Receipt.model_validate_json(p.read_text())
        for p in root.glob("calibration-*/deployment.json")
    ]
    amendment = json.loads((root / "calibration-amendment.json").read_text())
    if (
        amendment["calibration_seconds"] != 8700
        or amendment["combined_seconds"] != 50400
    ):
        raise ValueError("calibration amendment does not match bounded reallocation")
    remaining = 8700 - (time.time() - min(r.started for r in receipts))
    if remaining < 1250 or any(
        r.phase != "deleted" or r.estimated_dollars is None for r in receipts
    ):
        raise ValueError(
            "live workflow proof cannot fit remaining calibration or prior cleanup unresolved"
        )
    out = root / "calibration-live-workflow"
    control = root / "live-workflow-control"
    control.mkdir()
    base = TrainingRegime.model_validate_json(
        (root / "inputs/cuda-calibration-small.json").read_text()
    )
    base.id = "cuda-live-calibration"
    base.wall_seconds = 360
    base.stages = base.stages[:1]
    base.schedule_clock = "iteration_fraction"
    for stage in base.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.updates, stage.streams, stage.transitions = 128, streams, batch // streams
        stage.execution.wall_seconds = 350
    template = JobSpec.model_validate_json((root / "inputs/job.json").read_text())
    spec = JobSpec.model_validate(
        template.model_dump()
        | {
            "machine": template.machine.model_dump() | {"gpu_types": [gpu]},
            "lifetime_hours": 1100 / 3600,
            "setup_seconds": 420,
            "checkpoint_seconds": 120,
            "upload_seconds": 80,
            "cleanup_seconds": 120,
        }
    )
    plan = compile_plan(base.model_dump_json(), spec, current_source(Path.cwd()), 10350)
    if (
        sum(r.estimated_dollars or 0 for r in receipts) + plan.projected_dollars + 1
        > 15
    ):
        raise ValueError("aggregate dollar ceiling exceeded")
    live = LiveExports(control / "live")
    queue = CheckpointQueue(
        control / "evaluation",
        MonitoringBudget(
            seconds=120,
            attempt_seconds=120,
            include_initial=True,
            protocol=MonitorProtocol(deal_seeds=(1910103500,), game_seconds=60),
        ),
        resolve_artifact=live.resolve,
        protocols_for=lambda r, c: (
            [MonitorProtocol(deal_seeds=(1910103500,), game_seconds=60)]
            if c.coordinates.updates == 0
            else []
        ),
    )
    observations: list[LiveObservation] = []
    began = time.time()
    try:

        def observe(transport: Transport) -> None:
            source = live.retrieve(transport)
            if source is not None:
                run = TrainingRun.model_validate_json(source.read_text())
                queue.tick([source])
                observations.append(
                    LiveObservation(
                        at=time.time(),
                        run_status=run.status,
                        updates=run.updates_through(),
                        attempts=len(queue.attempts),
                    )
                )
                atomic_json(
                    control / "observations.json",
                    [o.model_dump() for o in observations],
                )

        def profile_collection(transport: Transport) -> None:
            transport.shell(
                f"export PATH=/root/.local/bin:/root/.cargo/bin:$PATH\ncd {REPO_DIR}\nuv run --no-sync python -m experiments.runners.cuda_collection_profile --recipe /workspace/regime.json --out /workspace/evidence/collection-profile --seconds 180 > /workspace/evidence/collection-profile.log 2>&1\nprintf '%s\\n' \"$?\" > /workspace/evidence/collection-profile-exit.txt\nexit 0",
                observe=lambda: (
                    queue.tick([live.latest]) if live.latest is not None else None
                ),
            )
            transport.shell(
                "mkdir -p /workspace/evidence/fit512/w384-d8-model-b512-s0; cp /workspace/regime.json /workspace/evidence/fit512/baseline.json"
            )
            transport.put(
                root / "inputs/observations.pt",
                "/workspace/evidence/fit512/observations.pt",
            )
            transport.shell(
                f'export PATH=/root/.local/bin:/root/.cargo/bin:$PATH\ncd {REPO_DIR}\ntimeout 70 uv run --no-sync python -m experiments.runners.cuda_performance --root /workspace/evidence/fit512 --cell \'{{"capacity":"w384-d8","kind":"model","batch":512,"streams":0}}\' > /workspace/evidence/fit512/worker.log 2>&1\nprintf \'%s\\n\' "$?" > /workspace/evidence/fit512/exit.txt\nexit 0',
                observe=lambda: (
                    queue.tick([live.latest]) if live.latest is not None else None
                ),
            )

        receipt = deploy(
            plan,
            out,
            Path.cwd(),
            observe=observe,
            checkpoint_seconds=3600,
            bulk_return=True,
            after_training=profile_collection,
        )
        while queue.process is not None:
            if time.time() - began > 1250:
                raise TimeoutError("live proof ceiling exhausted")
            assert live.latest is not None
            queue.tick([live.latest])
            time.sleep(2)
        if (
            not receipt.complete
            or len(queue.attempts) != 1
            or queue.attempts[0].status != "completed"
        ):
            raise ValueError("live workflow proof failed")
        if not any(o.run_status == "running" and o.attempts == 1 for o in observations):
            raise ValueError("no evaluation began while TrainingRun was running")
        atomic_json(
            control / "receipt.json",
            LiveCalibrationReceipt(
                status="completed",
                seconds=time.time() - began,
                evaluator_seconds=queue.charged_seconds,
                rental_estimated_dollars=receipt.estimated_dollars,
                observations=observations,
                purpose="timing-and-live-integration-only; no score used",
            ).model_dump(mode="json"),
        )
    finally:
        queue.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", choices=("NVIDIA L4", "NVIDIA A40"), required=True)
    parser.add_argument("--streams", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    launch(args.root, args.gpu, args.streams, args.batch)


if __name__ == "__main__":
    main()
