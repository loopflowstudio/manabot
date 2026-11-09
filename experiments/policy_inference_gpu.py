"""Measure trained-policy self-play through the existing GPU evaluation runner.

Both seats use independently seeded registrations of MTG-183's final checkpoint.
The job compares resident CUDA batch-1 with four-game batching, then replays
identical retained decision roots. Etude owns the rental; the baseline runner
owns bounded workers and timing. Collection preserves interrupted work too.
"""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import tarfile
import time

from etude.launch.records import Job
from pydantic import BaseModel, Field

from experiments import inference_baseline as baseline, qwen_mtg183 as inputs
from manabot.eval.local import checkpoint_request
from manabot.infra.artifacts import verify_file
from manabot.sim.execution import EvalAllocation
from manabot.sim.matchup import AuthoredMatchup


class Campaign(BaseModel):
    repetitions: int = Field(default=3, ge=1, le=3)
    phase_seconds: float = Field(default=900, gt=0, le=900)
    batch_delay_seconds: float = Field(default=0.005, ge=0, le=0.005)


def prepare(root: Path, checkpoint: Path, config: Campaign) -> baseline.Recipe:
    """Admit real checkpoint bytes before freezing the GPU measurement recipe."""
    import torch

    torch.set_num_threads(1)
    verify_file(checkpoint, inputs.CHECKPOINT_SHA256, inputs.CHECKPOINT_BYTES)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    matchup = AuthoredMatchup(
        pack_key="elemental-world-v2", role_a="elemental_ur", role_b="elemental_gw"
    )
    workloads: list[baseline.PolicyWorkload] = []
    for name, seeds, legs in (
        ("single", (18801,), (0,)),
        ("many", (18801, 18802, 18803, 18804), (0, 1)),
    ):
        request = checkpoint_request(
            checkpoint,
            opponent=checkpoint,
            deal_seeds=seeds,
            authored_matchup=matchup,
            game_seconds=120,
            max_commands=10_000,
        )
        path = root / f"{name}-request.json"
        baseline._write(path, request)
        workloads.append(
            baseline.PolicyWorkload(
                id=name, kind="policy", request=path, legs=legs, roots=64
            )
        )
    for size in (1, 4):
        baseline._write(
            root / f"cuda-{size}.json",
            EvalAllocation(
                device="cuda",
                max_active_games=size,
                max_batch_rows=size,
                max_delay_seconds=config.batch_delay_seconds if size > 1 else 0,
            ),
        )
    baseline._write(
        root / "admission.json",
        {
            "seconds": time.monotonic() - started,
            "checkpoint_sha256": inputs.CHECKPOINT_SHA256,
            "boundary": "CPU checkpoint/setup admission before GPU cells; included in job wall",
            "opponent": "same trained checkpoint, distinct player registration and policy RNG",
        },
    )
    return baseline.Recipe(
        workloads=tuple(workloads),
        cells=(
            baseline.Cell(
                id="cuda-1-single", workload="single", execution=root / "cuda-1.json"
            ),
            baseline.Cell(
                id="cuda-1-many", workload="many", execution=root / "cuda-1.json"
            ),
            baseline.Cell(
                id="cuda-4-many", workload="many", execution=root / "cuda-4.json"
            ),
        ),
        repetitions=config.repetitions,
        cell_seconds=300,
        total_seconds=config.phase_seconds,
        sample_device=True,
    )


def collect(out: Path) -> None:
    """Publish one archive, avoiding thousands of loose experiment outputs."""
    root = out.parent / "policy-inference"
    if not root.exists():
        return
    temporary = out / "policy-inference.pending"
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        archive.add(root, arcname="policy-inference")
    temporary.replace(out / "policy-inference.tar.gz")


def replay_recipe(recipe: baseline.Recipe, reference: Path) -> baseline.Recipe:
    """Resolve the portable trace receipt relative to its owning cell."""
    trace = json.loads((reference / "trace.json").read_text())
    artifact = trace.get("artifact_path")
    if not isinstance(artifact, str) or not artifact:
        raise ValueError("completed cell has no trace artifact path")
    workload = recipe.workloads[1].model_copy(update={"trace": reference / artifact})
    return recipe.model_copy(
        update={"workloads": (workload,), "cells": recipe.cells[1:]}
    )


def run(config: Campaign) -> None:
    import torch

    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    root = out.parent / "policy-inference"
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU fallback measurement")
        recipe = prepare(
            root, Path(os.environ["ETUDE_INPUTS_DIR"]) / "checkpoint.pt", config
        )
        baseline._write(root / "recipe.json", recipe)
        protocol = baseline.freeze(recipe)
        baseline._write(root / "protocol.json", protocol)
        print(
            "GPU policy self-play: 9 reserved cells, CUDA batch-1 versus four games/rows",
            flush=True,
        )
        games = baseline.run(protocol, root / "games")
        baseline._write(out / "game-attempts.json", games)
        for attempt in games.attempts:
            print(
                f"GAME CELL {attempt.id}: {attempt.status}: {attempt.reason}",
                flush=True,
            )
        # A completed many-game trace supplies the same real decisions to both
        # batch sizes. Never fabricate roots or convert replay into game throughput.
        reference = next(
            (
                a
                for a in games.attempts
                if a.cohort == "many" and a.status == "completed"
            ),
            None,
        )
        if reference is None or reference.root is None:
            raise RuntimeError(
                "no completed many-game cell; retained-input comparison unavailable"
            )
        replay_protocol = baseline.freeze(replay_recipe(recipe, Path(reference.root)))
        baseline._write(root / "replay-protocol.json", replay_protocol)
        replay = baseline.run(replay_protocol, root / "replay", replay=True)
        baseline._write(out / "replay-attempts.json", replay)
        for attempt in replay.attempts:
            print(
                f"INPUT CELL {attempt.id}: {attempt.status}: {attempt.reason}",
                flush=True,
            )
        if any(a.status != "completed" for a in (*games.attempts, *replay.attempts)):
            raise RuntimeError("measurement incomplete; all attempt outcomes retained")
    finally:
        collect(out)
        baseline._write(
            out / "job-wall.json",
            {
                "seconds": time.monotonic() - started,
                "boundary": "entry through admission, execution, replay and archive; excludes Etude setup/delivery and this final write",
            },
        )


def runs(native_artifact: str, native_source: str) -> Job:
    # Reuse the already verified Linux engine/checkpoint input contract, not the
    # Qwen command or environment setup. No language-model files are acquired.
    job = inputs.runs(native_artifact, native_source)
    return job.model_copy(
        update={
            "name": "mtg188-policy-v-policy-cuda",
            "command": "uv run --no-sync python -m etude.call --function experiments.policy_inference_gpu:run",
            "config": Campaign().model_dump_json(),
            "setup": "sh scripts/policy-inference-setup.sh",
            "cleanup": "uv run --no-sync python -m experiments.policy_inference_gpu --collect",
            "max_hours": 1,
            "max_usd": Decimal("1.00"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", required=True)
    parser.parse_args()
    collect(Path(os.environ["ETUDE_OUTPUT_DIR"]))


if __name__ == "__main__":
    main()
