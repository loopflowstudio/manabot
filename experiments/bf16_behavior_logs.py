"""Exploratory MTG-345 subnormal-exp and fixed-weight forward probe, then BF16 proof.

One capped Etude job runs `probe` (writes probe.json), then the ordinary
training worker on precision_ladder's BF16 recipe with its original 300-update
horizon and a shorter wall window. `smoke` runs the same probe paths and one
real worker update on CPU with a small model; it makes no CUDA claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile

from etude.launch.records import Run
from etude.report import Report
import numpy as np
import torch

from experiments.precision_ladder import arm
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.experiment.job import TrainingJob, run_worker
from manabot.infra.config import RewardConfig
from manabot.recipes.local import local_smoke
from manabot.training.panels import ML, PERF

ROOT = Path(__file__).resolve().parents[1]
SEED = 32401
# Setup, probe and final exports fit inside the rental's 6.99 hours. The regime
# retains the 300-update schedule; the ordinary wall-bound stop exports it.
HOURS = 6.99
TRAIN_SECONDS = HOURS * 3600 - 1200


@dataclass(frozen=True)
class ForwardDifference:
    precision: str
    live_tokens: bool
    bucket_scale: int
    buckets: dict[str, int]
    rows: int
    median: float
    p95: float
    p99: float
    maximum: float


def exp_probe(device: torch.device, *, smoke: bool) -> dict[str, object]:
    """Compare float32 device/host exp, retaining signed spacing differences.

    Negative float32 bit patterns are monotone in the opposite direction. Each
    integer in this interval therefore enumerates one representable input.
    Positive exp results have monotone bit patterns even across zero/subnormal.
    """
    low, high = np.float32(-104), np.float32(-87.3)
    bits = np.arange(
        high.view(np.uint32), int(low.view(np.uint32)) + 1, dtype=np.uint32
    )
    if smoke:
        bits = bits[::1024]
    logs = torch.from_numpy(bits.view(np.float32).copy())
    host = logs.exp()
    other = logs.to(device).exp().cpu()
    gap = other.view(torch.int32).long() - host.view(torch.int32).long()
    values, counts = gap.unique(return_counts=True)
    old_rejects = ~torch.isclose(host, other, atol=0, rtol=1e-5)
    return {
        "inputs": logs.numel(),
        "exhaustive": not smoke,
        "spacing_gaps_device_minus_host": dict(
            zip(map(str, values.tolist()), counts.tolist(), strict=True)
        ),
        "old_rule_rejects": int(old_rejects.sum()),
        "device_zero_count": int((other == 0).sum()),
    }


def _observations(
    space: ObservationSpace, match: Match, count: int
) -> dict[str, torch.Tensor]:
    """A bounded native trajectory; the policy sees only the acting viewer."""
    env = Env(match, space, Reward(RewardConfig()), seed=SEED)
    rng = np.random.default_rng(SEED)
    kept: list[dict[str, np.ndarray]] = []
    try:
        obs, _ = env.reset()
        for _ in range(count):
            kept.append({key: value.copy() for key, value in obs.items()})
            action = int(rng.choice(np.flatnonzero(obs["actions_valid"])))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
    return {
        key: torch.from_numpy(np.stack([row[key] for row in kept])) for key in kept[0]
    }


def forward_probe(device: torch.device, *, smoke: bool) -> list[ForwardDifference]:
    """Collection inference versus differentiable learner forward, same weights.

    The larger learner bucket pads the same rows with unused capacity; no
    observation or choice changes. Each learner chunk is admitted separately,
    as in training, and uses a fresh catalog graph rather than inference's cache.
    These deltas are measurements, not an admission tolerance.
    """
    raise RuntimeError("Run this frozen probe from revision 2284e4400c.")


def probe(output: Path, *, smoke: bool = False) -> None:
    device = torch.device("cpu" if smoke else "cuda")
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError("proof requires CUDA")
    torch.set_num_threads(1 if smoke else min(6, os.cpu_count() or 1))
    result = {
        "device": str(device),
        "gpu": None if smoke else torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "exp": exp_probe(device, smoke=smoke),
        "fixed_weights": [asdict(row) for row in forward_probe(device, smoke=smoke)],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def runs(revision: str) -> list[Run]:
    """One L4 proof, with the USD 5 ceiling on the job itself.

    The probe is its own process ahead of the unmodified training worker, so
    training starts from a fresh interpreter and binds its source exactly once.
    """
    recipe = arm("bf16")
    recipe.wall_seconds = TRAIN_SECONDS
    recipe.stages[0].execution.wall_seconds = TRAIN_SECONDS
    job = TrainingJob(
        regime=recipe,
        experiment=Path(__file__).stem,
        milestones=(140, 141, 200),
        finalization_seconds=300,
    ).job(repo=ROOT, revision=revision, max_hours=HOURS, max_usd=Decimal("5"))
    setup, train = job.command.split("\n")
    probe_command = "uv run --locked python -m experiments.bf16_behavior_logs probe"
    job = job.model_copy(update={"command": f"{setup}\n{probe_command} && {train}"})
    return job.runs(SEED)


def report() -> Report:
    return Report(perf=PERF, ml=ML)


def smoke(output: Path) -> None:
    """The probe, then one real training update through local SDK worker inputs."""
    output.mkdir(parents=True, exist_ok=True)
    spec = TrainingJob(
        regime=local_smoke(steps=1, device="cpu"), experiment=Path(__file__).stem
    )
    config = output / "config.json"
    config.write_text(spec.model_dump_json())
    for key in [key for key in os.environ if key.startswith("ETUDE_")]:
        del os.environ[key]
    os.environ.update(
        ETUDE_RUN="bf16-logs-smoke",
        ETUDE_TRACK_DIR=str(output / "tracking"),
        ETUDE_CONFIG=str(config),
        ETUDE_OUTPUT_DIR=str(output / "output"),
        ETUDE_STOP_FILE=str(output / "stop"),
        ETUDE_SEED=str(SEED),
        ETUDE_UNTIL_CLOCK="updates",
        ETUDE_UNTIL_VALUE=str(spec.updates),
        ETUDE_DEADLINE=(
            datetime.now(timezone.utc) + timedelta(seconds=180)
        ).isoformat(),
    )
    probe(output / "output", smoke=True)
    run_worker()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "probe"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "probe":
        probe(Path(os.environ["ETUDE_OUTPUT_DIR"]))
    else:
        output = args.out or Path(tempfile.mkdtemp(prefix="bf16-logs-smoke-"))
        smoke(output.resolve())
        print(output)


if __name__ == "__main__":
    main()
