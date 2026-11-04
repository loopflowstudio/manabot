"""Same-input state encoder comparison; declaration does not allocate compute.

Both arms run the existing complete-game self-play executor and checkpoint path.
The v7 protocol is experiments/protocols/state-representation-5da5f4-v7.md.
Original v5 declarations and excluded canaries remain bound to their historical
source; this prospective cohort has distinct identities and never admits them.
"""

import argparse
from dataclasses import dataclass
from decimal import Decimal
import math
from pathlib import Path
import statistics
import time
from typing import Literal

from etude.launch.records import Run
from pydantic import BaseModel, ConfigDict

from experiments.batched_baseline import batched_baseline
from manabot.eval.monitor import MonitorProtocol, MonitorResult
from manabot.experiment.checkpoint_evaluation import (
    evaluate_checkpoint,
    stage_checkpoint,
)
from manabot.experiment.job import TrainingJob
from manabot.infra.identity import canonical_sha256
from manabot.recipes.local import local_smoke
from manabot.sim.execution import EvalExecution
from manabot.training.records import TrainingRun
from manabot.training.regime import ArtifactReference, TrainingRegime, TrainSelfPlay

Arm = Literal["typed_pool", "typed_gru"]
ARMS: tuple[Arm, ...] = ("typed_pool", "typed_gru")
SEEDS = (73201, 73202, 73203, 73204)


def recipe(arm: Arm, *, smoke: bool = False) -> TrainingRegime:
    """Same v7 inputs; jointly compare effect and continuation token reducers."""
    regime = local_smoke(steps=1, seconds=120) if smoke else batched_baseline()
    regime.id = f"state-representation-v7-{arm}" + ("-smoke" if smoke else "")
    regime.agent.state_encoder = arm
    regime.agent.policy_film = True
    regime.agent.recent_events = False
    regime.observation.policy_input_version = 7
    regime.observation.max_facts = 128
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.learning_seats = "both"
    if not smoke:
        stage.games_per_stream = 1
        stage.updates = 128
        stage.streams = 32
        # Setup and finalization must also fit the rental's bounded window.
        stage.execution.wall_seconds = 4200
        regime.wall_seconds = 4200
    return TrainingRegime.model_validate(regime.model_dump())


def runs(revision: str, canary: bool = False) -> list[Run]:
    """Eight declared runs, no submission; each is bounded by its job allocation.

    Actual rental price and cohort controls must be inspected before launch.
    This module does not authorize a default machine or the existing pilot pool.
    """
    result: list[Run] = []
    regimes = {arm: recipe(arm) for arm in ARMS}
    if canary:
        for regime in regimes.values():
            regime.id += "-canary"
            stage = regime.stages[0]
            assert isinstance(stage, TrainSelfPlay)
            stage.updates = 1
            stage.execution.wall_seconds = regime.wall_seconds = 300
    jobs = {
        arm: TrainingJob(
            regime=regimes[arm],
            experiment="state_representation_v7",
            milestones=(),
            finalization_seconds=60 if canary else 120,
        ).job(
            repo=Path(__file__).resolve().parents[1],
            revision=revision,
            max_hours=0.2 if canary else 1.25,
            max_usd=Decimal("0.90") if canary else Decimal("1.75"),
        )
        for arm in ARMS
    }
    # Interleave treatments within each seed to reduce machine/time confounds.
    for index, seed in enumerate((73200,) if canary else SEEDS):
        for arm in (*ARMS[index % 2 :], *ARMS[: index % 2]):
            result.extend(jobs[arm].runs(seed))
    return result


def evaluate_run(run_path: Path, checkpoint_path: Path, out: Path) -> MonitorResult:
    """Score only the fixed endpoint, through existing admission/play/replay owners."""
    run = TrainingRun.model_validate_json(run_path.read_bytes())
    arm = run.regime.agent.state_encoder
    if (
        run.status != "completed"
        or run.seed not in SEEDS
        or run.regime_digest != canonical_sha256(recipe(arm).model_dump(mode="json"))
        or run.updates_through() != 128
    ):
        raise ValueError(
            "evaluation requires a declared seed and completed frozen recipe"
        )
    checkpoint = stage_checkpoint(run, "self-play")
    if checkpoint is None:
        raise ValueError("completed run has no admitted final raw checkpoint")
    # Relocate fetched bytes without altering their retained hash or producer.
    artifact: ArtifactReference = {
        **checkpoint.artifact,
        "path": str(checkpoint_path.resolve()),
    }
    return evaluate_checkpoint(
        run,
        artifact,
        checkpoint.coordinates,
        out,
        protocol=evaluation_protocol(),
        concurrent_activity="ordinary local evaluation; remote training may be active",
        execution=EvalExecution(
            device="cpu",
            max_active_games=2,
            max_batch_rows=16,
            resident_model_bytes=1024**3,
            deadline_unix=time.time() + 1800,
        ),
    )


def evaluation_protocol() -> MonitorProtocol:
    """Same held-out deals, both decks/seats; no failed-game retries or selection."""
    return MonitorProtocol(
        purpose="frozen-study-evaluation",
        study_protocol_sha256=canonical_sha256(SCORING),
        deal_seeds=tuple(range(84001, 84033)),
        opponent="heuristic",
        game_seconds=120,
        max_commands=10_000,
        bootstrap_seed=84000,
        bootstrap_replicates=2000,
    )


# Frozen with the final scientific source commit, after the excluded canary.
SCORING = {
    "name": "same-v7-input-state-representation",
    "arms": ARMS,
    "training_seeds": SEEDS,
    "endpoint": "final-raw-update-128",
    "games_per_seed": 4096,
    "streams": 32,
    "input": "v7 shared effects; 128 local facts",
    "treatment": "sum versus GRU in effects and typed continuations; no temporal memory",
    "deals": tuple(range(84001, 84033)),
    "legs": "both authored deck assignments and both seats",
    "opponent": "heuristic-at-training-source",
    "compute": "CPU float32; two active games; batch16; catalog cache; 1GiB residency",
    "deadline_seconds": 1800,
    "game_seconds": 120,
    "max_commands": 10_000,
    "primary": "typed_gru minus typed_pool paired seed score",
    "prediction": "mean >= 0.03; training cost ratio <= 1.25",
    "analysis": "95% Student-t interval over four paired seed means, df3; effects and continuation change jointly",
    "selection": "only if primary lower > 0, mean >= 0.03 and measured cost ratio <= 1.25",
    "failure": "all scheduled games and seeds required; no retries or best-checkpoint selection",
}


class ArmEvaluation(BaseModel):
    """One treatment's retained evaluation; the training seed remains in its result."""

    model_config = ConfigDict(extra="forbid")
    arm: Arm
    result: MonitorResult


@dataclass(frozen=True)
class PairedEstimate:
    differences: tuple[float, ...]
    mean: float
    lower: float
    upper: float


def paired_interval(differences: tuple[float, ...]) -> PairedEstimate:
    """The four training seeds, never individual games, are the inferential units.

    The df=3 t critical value is 3.182446305; the small-sample normal-difference
    assumption is explicit. This interval is not a claim that four seeds suffice.
    """
    if len(differences) != len(SEEDS) or not all(math.isfinite(d) for d in differences):
        raise ValueError(
            "the paired interval requires all four finite seed differences"
        )
    mean = statistics.mean(differences)
    radius = (
        3.182446305284263 * statistics.stdev(differences) / math.sqrt(len(differences))
    )
    return PairedEstimate(differences, mean, mean - radius, mean + radius)


def compare(results: list[ArmEvaluation]) -> PairedEstimate:
    """Reject missing/duplicate/incomplete attempts rather than select successful seeds."""
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    found: dict[tuple[Arm, int], float] = {}
    for item in results:
        result = item.result
        key = item.arm, result.training_seed
        result.validate_cohort()
        if (
            key not in expected
            or key in found
            or result.status != "completed"
            or result.protocol != evaluation_protocol()
            or result.score is None
            or result.regime_digest
            != canonical_sha256(recipe(item.arm).model_dump(mode="json"))
            or result.coordinates.updates != 128
        ):
            raise ValueError(
                "evaluation is duplicate, incomplete or outside the frozen cohort"
            )
        found[key] = result.score.mean
    if set(found) != expected:
        raise ValueError("every declared arm and independent seed must be present")
    return paired_interval(
        tuple(found["typed_gru", seed] - found["typed_pool", seed] for seed in SEEDS)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_run(args.run, args.checkpoint, args.out)
    print(result.status, "games", len(result.rows), "score", result.score)
    return int(result.status != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
