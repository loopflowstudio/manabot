"""Read frozen attack results and report their paired learning curves.

Execution is retired; its entry point names the last runnable revision.
"""

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from manabot.experiment.attacks import AttackPlan
from manabot.training.regime import Strict


class AttackPoint(Strict):
    target: str
    seed: int
    updates: int
    training_seconds: float
    evaluation_seconds: float = 0
    score: float | None = None
    games: int = 0
    replay_passed: bool = False
    evidence_path: str


class AttackResult(Strict):
    plan_digest: str
    status: Literal["running", "completed", "failed"] = "running"
    runs: list[str] = Field(default_factory=list)
    run_paths: list[str] = Field(default_factory=list)
    points: list[AttackPoint] = Field(default_factory=list)
    seconds: float = 0
    error: str | None = None


def _report(plan: AttackPlan, result: AttackResult, out: Path) -> None:
    lines = [
        "# Frozen-policy attack results",
        "",
        f"Status: {result.status}. Total elapsed: {result.seconds:.2f} seconds.",
        "",
        "A failed attack does not certify equilibrium or exact exploitability.",
        "",
        "| Target | Attacker seed | Updates | Training seconds | Score | Games | Replay |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for point in result.points:
        score = "unavailable" if point.score is None else f"{point.score:.4f}"
        lines.append(
            f"| {point.target} | {point.seed} | {point.updates} | {point.training_seconds:.2f} | {score} | {point.games} | {point.replay_passed} |"
        )
    if result.status == "completed":
        lines += [
            "",
            "## Endpoint improvement",
            "",
            "Exploratory 95% percentile bootstrap intervals resample attacker seeds, not individual games. Targets are reported separately; these intervals do not include uncertainty across target-policy training seeds.",
            "",
        ]
        for target in plan.targets:
            changes = []
            for seed in plan.attacker_seeds:
                points = [
                    p for p in result.points if p.target == target.id and p.seed == seed
                ]
                first, last = points[0], points[-1]
                assert first.score is not None and last.score is not None
                changes.append(last.score - first.score)
            rng = np.random.default_rng(937)
            values = np.asarray(changes)
            samples = rng.choice(values, size=(2000, len(values)), replace=True).mean(
                axis=1
            )
            low, high = np.quantile(samples, [0.025, 0.975])
            lines.append(
                f"- {target.id}: mean paired improvement {values.mean():.4f}, interval [{low:.4f}, {high:.4f}], {len(values)} independent attacker seeds."
            )
    if result.error:
        lines += ["", f"Failure: {result.error}"]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def execute_attack_plan(*args: object, **kwargs: object) -> None:
    """Retired frozen execution entry point; retained data and evaluation still read."""
    raise RuntimeError(
        "Frozen execution requires revision f09f5c2919b8b1a5cf6d72c4f4f4721dac68f653"
    )


def main() -> None:
    """`uv run python -m manabot.experiment.attack_execution --plan PLAN --out DIR`."""
    parser = argparse.ArgumentParser(description="Execute a frozen-policy attack plan.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    execute_attack_plan(AttackPlan.model_validate_json(args.plan.read_text()), args.out)


if __name__ == "__main__":
    main()
