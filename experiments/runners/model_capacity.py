"""Declare registered capacity cases; workload and evaluation stay with the caller.

No accounting/calibration or training happens here. Existing case IDs and complete
scientific configurations are preserved; these are not checkpoint weight transfers.
"""

from manabot.infra.hypers import AgentSpec
from manabot.training.experiments import Baseline, Case, Experiment, Model
from manabot.training.models import TrainingRegime


def experiment(base: TrainingRegime) -> Experiment:
    return Experiment(
        name="",
        baseline=Baseline.capture("capacity-control-v1", base),
        cases=(
            Case(
                "w64-d1",
                (
                    Model(
                        AgentSpec(
                            hidden_dim=64, attention_layers=1, num_attention_heads=4
                        )
                    ),
                ),
                "64 × 1",
            ),
            Case(
                "w64-d2",
                (
                    Model(
                        AgentSpec(
                            hidden_dim=64, attention_layers=2, num_attention_heads=4
                        )
                    ),
                ),
                "64 × 2",
            ),
            Case(
                "w128-d2",
                (
                    Model(
                        AgentSpec(
                            hidden_dim=128, attention_layers=2, num_attention_heads=4
                        )
                    ),
                ),
                "128 × 2",
            ),
        ),
    )


def regimes(base: TrainingRegime) -> dict[str, TrainingRegime]:
    """Hold information, pooling, learning and budget fixed across capacities."""
    return experiment(base).resolve().regimes
