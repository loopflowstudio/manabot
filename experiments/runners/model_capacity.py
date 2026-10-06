"""Declare registered capacity cases; workload and evaluation stay with the caller.

No accounting/calibration or training happens here. Existing case IDs and complete
scientific configurations are preserved; these are not checkpoint weight transfers.
"""

from manabot.infra.hypers import AgentSpec
from manabot.training.experiments import Baseline, Case, Experiment, Model
from manabot.training.models import TrainingRegime


def experiment(base: TrainingRegime, *, include_ataraxos: bool = False) -> Experiment:
    models = [
        AgentSpec(hidden_dim=64, attention_layers=1, num_attention_heads=4),
        AgentSpec(hidden_dim=64, attention_layers=2, num_attention_heads=4),
        AgentSpec(hidden_dim=128, attention_layers=2, num_attention_heads=4),
    ]
    if include_ataraxos:
        models.append(
            AgentSpec(
                hidden_dim=384,
                attention_layers=8,
                num_attention_heads=4,
                attention_feedforward_dim=1536,
            )
        )
    return Experiment(
        name="",
        baseline=Baseline.capture("capacity-control-v1", base),
        cases=tuple(
            Case(
                f"w{model.hidden_dim}-d{model.attention_layers}",
                (Model(model),),
                f"{model.hidden_dim} × {model.attention_layers}",
            )
            for model in models
        ),
    )


def regimes(
    base: TrainingRegime, *, include_ataraxos: bool = False
) -> dict[str, TrainingRegime]:
    """Hold information, pooling, learning and budget fixed across capacities."""
    return experiment(base, include_ataraxos=include_ataraxos).resolve().regimes
