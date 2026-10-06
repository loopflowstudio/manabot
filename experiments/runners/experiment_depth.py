"""Depth comparison declared through Experiment; no private scheduler."""

from experiments.runners.experiment_screen import screen_main
from experiments.runners.training_protocol import ResolvedStudy
from manabot.infra.hypers import AgentSpec
from manabot.training.experiment_execution import ExperimentSchedule
from manabot.training.experiments import Baseline, Case, Experiment, Model
from manabot.training.models import TrainingRegime


def declaration(plan: ResolvedStudy, schedule: ExperimentSchedule) -> Experiment:
    if plan.protocol.study != "depth-screen":
        raise ValueError("depth comparison requires a depth-screen plan")
    baseline = TrainingRegime.model_validate(plan.recipes[0])
    return Experiment(
        name="",
        baseline=Baseline.capture("depth-control", baseline),
        cases=(
            Case("depth-1"),
            Case("depth-2", (Model(AgentSpec(attention_layers=2)),)),
        ),
        schedule=schedule,
    )


if __name__ == "__main__":
    screen_main(declaration)
