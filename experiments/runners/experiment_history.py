"""History comparison declared through Experiment; the shared runner owns execution.

A separately admitted current-source plan supplies calibrated counts and frozen
scientific deal reservations. This entry point does not calibrate or allocate new
science, and cannot attach to an existing live campaign.
"""

from experiments.runners.experiment_screen import screen_main
from experiments.runners.training_protocol import ResolvedStudy
from manabot.infra.hypers import AgentSpec
from manabot.training.experiment_execution import ExperimentSchedule
from manabot.training.experiments import Baseline, Case, Experiment, Model
from manabot.training.models import TrainingRegime


def declaration(plan: ResolvedStudy, schedule: ExperimentSchedule) -> Experiment:
    if plan.protocol.study != "history-input":
        raise ValueError("history comparison requires a history-input plan")
    baseline = TrainingRegime.model_validate(plan.recipes[0])
    return Experiment(
        name="",
        baseline=Baseline.capture("history-control", baseline),
        cases=(
            Case("history-off"),
            Case("history-on", (Model(AgentSpec(recent_events=True)),)),
        ),
        schedule=schedule,
    )


if __name__ == "__main__":
    screen_main(declaration)
