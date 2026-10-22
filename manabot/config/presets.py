"""Built-in runtime presets for simulation commands."""

from manabot.infra import ExperimentHypers, SimulationHypers

DEFAULT_SIM_PRESET = "sim"


def get_sim_preset(name: str) -> dict:
    if name != DEFAULT_SIM_PRESET:
        raise ValueError(
            f"Unknown simulation preset '{name}'. Available: {DEFAULT_SIM_PRESET}"
        )

    experiment = ExperimentHypers().model_dump()
    experiment.update(
        {
            "exp_name": DEFAULT_SIM_PRESET,
            "wandb": False,
            "log_level": "DEBUG",
        }
    )
    return {
        "experiment": experiment,
        "sim": SimulationHypers().model_dump(),
    }
