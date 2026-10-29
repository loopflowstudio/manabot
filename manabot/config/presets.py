"""Built-in runtime presets for simulation commands."""

from manabot.infra.config import RuntimeConfig, SimulationConfig

DEFAULT_SIM_PRESET = "sim"


def get_sim_preset(name: str) -> dict:
    if name != DEFAULT_SIM_PRESET:
        raise ValueError(
            f"Unknown simulation preset '{name}'. Available: {DEFAULT_SIM_PRESET}"
        )

    runtime = RuntimeConfig().model_dump()
    runtime.update(
        {
            "name": DEFAULT_SIM_PRESET,
            "log_level": "DEBUG",
        }
    )
    return {
        "runtime": runtime,
        "sim": SimulationConfig().model_dump(),
    }
