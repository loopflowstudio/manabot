"""Built-in runtime presets for training and simulation commands."""

from copy import deepcopy

from manabot.infra import Hypers, SimulationHypers

DEFAULT_TRAIN_PRESET = "local"
DEFAULT_SIM_PRESET = "sim"

TRAIN_PRESETS = {
    "local": {
        "train": {
            "num_envs": 4,
            "num_steps": 16,
            "total_timesteps": 10_000,
        },
        "experiment": {
            "exp_name": "local",
            "profiler_enabled": True,
        },
    },
    "simple": {
        "train": {
            "num_envs": 16,
            "num_steps": 256,
            "total_timesteps": 10_000_000,
        },
        "experiment": {
            "exp_name": "simple",
            "device": "cuda",
            "profiler_enabled": True,
        },
        "agent": {"attention_on": False},
    },
    "attention": {
        "train": {
            "num_envs": 16,
            "num_steps": 256,
            "total_timesteps": 10_000_000,
        },
        "experiment": {
            "exp_name": "attention",
            "device": "cuda",
            "profiler_enabled": True,
        },
    },
}


def get_training_preset(name: str) -> dict:
    if name not in TRAIN_PRESETS:
        available = ", ".join(sorted(TRAIN_PRESETS))
        raise ValueError(f"Unknown training preset '{name}'. Available: {available}")
    return deepcopy(TRAIN_PRESETS[name])


def get_training_base() -> dict:
    return Hypers().model_dump()


def get_sim_preset(name: str) -> dict:
    if name != DEFAULT_SIM_PRESET:
        available = DEFAULT_SIM_PRESET
        raise ValueError(f"Unknown simulation preset '{name}'. Available: {available}")
    preset = {
        "experiment": Hypers().experiment.model_dump(),
        "sim": SimulationHypers().model_dump(),
    }
    preset["experiment"].update(
        {
            "exp_name": DEFAULT_SIM_PRESET,
            "wandb": False,
            "log_level": "DEBUG",
        }
    )
    return preset
