"""Built-in runtime presets for training and simulation commands."""

from copy import deepcopy

DEFAULT_TRAIN_PRESET = "local"
DEFAULT_SIM_PRESET = "sim"


def _base_training_preset() -> dict:
    return {
        "observation": {
            "max_cards_per_player": 20,
            "max_permanents_per_player": 15,
            "max_actions": 10,
            "max_focus_objects": 2,
        },
        "match": {
            "hero": "gaea",
            "villain": "urza",
            "hero_deck": {
                "Mountain": 12,
                "Forest": 12,
                "Llanowar Elves": 18,
                "Grey Ogre": 18,
            },
            "villain_deck": {
                "Mountain": 12,
                "Forest": 12,
                "Llanowar Elves": 18,
                "Grey Ogre": 18,
            },
        },
        "train": {
            "total_timesteps": 20_000_000,
            "learning_rate": 2.5e-4,
            "num_envs": 16,
            "num_steps": 128,
            "anneal_lr": True,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "num_minibatches": 4,
            "update_epochs": 4,
            "norm_adv": True,
            "clip_coef": 0.1,
            "clip_vloss": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": "inf",
            "opponent_policy": "passive",
        },
        "reward": {
            "trivial": False,
            "managym": False,
            "win_reward": 1.0,
            "lose_reward": -1.0,
        },
        "agent": {
            "hidden_dim": 64,
            "num_attention_heads": 4,
            "attention_on": True,
        },
        "experiment": {
            "exp_name": "manabot",
            "seed": 1,
            "torch_deterministic": True,
            "device": "cpu",
            "wandb": True,
            "wandb_project_name": "manabot",
            "log_level": "INFO",
            "profiler_enabled": False,
        },
    }


TRAIN_PRESETS = {
    "local": {
        "train": {
            "num_envs": 4,
            "num_steps": 16,
            "total_timesteps": 10_000,
            "learning_rate": 2.5e-4,
        },
        "experiment": {
            "exp_name": "local",
            "device": "cpu",
            "profiler_enabled": True,
        },
        "agent": {"attention_on": True},
    },
    "simple": {
        "train": {
            "num_envs": 16,
            "num_steps": 256,
            "total_timesteps": 10_000_000,
            "learning_rate": 2.5e-4,
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
            "learning_rate": 2.5e-4,
        },
        "experiment": {
            "exp_name": "attention",
            "device": "cuda",
            "profiler_enabled": True,
        },
        "agent": {"attention_on": True},
    },
}

SIM_PRESETS = {
    "sim": {
        "experiment": {
            "exp_name": "sim",
            "wandb": False,
            "log_level": "DEBUG",
            "device": "cpu",
        },
        "sim": {
            "hero": "simple",
            "villain": "default",
            "num_games": 100,
            "num_threads": 4,
            "max_steps": 2000,
            "match": {
                "hero": "gaea",
                "villain": "urza",
                "hero_deck": {
                    "Mountain": 12,
                    "Forest": 12,
                    "Llanowar Elves": 18,
                    "Grey Ogre": 18,
                },
                "villain_deck": {
                    "Mountain": 12,
                    "Forest": 12,
                    "Llanowar Elves": 18,
                    "Grey Ogre": 18,
                },
            },
            "reward": {
                "trivial": False,
                "managym": False,
                "win_reward": 1.0,
                "lose_reward": -1.0,
            },
        },
    }
}


def get_training_preset(name: str) -> dict:
    if name not in TRAIN_PRESETS:
        available = ", ".join(sorted(TRAIN_PRESETS))
        raise ValueError(f"Unknown training preset '{name}'. Available: {available}")
    return deepcopy(TRAIN_PRESETS[name])


def get_training_base() -> dict:
    return deepcopy(_base_training_preset())


def get_sim_preset(name: str) -> dict:
    if name not in SIM_PRESETS:
        available = ", ".join(sorted(SIM_PRESETS))
        raise ValueError(f"Unknown simulation preset '{name}'. Available: {available}")
    return deepcopy(SIM_PRESETS[name])
