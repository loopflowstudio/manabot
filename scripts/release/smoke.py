"""Exercise installed release artifacts from a directory outside the checkout."""

from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

import torch

import manabot
from manabot.semantic.policy_catalog import bind_policy_catalog
import managym


def main() -> None:
    assert version("manabot") == version("managym")
    # The runner copies this script into a temporary directory and unsets PYTHONPATH.
    assert Path(manabot.__file__).is_relative_to(Path(sys.prefix))
    assert Path(managym.__file__).is_relative_to(Path(sys.prefix))
    for pack in ("ur-lessons-vs-gw-allies", "elemental-world-v2"):
        bind_policy_catalog(pack)
    value = torch.tensor([1.0], requires_grad=True)
    value.square().sum().backward()
    assert value.grad is not None and value.grad.item() == 2.0
    subprocess.run(["manabot", "--help"], check=True)
    subprocess.run(["manabot", "train", "--help"], check=True)
    subprocess.run(
        [
            "manabot",
            "train",
            "--set",
            "train.total_timesteps=32",
            "--set",
            "train.num_envs=2",
            "--set",
            "train.num_steps=16",
        ],
        check=True,
        timeout=180,
    )
    print(
        f"release smoke passed: manabot {version('manabot')}, torch {torch.__version__}"
    )


if __name__ == "__main__":
    main()
