"""Exercise installed release artifacts from a directory outside the checkout."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys

import torch

import manabot
from manabot import managym
from manabot.model.checkpoint import load_checkpoint
from manabot.semantic.policy_catalog import bind_policy_catalog
import managym as legacy_managym


def main() -> None:
    try:
        version("managym")
    except PackageNotFoundError:
        pass
    else:
        raise AssertionError("managym must not be installed as a separate distribution")
    assert legacy_managym is managym
    assert "/manabot/managym/" in managym._managym.__file__
    # The runner copies this script into a temporary directory and unsets PYTHONPATH.
    assert Path(manabot.__file__).is_relative_to(Path(sys.prefix))
    assert Path(managym.__file__).is_relative_to(Path(sys.prefix))
    for pack in ("ur-lessons-vs-gw-allies", "elemental-world-v2"):
        bind_policy_catalog(pack)
    value = torch.tensor([1.0], requires_grad=True)
    value.square().sum().backward()
    assert value.grad is not None and value.grad.item() == 2.0
    command = ["uv", "run", "--no-project", "--python", sys.executable, "manabot"]
    subprocess.run([*command, "--help"], check=True)
    subprocess.run([*command, "train", "--help"], check=True)
    output = Path.cwd() / "release-smoke"
    subprocess.run(
        [*command, "train", "--recipe", "local", "--out", str(output)],
        check=True,
        timeout=180,
    )
    policy, _ = load_checkpoint(str(output / "self-play-raw.pt"))
    assert not policy.training
    print(
        f"release smoke passed: manabot {version('manabot')}, torch {torch.__version__}"
    )


if __name__ == "__main__":
    main()
