"""Exercise installed release artifacts from a directory outside the checkout."""

from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

import torch

import manabot
from manabot import managym as compatibility
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.recipes.local import local_smoke
from manabot.semantic.policy_catalog import bind_policy_catalog
from manabot.training.regime import TrainingRegime, TrainSelfPlay
import managym
from managym.academy.extensions import bot_extension


def _training_regime() -> TrainingRegime:
    """Check installed learning with bounded native targets, not game duration."""
    regime = local_smoke(steps=1, device="cpu")
    regime.id = "release-smoke-v1"
    regime.agent = regime.agent.model_copy(
        update={"hidden_dim": 16, "attention_layers": 1, "num_attention_heads": 2}
    )
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.root = "lethal-target-v1"
    stage.transitions = 4
    return TrainingRegime.model_validate(regime.model_dump())


def main() -> None:
    assert version("managym") == version("manabot")
    assert compatibility is managym
    assert "/managym/" in managym._managym.__file__
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
    regime_path = Path.cwd() / "release-smoke-regime.json"
    regime_path.write_text(_training_regime().model_dump_json(indent=2))
    subprocess.run(
        [
            *command,
            "train",
            "--regime",
            str(regime_path),
            "--out",
            str(output),
        ],
        check=True,
        timeout=180,
    )
    policy, _ = load_checkpoint(str(output / "self-play-raw.pt"))
    assert not policy.training
    extension = bot_extension()
    assert extension is not None and extension.name == "manabot", (
        "Installed Academy bot entry point is missing"
    )
    assert extension.watch_players == ("greedy", "heuristic")
    checkpoints, missing = extension.discover(
        [output / "self-play-raw.pt"], runs_dir=output / "no-other-runs"
    )
    assert not missing, missing
    assert len(checkpoints) == 1 and checkpoints[0].compatible, checkpoints
    print(
        f"release smoke passed: Python {sys.version.split()[0]}, manabot {version('manabot')}, "
        f"torch {torch.__version__}, etude-sdk {version('etude-sdk')}, Academy bot extension"
    )


if __name__ == "__main__":
    main()
