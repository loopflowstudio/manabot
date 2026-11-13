"""Exercise installed release artifacts from a directory outside the checkout."""

from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

import torch

import manabot
from manabot import managym as compatibility
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.eval.admission import register_anchor
from manabot.eval.evaluate import EvaluationError, validate_registration_source
from manabot.infra.identity import file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.semantic.policy_catalog import bind_policy_catalog
from manabot.sim import anchors
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.registration import PlayerRegistration
import managym


def check_heuristic_registration() -> str:
    """Register installed bytes, then reject changes to each bound source file."""
    runtime = runtime_fingerprints()
    candidate = PlayerRegistration(
        player_id="release-smoke",
        display_name="Release smoke",
        role="anchor",
        runner_kind="code",
        player_spec={"kind": "scripted_greedy"},
        compute_class_id="scripted-greedy-cpu-v1",
        information_boundary="acting-viewer",
        world=runtime["world"],
        content_suite="release-smoke",
        observation_abi_sha256=runtime["observation_abi_sha256"],
        action_abi_sha256=runtime["action_abi_sha256"],
        matchup_sha256=runtime["matchup_sha256"],
        source_sha256=file_sha256(Path(anchors.__file__)),
        player_seed_derivation_id="arena-pair-deal-player-v1",
    )
    registration = register_anchor(candidate, "heuristic", player_id="heuristic")
    validate_registration_source(registration)
    for path in heuristic_source_paths():
        # This smoke must never edit a development checkout.
        assert path.is_relative_to(Path(sys.prefix).resolve())
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            try:
                validate_registration_source(registration)
            except EvaluationError as error:
                assert "source drift" in str(error)
            else:
                raise AssertionError(f"source drift was not rejected: {path}")
        finally:
            path.write_bytes(original)
    validate_registration_source(registration)
    assert registration.source_sha256 is not None
    return registration.source_sha256


def main() -> None:
    assert version("managym") == version("manabot")
    assert compatibility is managym
    assert "/managym/" in managym._managym.__file__
    # The runner copies this script into a temporary directory and unsets PYTHONPATH.
    assert Path(manabot.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert Path(managym.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    for pack in ("ur-lessons-vs-gw-allies", "elemental-world-v2"):
        bind_policy_catalog(pack)
    check_heuristic_registration()
    value = torch.tensor([1.0], requires_grad=True)
    value.square().sum().backward()
    assert value.grad is not None and value.grad.item() == 2.0
    command = ["uv", "run", "--no-project", "--python", sys.executable, "manabot"]
    subprocess.run([*command, "--help"], check=True)
    subprocess.run([*command, "train", "--help"], check=True)
    output = Path.cwd() / "release-smoke"
    subprocess.run(
        [
            *command,
            "train",
            "--recipe",
            "local",
            "--device",
            "cpu",
            "--out",
            str(output),
        ],
        check=True,
        timeout=180,
    )
    policy, _ = load_checkpoint(str(output / "self-play-raw.pt"))
    assert not policy.training
    print(
        f"release smoke passed: Python {sys.version.split()[0]}, manabot {version('manabot')}, "
        f"torch {torch.__version__}, etude-sdk {version('etude-sdk')}"
    )


if __name__ == "__main__":
    main()
