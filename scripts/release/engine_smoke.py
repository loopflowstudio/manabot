"""Play a bounded complete game using only an installed managym wheel."""

from importlib import import_module
from importlib.metadata import distributions, version
import json
from pathlib import Path
import pkgutil
import random
import subprocess
import sys
from tempfile import TemporaryDirectory

import managym
from managym.decision import Command
from managym.recording import ScalarCapture, read_failure
from managym.replay import replay_failure_inputs, require_replay_runtime


def interrupted_recording(path: Path) -> None:
    """Exit inside a controlled native-call substitute, without exception cleanup."""
    code = """
import os
import sys
from pathlib import Path
import managym
from managym.recording import ScalarCapture

game = managym.Env(seed=42)
capture = ScalarCapture(42, journal_path=Path(sys.argv[1]))
capture.reset(game, [managym.PlayerConfig(n, {"Mountain": 20}) for n in ("a", "b")])

class Interrupted:
    def current_agent_index(self) -> int:
        return game.current_agent_index()
    def semantic_decision_frame_json(self) -> str:
        return game.semantic_decision_frame_json()
    def step(self, action: int) -> None:
        os._exit(73)

capture.step(Interrupted(), 0)
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 73, child.stderr
    evidence = read_failure(path)
    assert evidence.boundary is not None
    assert evidence.boundary.phase == "executing" and not evidence.boundary.returned
    require_replay_runtime(
        evidence.identities, allow_changed_runtime=False, require_source=True
    )
    result = replay_failure_inputs(evidence, statistics=True)[0]
    assert result["status"] == "advanced"
    assert result["original_outcome"] == "execution_outcome_unknown"
    assert result["statistics"] is not None
    before = path.read_bytes()
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "managym.replay",
            str(path),
            "--seconds",
            "10",
            "--statistics",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["results"][0]["status"] == "advanced"
    assert path.read_bytes() == before


def main() -> None:
    installed = {d.metadata["Name"].lower() for d in distributions()}
    assert not installed.intersection(
        {"manabot", "torch", "gymnasium", "etude", "etude-sdk"}
    )
    assert Path(managym.__file__).is_relative_to(Path(sys.prefix))
    for module in pkgutil.iter_modules(managym.__path__, "managym."):
        import_module(module.name)
    for cls in (
        managym.Deck,
        managym.DeckEntry,
        managym.Decklist,
        managym.DeckImport,
        managym.DeckIssue,
    ):
        assert cls.__module__ == "managym"
    assert Command.__module__ == "managym.decision"
    game = managym.Env(seed=42)
    observation, _ = game.reset(
        [
            managym.authored_deck_setup("tla-baseline-v1", key)
            for key in ("tla_red", "tla_white")
        ]
    )
    with TemporaryDirectory() as directory:
        recorded = managym.Env(seed=42)
        recorded.enable_statistics()
        capture = ScalarCapture(
            42, journal_path=Path(directory) / "private" / "game.jsonl"
        )
        capture.reset(
            recorded,
            [
                managym.authored_deck_setup("tla-baseline-v1", k)
                for k in ("tla_red", "tla_white")
            ],
        )
        capture.step(recorded, 0)
        assert recorded.statistics() is not None
        evidence = read_failure(Path(directory) / "private" / "game.jsonl")
        require_replay_runtime(
            evidence.identities, allow_changed_runtime=False, require_source=True
        )
        assert replay_failure_inputs(evidence)[0]["status"] == "advanced"
        interrupted_recording(Path(directory) / "interrupted.jsonl")
        assert not {"manabot", "torch", "etude"}.intersection(sys.modules)
    rng = random.Random(42)
    for steps in range(10_000):
        action = rng.randrange(len(observation.action_space.actions))
        observation, _, terminated, truncated, _ = game.step(action)
        if terminated or truncated:
            assert terminated and not truncated
            print(
                f"engine smoke passed: managym {version('managym')}, {steps + 1} decisions"
            )
            return
    raise AssertionError("engine game exceeded its decision bound")


if __name__ == "__main__":
    main()
