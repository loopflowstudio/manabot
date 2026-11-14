"""Play a bounded complete game using only an installed managym wheel."""

from importlib import import_module
from importlib.metadata import distributions, version
from pathlib import Path
import pkgutil
import random
import sys

import managym
from managym.decision import Command


def main() -> None:
    installed = {d.metadata["Name"].lower() for d in distributions()}
    assert not installed.intersection({"manabot", "torch", "gymnasium", "etude-sdk"})
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
