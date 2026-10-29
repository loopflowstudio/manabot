"""Every game runner skips forced decisions; only debugging fixtures opt out.

A forced decision has exactly one legal action; sending it to a model or
policy costs inference and carries no decision value. Native skipping
(`skip_trivial`) resolves them, so it must be the default everywhere.
"""

import inspect
from pathlib import Path
import re

from manabot.training.regime import TrainCompound
from manabot.training.rollout.compound import collect_game
from manabot.verify.llm_game import GameConfig
import managym

ROOT = Path(__file__).parents[2]

# Debugging fixtures: scripted walkers that drive both seats through
# constructed positions one native window at a time. Neither plays a game for
# a person, model or policy; anything else that turns skipping off is a
# regression.
DIAGNOSTIC_EXCEPTIONS = {
    "manabot/verify/action_cases.py",
    "manabot/verify/text_observation.py",
}

_OFF = re.compile(r"skip_trivial\s*(?::\s*bool\s*)?=\s*(False|false)\b")


def _deck() -> dict[str, int]:
    return {"Mountain": 24, "Gray Ogre": 12, "Lightning Bolt": 4}


def test_native_default_never_surfaces_a_forced_decision() -> None:
    env = managym.Env(seed=3)
    obs, _ = env.reset([managym.PlayerConfig(name, _deck()) for name in ("a", "b")])
    done = False
    for _ in range(2000):
        if done:
            break
        assert len(obs.action_space.actions) > 1
        obs, _, done, _, _ = env.step(0)
    assert done
    assert env.skip_trivial_count() > 0


def test_vector_default_skips_forced_decisions() -> None:
    vector = managym.VectorEnv(1, seed=3)
    vector.reset_all([managym.PlayerConfig(name, _deck()) for name in ("a", "b")])
    for _ in range(50):
        vector.step([0])
    assert sum(vector.skip_trivial_counts()) > 0


def test_runner_configuration_defaults_skip() -> None:
    assert GameConfig().skip_trivial
    assert TrainCompound.model_fields["skip_trivial"].default is True
    assert inspect.signature(collect_game).parameters["skip_trivial"].default is True


def test_no_runner_turns_skipping_off() -> None:
    sources = [
        path
        for package in ("manabot", "academy", "scripts", "experiments", "managym/src")
        for path in (ROOT / package).rglob("*")
        if path.suffix in {".py", ".rs"} and "data" not in path.parts
    ]
    offenders = sorted(
        f"{relative}:{number}"
        for path in sources
        if (relative := path.relative_to(ROOT).as_posix()) not in DIAGNOSTIC_EXCEPTIONS
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if _OFF.search(line)
    )
    assert offenders == []
