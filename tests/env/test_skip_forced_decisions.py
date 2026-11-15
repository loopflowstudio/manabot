"""Every game runner skips forced decisions; only debugging fixtures opt out.

A forced decision has exactly one legal action; sending it to a model or
policy costs inference and carries no decision value. Native skipping
(`skip_trivial`) resolves them, so it must be the default everywhere.
"""

from pathlib import Path
import re

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
    # A one-mana cast taps the only source and leaves forced priority passes;
    # action zero alone may instead keep opening deliberate mana activations.
    (initial,) = vector.reset_all(
        [
            managym.PlayerConfig(name, {"Plains": 14, "Savannah Lions": 8})
            for name in ("a", "b")
        ]
    )
    obs, _ = initial
    for _ in range(50):
        assert len(obs.action_space.actions) > 1
        action = next(
            (
                index
                for kind in (
                    managym.ActionEnum.PRIORITY_PLAY_LAND,
                    managym.ActionEnum.PRIORITY_CAST_SPELL,
                    managym.ActionEnum.PAY_COST,
                    managym.ActionEnum.PRIORITY_PASS_PRIORITY,
                )
                for index, choice in enumerate(obs.action_space.actions)
                if choice.action_type == kind
            ),
            0,
        )
        (result,) = vector.step([action])
        obs = result[0]
    assert sum(vector.skip_trivial_counts()) > 0


def test_runner_configuration_defaults_skip() -> None:
    assert GameConfig().skip_trivial


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
