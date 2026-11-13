"""Shared installed scalar/vector controller configuration."""

from collections.abc import Callable

import pytest

import managym


def players() -> list[managym.PlayerConfig]:
    return [
        managym.PlayerConfig(name, {"Forest": 8, "Llanowar Elves": 4})
        for name in ("a", "b")
    ]


@pytest.mark.parametrize(
    "policies",
    [
        (first, second)
        for first in ("agent", "random", "heuristic")
        for second in ("agent", "random", "heuristic")
    ],
)
def test_scalar_vector_players_finish_with_same_rewards(
    policies: tuple[str, str],
) -> None:
    scalar = managym.Env(seed=213, player_policies=policies)
    batch = managym.VectorEnv(1, seed=213, player_policies=policies)
    obs, _ = scalar.reset(players())
    assert obs.toJSON() == batch.reset_all(players())[0][0].toJSON()
    for _ in range(1024):
        chooser = scalar.current_agent_index()
        assert chooser is not None
        external = policies[chooser] == "agent"
        next_obs, reward, done, truncated, _ = scalar.step(0 if external else None)
        other = batch.step([0 if external else -1])[0]
        assert (reward, done, truncated) == other[1:4]
        if done:
            return
        assert next_obs.toJSON() == other[0].toJSON()
    pytest.fail("bounded fixture did not finish")


@pytest.mark.parametrize(
    "api", [managym.Env, lambda **kwargs: managym.VectorEnv(1, **kwargs)]
)
@pytest.mark.parametrize("policies", [("agent",), ("agent", "bad"), ("none", "random")])
def test_invalid_player_policies(
    api: Callable[..., managym.Env | managym.VectorEnv], policies: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="two seats|unsupported player policy"):
        api(player_policies=policies)


def test_legacy_opponent_argument_warns_and_conflicts() -> None:
    with pytest.warns(DeprecationWarning, match="0.3.0"):
        managym.VectorEnv(1, opponent_policy="heuristic")
    with pytest.raises(ValueError, match="not both"):
        managym.VectorEnv(
            1, opponent_policy="random", player_policies=("agent", "heuristic")
        )


def test_two_builtins_reset_is_unplayed_and_forks_continue_identically() -> None:
    env = managym.Env(seed=213, player_policies=("heuristic", "heuristic"))
    obs, _ = env.reset(players())
    manual = managym.Env(seed=213)
    assert obs.toJSON() == manual.reset(players())[0].toJSON()
    with pytest.raises(managym.AgentError, match="built-in chooser"):
        env.step(0)
    env.step()
    fork = env.clone_env()
    assert env.step()[0].toJSON() == fork.step()[0].toJSON()
    env.set_seed(213)
    assert env.reset(players())[0].toJSON() == obs.toJSON()


def test_builtins_match_separate_native_selectors_with_live_combat_events() -> None:
    native = managym.Env(seed=213, player_policies=("heuristic", "heuristic"))
    manual = managym.Env(seed=213)
    selectors = [managym.Heuristic(), managym.Heuristic()]
    decks = [
        managym.authored_deck_setup("elemental-world-v2", key)
        for key in ("elemental_ur", "elemental_gw")
    ]
    native.reset(decks)
    obs, _ = manual.reset(decks)
    for _ in range(256):
        chooser = manual.current_agent_index()
        assert chooser is not None
        expected = manual.step(selectors[chooser].select_action(obs))
        actual = native.step()
        assert actual[0].toJSON() == expected[0].toJSON()
        assert actual[1:4] == expected[1:4]
        obs = expected[0]
        if expected[2] or expected[3]:
            break
