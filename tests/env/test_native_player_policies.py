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
    mixed = managym.Env(seed=213, player_policies=("agent", "heuristic"))
    mixed.reset(players())
    with pytest.raises(managym.AgentError, match="external chooser.*action index"):
        mixed.step()
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


def test_normal_reset_offers_both_seats_the_same_opening_choices_in_scalar_and_vector() -> (
    None
):
    scalar = managym.Env(seed=213)
    batch = managym.VectorEnv(1, seed=213)
    observation, _ = scalar.reset(players())
    batch_observation, _ = batch.reset_all(players())[0]
    for seat, labels, choice in (
        (0, ["Keep hand", "Mulligan"], 1),
        (1, ["Keep hand", "Mulligan"], 0),
        (0, None, 3),
        (0, ["Keep hand", "Mulligan"], 0),
    ):
        assert observation.toJSON() == batch_observation.toJSON()
        assert scalar.current_agent_index() == seat
        assert observation.turn.turn_number == 0
        offers = [action.description for action in observation.action_space.actions]
        if labels is None:
            assert len(offers) == 7
            assert all(label and "deepest first" in label for label in offers)
        else:
            assert offers == labels
        assert not scalar.observation_for_player(1 - seat).action_space.actions
        observation = scalar.step(choice)[0]
        batch_observation = batch.step([choice])[0][0]
    assert observation.toJSON() == batch_observation.toJSON()
    assert observation.turn.turn_number == 1


def test_other_seat_cannot_distinguish_private_bottom_choices() -> None:
    env = managym.Env(seed=213)
    env.reset(players())
    env.step(1)
    env.step(0)
    choices = env.observation_for_player(0).action_space.actions
    other_index = next(
        i for i, a in enumerate(choices) if a.description != choices[0].description
    )
    branch = env.clone_env()
    env.step(0)
    branch.step(other_index)
    assert (
        env.observation_for_player(1).toJSON()
        == branch.observation_for_player(1).toJSON()
    )
    assert (
        env.observation_for_player(0).agent.known_library
        != branch.observation_for_player(0).agent.known_library
    )
    owner_view = env.observation_for_player(0).toJSON()
    sampled = env.clone_env()
    sampled.determinize(seed=704, perspective=0)
    assert sampled.observation_for_player(0).toJSON() == owner_view
