"""Open and closed decklists through the match, encoders, text and Academy wire."""

from typing import Literal

import numpy as np
import pytest

from manabot.env.match import Match
from manabot.env.observation import DECKLIST_DIM, ObservationSpace
from manabot.infra.config import MatchConfig, ObservationConfig
from manabot.managym.experience_projection import serialize_observation
from manabot.semantic.text_observation import capture_text_decision
from manabot.sim.registration import (
    OBSERVATION_ONLY_KINDS,
    PlayerRegistration,
    require_open_decklists,
)
import managym

Decklists = Literal["open", "closed"]


def _match(decklists: Decklists) -> MatchConfig:
    return MatchConfig.authored(
        "ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies"
    ).model_copy(update={"decklists": decklists})


def _env(decklists: Decklists) -> tuple[managym.Env, managym.Observation]:
    env = managym.Env(seed=3)
    obs, _ = env.reset(Match(_match(decklists)).to_rust())
    return env, obs


@pytest.mark.parametrize("decklists", ["open", "closed"])
def test_opponent_lists_appear_exactly_when_open(decklists: Decklists) -> None:
    env, obs = _env(decklists)
    agent, opponent = obs.agent, obs.opponent
    assert sum(agent.main_counts.values()) == 41
    assert sum(agent.unseen_counts.values()) == agent.zone_counts[0]
    assert sum(agent.sideboard_counts.values()) == 3
    assert agent.remaining_sideboard_counts == agent.sideboard_counts
    fields = (
        opponent.main_counts,
        opponent.unseen_counts,
        opponent.sideboard_counts,
        opponent.remaining_sideboard_counts,
    )
    if decklists == "open":
        assert sum(opponent.main_counts.values()) == 40
        assert sum(opponent.unseen_counts.values()) == (
            opponent.zone_counts[0]
            + opponent.zone_counts[1]
            - sum(opponent.known_hand.values())
        )
        assert sum(opponent.sideboard_counts.values()) == 2
    else:
        assert fields == (None, None, None, None)

    # Policy input v5: Python reference and native encoders agree.
    space = ObservationSpace(ObservationConfig())
    python = space.encode(obs)["decklists"]
    native = env.encode_observation(obs)["decklists"]
    assert python.shape == native.shape == (2, 60, DECKLIST_DIM)
    np.testing.assert_array_equal(python, native)
    assert python[0, :, 1].sum() == 41
    assert python[1, :, 1].sum() == (40 if decklists == "open" else 0)
    assert not python[1].any() if decklists == "closed" else python[1].any()
    assert "decklists" not in ObservationSpace(
        ObservationConfig(policy_input_version=4)
    ).encode(obs)

    # LLM text and the Academy projection carry the same disclosure.
    text = capture_text_decision(env).text
    assert ("Decklist: closed." in text) == (decklists == "closed")
    assert text.count("Main deck:") == (2 if decklists == "open" else 1)
    projection = serialize_observation(obs)
    assert "main_counts" in projection["agent"]
    assert ("main_counts" in projection["opponent"]) == (decklists == "open")
    assert ("sideboard_counts" in projection["opponent"]) == (decklists == "open")


def test_closed_decklists_refuse_hidden_card_sampling() -> None:
    env, _ = _env("closed")
    with pytest.raises(Exception, match="decklist closed"):
        env.determinize(perspective=0, seed=1)
    with pytest.raises(Exception, match="decklist closed"):
        env.possible_world_space_json(0)
    open_env, _ = _env("open")
    open_env.determinize(perspective=0, seed=1)


def test_closed_match_rejects_search_players_when_built() -> None:
    def registration(kind: str) -> PlayerRegistration:
        return PlayerRegistration.model_construct(
            player_id=f"{kind.replace('_', '-')}-player", player_spec={"kind": kind}
        )

    observation_only = [registration(kind) for kind in sorted(OBSERVATION_ONLY_KINDS)]
    require_open_decklists("closed", observation_only)
    search = registration("determinized_puct")
    require_open_decklists("open", [search])
    with pytest.raises(ValueError, match="closed-decklist match"):
        require_open_decklists("closed", [*observation_only, search])


def test_open_is_the_default_and_omitted_from_serialized_configuration() -> None:
    default = MatchConfig()
    assert default.decklists == "open"
    assert "decklists" not in default.model_dump()
    assert _match("closed").model_dump()["decklists"] == "closed"
    hero, villain = Match(_match("closed")).swapped().to_rust()
    assert hero.disclosure == villain.disclosure == "closed"
    with pytest.raises(ValueError):
        managym.PlayerConfig("x", {"Island": 1}, disclosure="partial")
