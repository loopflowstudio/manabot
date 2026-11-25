"""Open and closed decklists through the match, encoders, text and Academy wire."""

from dataclasses import asdict
from functools import cache
import random

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import (
    DECKLIST_DIM,
    ObservationSpace,
    column,
    encode_policy_batch,
)
from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.managym.experience_projection import serialize_observation
from manabot.policy.prepare import Prepared
from manabot.semantic.text_observation import capture_text_decision
from manabot.sim.registration import (
    OBSERVATION_ONLY_KINDS,
    PlayerRegistration,
    require_open_decklists,
)
from manabot.verify.text_observation import keep_opening_hands
import managym
from managym.disclosure import DecklistDisclosure, DecklistSampling
from tests.fixtures.observation_rows import observation_arrays


def _match(decklists: DecklistDisclosure) -> MatchConfig:
    return MatchConfig.authored(
        "ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies"
    ).model_copy(update={"decklists": decklists})


def _env(decklists: DecklistDisclosure) -> tuple[managym.Env, managym.Observation]:
    env = managym.Env(seed=3)
    obs, _ = env.reset(Match(_match(decklists)).to_rust())
    return env, obs


@cache
def _policy() -> ActorCritic:
    torch.set_num_threads(1)
    return ActorCritic(
        ObservationSpace(), ModelConfig(hidden_dim=8, num_attention_heads=2)
    )


def _prepared(
    space: ObservationSpace, obs: managym.Observation
) -> Prepared[np.ndarray]:
    return _policy().prepare(encode_policy_batch(space, [obs]))


def _assert_same_view(a: managym.Observation, b: managym.Observation) -> None:
    """Viewer JSON, every encoded buffer and the Prepared input all agree."""
    space = ObservationSpace()
    assert a.toJSON() == b.toJSON()
    left, right = observation_arrays(space, a), observation_arrays(space, b)
    for key, value in left.items():
        np.testing.assert_array_equal(value, right[key], err_msg=key)
    np.testing.assert_equal(asdict(_prepared(space, a)), asdict(_prepared(space, b)))


@pytest.mark.parametrize("decklists", ["open", "closed"])
def test_opponent_lists_appear_exactly_when_open(decklists: DecklistDisclosure) -> None:
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

    # The policy input flags disclosure for both viewers and encodes open lists.
    space = ObservationSpace(ObservationConfig())
    assert obs.decklists_closed == (decklists == "closed")
    for viewer in (0, 1):
        encoded = observation_arrays(space, env.observation_for_player(viewer))
        for key in ("agent_player", "opponent_player"):
            assert encoded[key][0, column(key, "decklists_open")] == float(
                decklists == "open"
            )
    lists = observation_arrays(space, obs)["decklists"]
    assert lists.shape == (2, 60, DECKLIST_DIM)
    assert lists[0, :, 1].sum() == 41
    assert lists[1, :, 1].sum() == (40 if decklists == "open" else 0)
    assert not lists[1].any() if decklists == "closed" else lists[1].any()

    # LLM text and the Academy projection carry the same disclosure.
    text = capture_text_decision(env).text
    assert ("Decklist: closed." in text) == (decklists == "closed")
    assert f"Decklists: {decklists} (" in text
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
    setup = Match(_match("closed")).swapped().to_rust()
    assert setup.decklists == "closed"
    with pytest.raises(TypeError, match="disclosure"):
        managym.PlayerConfig("x", {"Island": 1}, disclosure="closed")


def test_match_from_players_retains_setup_and_rejects_different_worlds() -> None:
    players = Match(_match("closed")).to_rust()
    actual = MatchConfig.from_players(*players, decklists=players.decklists)
    assert actual.decklists == "closed"
    assert actual.hero_deck == players[0].decklist
    assert actual.villain_sideboard == players[1].sideboard
    assert (actual.hero, actual.villain, actual.world) == (
        players[0].name,
        players[1].name,
        players[0].world,
    )
    players = managym.GameSetup(
        [players[0], managym.Deck({"Island": 40}).player()],
        decklists=players.decklists,
    )
    with pytest.raises(ValueError, match=r"same world: seat 0 .*seat 1"):
        MatchConfig.from_players(*players, decklists=players.decklists)


@pytest.mark.parametrize("viewer", [0, 1])
def test_unseen_opponent_list_does_not_change_observation_or_prepared(
    viewer: int,
) -> None:
    space = ObservationSpace()

    def game(land: str, disclosure: DecklistDisclosure) -> managym.Env:
        decks = [{"Forest": 40}, {land: 40}]
        if viewer == 1:
            decks.reverse()
        env = managym.Env(seed=3)
        raw, _ = env.reset(
            Match(
                MatchConfig(
                    hero_deck=decks[0], villain_deck=decks[1], decklists=disclosure
                )
            ).to_rust()
        )
        keep_opening_hands(env, raw)
        return env

    # Open control: the substitution is visible everywhere it could leak.
    opened = [
        game(land, "open").observation_for_player(viewer)
        for land in ("Island", "Mountain")
    ]
    assert opened[0].toJSON() != opened[1].toJSON()
    assert not np.array_equal(
        observation_arrays(space, opened[0])["decklists"],
        observation_arrays(space, opened[1])["decklists"],
    )
    assert not np.array_equal(
        _prepared(space, opened[0]).deck_rows, _prepared(space, opened[1]).deck_rows
    )

    left, right = game("Island", "closed"), game("Mountain", "closed")
    _assert_same_view(*(env.observation_for_player(viewer) for env in (left, right)))
    decisions = 0
    # All-land decks have identical action ordering until the opponent plays a
    # land. Equal sizes preserve shuffle RNG and the viewer's private cards.
    for _ in range(100):
        a, b = (env.observation_for_player(viewer) for env in (left, right))
        if a.opponent_permanents:
            break
        if left.current_agent_index() == viewer:
            _assert_same_view(a, b)
            decisions += 1
        index = 0
        if left.current_agent_index() != viewer and decisions == 0:
            # Let the on-the-draw viewer act before the opponent exposes a land.
            live = left.observation_for_player(left.current_agent_index())
            index = next(
                i
                for i, action in enumerate(live.action_space.actions)
                if int(action.action_type)
                == int(managym.ActionEnum.PRIORITY_PASS_PRIORITY)
            )
        left.step(index)
        right.step(index)
    else:
        pytest.fail("opponent never exposed a land")
    assert decisions > 0


def test_public_reveal_then_departure_updates_closed_known_hand() -> None:
    space = ObservationSpace()
    env = managym.Env(seed=0)
    obs, _ = env.reset(Match(_match("closed")).to_rust())
    rng = random.Random(0)
    owner: int | None = None
    revealed: int | None = None
    prepared_row: int | None = None
    for _ in range(3000):
        if owner is not None:
            view = env.observation_for_player(1 - owner)
            encoded = observation_arrays(space, view)
            assert not encoded["decklists"][1].any()
            prepared = _policy().prepare(encode_policy_batch(space, [view]))
            assert revealed is not None
            selected = encoded["known_hand"][1, :, 0] == revealed + 1
            if prepared_row is None:
                prepared_row = int(prepared.known_rows[0, 1, selected][0])
                assert prepared_row > 0
            if revealed not in view.opponent.known_hand:
                assert not selected.any()
                assert not prepared.known_counts[0, 1][
                    prepared.known_rows[0, 1] == prepared_row
                ].any()
                return
            count = view.opponent.known_hand[revealed]
            assert encoded["known_hand"][1, selected, 1].sum() == count
            assert (
                prepared.known_counts[0, 1][
                    prepared.known_rows[0, 1] == prepared_row
                ].sum()
                == count
            )
        actions = obs.action_space.actions
        retrievals = [
            i
            for i, a in enumerate(actions)
            if int(a.action_type) == int(managym.ActionEnum.LEARN_TAKE_LESSON)
        ]
        if retrievals and owner is None:
            owner = env.current_agent_index()
            obs, *_ = env.step(retrievals[-1])
            known = env.observation_for_player(1 - owner).opponent.known_hand
            assert sum(known.values()) == 1
            revealed = next(iter(known))
        else:
            assert not obs.game_over, "game ended before revealed card left hand"
            obs, *_ = env.step(rng.randrange(len(actions)))
    pytest.fail("no public reveal followed by hand departure in bounded game")


@pytest.mark.parametrize("decklists", ["closed", "mixed"])
def test_game_setup_rewrap_preserves_disclosure(
    decklists: DecklistSampling,
) -> None:
    original = managym.GameSetup(
        [managym.PlayerConfig(name, {"Island": 40}) for name in ("a", "b")],
        decklists=decklists,
    )
    wrapped = managym.GameSetup(original)
    assert wrapped.decklists == decklists
    assert wrapped.players == original.players
    assert managym.GameSetup(original, decklists="open").decklists == "open"
    assert managym.GameSetup(list(original)).decklists == "open"
    modes: set[bool] = set()
    for seed in range(8):
        expected = managym.resolve_decklists(original, seed) == "open"
        observation, _ = managym.Env(seed=seed).reset(wrapped)
        assert observation.decklists_closed is (not expected)
        ((observation, _),) = managym.VectorEnv(num_envs=1, seed=seed).reset_all(
            wrapped
        )
        assert observation.decklists_closed is (not expected)
        modes.add(expected)
    assert modes == ({False} if decklists == "closed" else {False, True})


@pytest.mark.parametrize("viewer", [0, 1])
def test_closed_draw_in_mixed_run_matches_closed_game(viewer: int) -> None:
    match = Match(_match("closed"))
    modes: set[bool] = set()
    for seed in range(8):
        mixed, closed = managym.Env(seed=seed), managym.Env(seed=seed)
        mixed.reset(managym.GameSetup(match.to_rust(), decklists="mixed"))
        closed.reset(match.to_rust())
        view = mixed.observation_for_player(viewer)
        modes.add(view.decklists_closed)
        if view.decklists_closed:
            _assert_same_view(view, closed.observation_for_player(viewer))
    assert modes == {False, True}


@pytest.mark.parametrize("decklists", ["closed", "mixed"])
def test_game_setup_slices_preserve_disclosure(
    decklists: DecklistSampling,
) -> None:
    setup = managym.GameSetup(
        [managym.PlayerConfig(name, {"Island": 40}) for name in ("a", "b")],
        decklists=decklists,
    )
    assert len(setup) == 2
    assert tuple(setup) == setup.players
    assert setup[0].name == "a"
    assert not isinstance(setup, list)
    for sliced in (setup[:], setup[::-1]):
        assert isinstance(sliced, managym.GameSetup)
        assert sliced.decklists == decklists
        scalar = managym.Env(seed=3)
        obs, _ = scalar.reset(sliced)
        assert obs.decklists_closed is (managym.resolve_decklists(setup, 3) == "closed")
        vector = managym.VectorEnv(num_envs=1, seed=3)
        ((vector_obs, _),) = vector.reset_all(sliced)
        assert vector_obs.decklists_closed is obs.decklists_closed
        assert vector.current_player_configs()[0].decklists == (
            "closed" if obs.decklists_closed else "open"
        )
    with pytest.raises(TypeError):
        setup[0] = setup[1]  # type: ignore[index] — the public sequence is read-only.
    with pytest.raises(AttributeError):
        setup.decklists = "open"  # type: ignore[misc] — disclosure is read-only.


def test_bare_player_list_defaults_open_without_reading_unrelated_attributes() -> None:
    class PlayerList(list[managym.PlayerConfig]):
        decklists = "closed"

    players = PlayerList(
        [managym.PlayerConfig(name, {"Island": 40}) for name in ("a", "b")]
    )
    env = managym.Env(seed=3)
    obs, _ = env.reset(players)
    assert obs.decklists_closed is False
    ((vector_obs, _),) = managym.VectorEnv(1, seed=3).reset_all(players)
    assert vector_obs.decklists_closed is False
