"""Open and closed decklists through the match, encoders, text and Academy wire."""

from dataclasses import asdict
import random
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import DECKLIST_DIM, ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.managym.experience_projection import serialize_observation
from manabot.policy.prepare import Prepared
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

    # Python reference and native encoders agree.
    space = ObservationSpace(ObservationConfig())
    assert obs.decklists_open == (decklists == "open")
    for viewer in (0, 1):
        view = env.observation_for_player(viewer)
        encoded = space.encode(view)
        native_buffers = env.encode_observation(view)
        for key in ("agent_player", "opponent_player"):
            assert encoded[key][0, -1] == float(decklists == "open")
            np.testing.assert_array_equal(encoded[key], native_buffers[key])
    python = space.encode(obs)["decklists"]
    native = env.encode_observation(obs)["decklists"]
    assert python.shape == native.shape == (2, 60, DECKLIST_DIM)
    np.testing.assert_array_equal(python, native)
    assert python[0, :, 1].sum() == 41
    assert python[1, :, 1].sum() == (40 if decklists == "open" else 0)
    assert not python[1].any() if decklists == "closed" else python[1].any()

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
    with pytest.warns(DeprecationWarning, match="0.4.0"), pytest.raises(ValueError):
        managym.PlayerConfig("x", {"Island": 1}, disclosure="partial")


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
    torch.set_num_threads(1)
    space = ObservationSpace()
    policy = ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2))

    def game(land: str, disclosure: Decklists) -> managym.Env:
        decks = [{"Forest": 40}, {land: 40}]
        if viewer == 1:
            decks.reverse()
        env = managym.Env(seed=3)
        env.reset(
            Match(
                MatchConfig(
                    hero_deck=decks[0], villain_deck=decks[1], decklists=disclosure
                )
            ).to_rust()
        )
        return env

    def prepared(obs: managym.Observation) -> Prepared[np.ndarray]:
        return policy.prepare({k: v[None] for k, v in space.encode(obs).items()})

    def assert_same_view(a: managym.Observation, b: managym.Observation) -> None:
        assert a.toJSON() == b.toJSON()
        buffers = [space.encode(obs) for obs in (a, b)]
        for key, value in buffers[0].items():
            np.testing.assert_array_equal(value, buffers[1][key], err_msg=key)
        np.testing.assert_equal(asdict(prepared(a)), asdict(prepared(b)))

    # Open control: the substitution is visible everywhere it could leak.
    opened = [
        game(land, "open").observation_for_player(viewer)
        for land in ("Island", "Mountain")
    ]
    assert opened[0].toJSON() != opened[1].toJSON()
    assert not np.array_equal(
        space.encode(opened[0])["decklists"], space.encode(opened[1])["decklists"]
    )
    assert not np.array_equal(
        prepared(opened[0]).deck_rows, prepared(opened[1]).deck_rows
    )

    left, right = game("Island", "closed"), game("Mountain", "closed")
    assert_same_view(*(env.observation_for_player(viewer) for env in (left, right)))
    decisions = 0
    # All-land decks have identical action ordering until the opponent plays a
    # land. Equal sizes preserve shuffle RNG and the viewer's private cards.
    for _ in range(100):
        a, b = (env.observation_for_player(viewer) for env in (left, right))
        if a.opponent_permanents:
            break
        if left.current_agent_index() == viewer:
            assert_same_view(a, b)
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
    torch.set_num_threads(1)
    space = ObservationSpace()
    policy = ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2))
    env = managym.Env(seed=0)
    obs, _ = env.reset(Match(_match("closed")).to_rust())
    rng = random.Random(0)
    owner: int | None = None
    revealed: int | None = None
    prepared_row: int | None = None
    for _ in range(3000):
        if owner is not None:
            view = env.observation_for_player(1 - owner)
            encoded = space.encode(view)
            assert not encoded["decklists"][1].any()
            prepared = policy.prepare({k: v[None] for k, v in encoded.items()})
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


def test_deprecated_seat_disclosure_rejects_disagreement() -> None:
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        first = managym.PlayerConfig("first", {"Island": 12}, disclosure="closed")
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        second = managym.PlayerConfig("second", {"Mountain": 12}, disclosure="open")
    with pytest.raises(ValueError, match="per-seat disclosure values disagree"):
        managym.Env(seed=7).reset([first, second])


@pytest.mark.parametrize(
    ("hint", "decklists"),
    [("closed", "open"), ("open", "closed"), ("closed", "mixed"), ("open", "mixed")],
)
def test_deprecated_seat_disclosure_rejects_game_conflict(
    hint: Decklists,
    decklists: Literal["open", "closed", "mixed"],
) -> None:
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        first = managym.PlayerConfig("first", {"Island": 12}, disclosure=hint)
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        second = managym.PlayerConfig("second", {"Mountain": 12}, disclosure=hint)
    # Omission exercises the constructor's ordinary open default.
    setup = (
        managym.GameSetup([first, second])
        if decklists == "open"
        else managym.GameSetup([first, second], decklists=decklists)
    )
    with pytest.raises(ValueError, match="conflicts with GameSetup.decklists"):
        managym.Env(seed=7).reset(setup)
    with pytest.raises(ValueError, match="conflicts with GameSetup.decklists"):
        managym.VectorEnv(num_envs=1, seed=7).reset_all(setup)


@pytest.mark.parametrize("hint", ["open", "closed"])
@pytest.mark.parametrize("hinted_seat", [0, 1])
def test_deprecated_seat_disclosure_ignores_unset_hints(
    hint: Decklists,
    hinted_seat: int,
) -> None:
    players = [
        managym.PlayerConfig("first", {"Island": 12}),
        managym.PlayerConfig("second", {"Mountain": 12}),
    ]
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        players[hinted_seat].disclosure = hint
    for setup in (players, managym.GameSetup(players, decklists=hint)):
        observation, _ = managym.Env(seed=7).reset(setup)
        assert observation.decklists_open is (hint == "open")
        ((observation, _),) = managym.VectorEnv(num_envs=1, seed=7).reset_all(setup)
        assert observation.decklists_open is (hint == "open")


@pytest.mark.parametrize("decklists", ["closed", "mixed"])
def test_game_setup_rewrap_preserves_disclosure(
    decklists: Literal["closed", "mixed"],
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
        assert observation.decklists_open is expected
        ((observation, _),) = managym.VectorEnv(num_envs=1, seed=seed).reset_all(
            wrapped
        )
        assert observation.decklists_open is expected
        modes.add(expected)
    assert modes == ({False} if decklists == "closed" else {False, True})


@pytest.mark.parametrize("viewer", [0, 1])
def test_closed_draw_in_mixed_run_matches_closed_game(viewer: int) -> None:
    torch.set_num_threads(1)
    space = ObservationSpace()
    policy = ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2))
    match = Match(_match("closed"))
    modes: set[bool] = set()
    for seed in range(8):
        mixed, closed = managym.Env(seed=seed), managym.Env(seed=seed)
        setup = match.to_rust()
        setup = managym.GameSetup(setup, decklists="mixed")
        mixed.reset(setup)
        modes.add(mixed.observation_for_player(viewer).decklists_open)
        closed.reset(match.to_rust())
        view = mixed.observation_for_player(viewer)
        if view.decklists_open:
            continue
        control = closed.observation_for_player(viewer)
        assert view.toJSON() == control.toJSON()
        for key, value in space.encode(view).items():
            np.testing.assert_array_equal(value, space.encode(control)[key])
        left = policy.prepare({k: v[None] for k, v in space.encode(view).items()})
        right = policy.prepare({k: v[None] for k, v in space.encode(control).items()})
        np.testing.assert_equal(asdict(left), asdict(right))
    assert modes == {False, True}


@pytest.mark.parametrize("decklists", ["closed", "mixed"])
def test_game_setup_slices_preserve_disclosure(
    decklists: Literal["closed", "mixed"],
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
        assert obs.decklists_open is (managym.resolve_decklists(setup, 3) == "open")
        vector = managym.VectorEnv(num_envs=1, seed=3)
        ((vector_obs, _),) = vector.reset_all(sliced)
        assert vector_obs.decklists_open is obs.decklists_open
        assert vector.current_player_configs()[0].decklists == (
            "open" if obs.decklists_open else "closed"
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
    assert obs.decklists_open is True
