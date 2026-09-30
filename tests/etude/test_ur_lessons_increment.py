"""ETU-88 custom-deck execution and the real play projection in world w4."""

import random

import numpy as np
import pytest

from etude.server import describe_actions, serialize_observation
from manabot.env import Env, Reward
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.hypers import MatchHypers, RewardHypers
import managym


def candidate_match():
    authored = MatchHypers.authored(
        "ur-lessons-vs-gw-allies", "ur_lessons", "gw_allies"
    )
    deck = dict(authored.hero_deck)
    for name, count in {
        "It'll Quench Ya!": 2,
        "First-Time Flyer": 2,
        "Pop Quiz": 1,
        "Igneous Inspiration": 1,
    }.items():
        deck[name] -= count
        if not deck[name]:
            del deck[name]
    for name, count in {
        "Combustion Technique": 2,
        "Proft's Eidetic Memory": 1,
        "Gran-Gran": 1,
        "Accumulate Wisdom": 1,
    }.items():
        deck[name] = deck.get(name, 0) + count
    assert sum(deck.values()) == 40
    assert sum(authored.hero_deck.values()) == 41
    return Match(MatchHypers(**{**authored.model_dump(), "hero_deck": deck}))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_match_hypers_candidate_finishes_with_complete_native_and_python_encoding(
    reverse, seed
):
    match = candidate_match()
    if reverse:
        match = match.swapped()
    space = ObservationSpace()
    assert space.encoder.permanent_dim == 25
    assert managym.WORLD_VERSION == "w4"
    env = Env(match, space, Reward(RewardHypers()), seed=seed)
    obs, _ = env.reset(seed=seed)
    assert env.content_pack_manifest()["world_version"] == "w4"
    rng = random.Random(seed)
    for _ in range(5000):
        actions = np.flatnonzero(obs["actions_valid"])
        obs, _, terminal, truncated, info = env.step(int(rng.choice(actions)))
        assert not truncated
        assert not info.get("action_truncated", False)
        if terminal:
            break
    else:
        pytest.fail("candidate game did not terminate")


def test_play_offers_and_registered_rules_text_cover_all_three_cards():
    env = managym.Env(seed=0, skip_trivial=False)
    obs, _ = env.reset(candidate_match().to_rust())
    while int(obs.turn.step) != 3:
        actions = describe_actions(obs)
        action = next(a for a in actions if a["type"] == "PRIORITY_PASS_PRIORITY")
        obs, *_ = env.step(action["index"])
    env.scenario_clear_hand(0)
    for name in ["Gran-Gran", "Proft's Eidetic Memory", "Combustion Technique"]:
        env.scenario_force_card_in_hand(0, name)
    for name in ["Island", "Island", "Mountain", "Mountain"]:
        env.scenario_force_battlefield(0, name, True)
    env.scenario_force_battlefield(1, "Water Tribe Rallier", True)
    obs = env.scenario_refresh()
    payload = serialize_observation(obs)
    text = {card["name"]: card["text_box"] for card in payload["agent"]["hand"]}
    assert "then discard a card" in text["Gran-Gran"]
    assert "no maximum hand size" in text["Proft's Eidetic Memory"]
    assert "exile it instead" in text["Combustion Technique"]
    labels = {action["description"] for action in describe_actions(obs)}
    for name in text:
        assert f"Cast {name}" in labels


@pytest.mark.parametrize("legend", [False, True])
def test_play_copy_distinguishes_mandatory_discard_and_legend_choice(legend):
    env = managym.Env(seed=0, skip_trivial=False)
    env.reset(candidate_match().to_rust())
    if legend:
        # A custom setup supplies the second copy, without changing authored decks.
        match = candidate_match()
        match.hero_deck["Gran-Gran"] = 2
        env.reset(match.to_rust())
        env.scenario_force_battlefield(0, "Gran-Gran", True)
        env.scenario_force_battlefield(0, "Gran-Gran", True)
    else:
        env.scenario_force_battlefield(0, "Gran-Gran", True)
    obs = env.scenario_refresh()
    for _ in range(200):
        kind = int(obs.action_space.action_space_type)
        if kind == (12 if legend else 11):
            descriptions = [a["description"] for a in describe_actions(obs)]
            assert descriptions
            assert all(
                label.startswith("Keep " if legend else "Discard ")
                for label in descriptions
            )
            assert all("into your hand" not in label for label in descriptions)
            root = env.state_digest()
            scores, simulations, cap_hits = env.flat_mc_scores(1, 1, 88, 5000)
            assert len(scores) == simulations == len(descriptions)
            assert cap_hits == 0
            assert env.state_digest() == root
            break
        actions = describe_actions(obs)
        chosen = next(
            (a for a in actions if a["type"] == "DECLARE_ATTACKER" and a["declared"]),
            None,
        )
        if chosen is None:
            chosen = next(
                (a for a in actions if a["type"] == "PRIORITY_PASS_PRIORITY"),
                actions[0],
            )
        obs, _, done, truncated, _ = env.step(chosen["index"])
        assert not done and not truncated
    else:
        pytest.fail("required choice was not reached")


def test_world_identity_records_w4_and_refuses_earlier_runtime_labels():
    from manabot.sim.teacher1_evidence import runtime_fingerprints
    from scripts.train_challenger import world_identity

    identity = world_identity({})
    assert identity["version"] == "w4"
    assert identity["content_manifest"]["world_version"] == "w4"
    assert identity["dims"] == {
        "player": 28,
        "card": 39,
        "permanent": 25,
        "action_types": 16,
    }
    for old_world in ("w2", "w3"):
        with pytest.raises(ValueError, match="cannot register"):
            runtime_fingerprints(world=old_world)
