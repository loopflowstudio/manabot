"""Tests for the control-competency harness (manabot/verify/competency.py).

Scripted hero stubs play known lines through the scenario positions; the
trackers must score them the way the scenario docs say they should.
"""

from __future__ import annotations

import pytest

from manabot.verify.competency import (
    MICRO_AGGRO,
    MICRO_CONTROL,
    SCENARIOS,
    SPACE_ATTACKER,
    SPACE_BLOCKER,
    SPACE_PRIORITY,
    SPACE_TARGET,
    _find_cast_action,
    _pass_index,
    _stack_names,
    build_scenario_env,
    parse_agent_specs,
    play_probed_games,
    run_scenario_once,
)


class PassHero:
    """Passes priority, declines attacks/blocks, takes default choices."""

    def act(self, env, obs):
        raw = env.last_raw_obs
        kind = int(raw.action_space.action_space_type)
        actions = raw.action_space.actions
        if kind == SPACE_PRIORITY:
            return _pass_index(raw)
        if kind == SPACE_ATTACKER:
            return 1  # decline
        if kind == SPACE_BLOCKER:
            return len(actions) - 1  # no block
        if kind == SPACE_TARGET:
            return 0
        return len(actions) - 1


class CastWhenLegalHero(PassHero):
    """Casts the named card at the first legal opportunity, else passes."""

    def __init__(self, name: str):
        self.name = name

    def act(self, env, obs):
        raw = env.last_raw_obs
        if int(raw.action_space.action_space_type) == SPACE_PRIORITY:
            index = _find_cast_action(raw, self.name)
            if index is not None:
                return index
        return super().act(env, obs)


class CounterBombHero(PassHero):
    """Holds Counterspell until the named bomb is on the stack."""

    def __init__(self, bomb: str):
        self.bomb = bomb

    def act(self, env, obs):
        raw = env.last_raw_obs
        if int(raw.action_space.action_space_type) == SPACE_PRIORITY:
            index = _find_cast_action(raw, "Counterspell")
            if index is not None and self.bomb in _stack_names(raw):
                return index
        return super().act(env, obs)


def test_micro_decks_are_forty_cards_and_six_or_fewer_names():
    assert sum(MICRO_AGGRO.values()) == 40
    assert sum(MICRO_CONTROL.values()) == 40
    assert len(MICRO_AGGRO) <= 6
    assert len(MICRO_CONTROL) <= 6


def test_parse_agent_specs():
    specs = parse_agent_specs("random,search-16,search-256")
    assert specs[0] == {"kind": "random"}
    assert specs[1] == {"kind": "search", "sims": 16}
    assert specs[2] == {"kind": "search", "sims": 256}
    with pytest.raises(ValueError):
        parse_agent_specs("mystery-agent")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_positions_construct(name):
    """Every scenario's decks/cards are registered and the injection holds."""

    scenario = SCENARIOS[name]
    env, obs, raw = build_scenario_env(scenario, None, seed=5)
    try:
        # Exact hands.
        assert int(raw.agent.zone_counts[1]) == len(scenario.hero_hand)
        assert int(raw.opponent.zone_counts[1]) == len(scenario.villain_hand)
        # Exact battlefields.
        assert int(raw.agent.zone_counts[2]) == len(scenario.hero_battlefield)
        assert int(raw.opponent.zone_counts[2]) == len(scenario.villain_battlefield)
        # Life totals.
        assert int(raw.agent.life) == scenario.hero_life
        assert int(raw.opponent.life) == scenario.villain_life
        # The hero (seat 0) holds the first decision.
        assert int(raw.agent.player_index) == 0
    finally:
        env.close()


def test_counter_bomb_scored_correct_when_bomb_countered():
    scenario = SCENARIOS["s1_counter_the_bomb"]
    result = run_scenario_once(
        scenario, CounterBombHero("Shivan Dragon"), None, seed=3
    )
    assert result["correct"] is True
    assert result["counter_cast_on"] == "Shivan Dragon"
    assert not result["bomb_resolved"]


def test_counter_bomb_scored_wrong_when_bait_countered():
    scenario = SCENARIOS["s1_counter_the_bomb"]
    result = run_scenario_once(
        scenario, CastWhenLegalHero("Counterspell"), None, seed=3
    )
    assert result["correct"] is False
    assert result["counter_cast_on"] == "Gray Ogre"
    assert result["countered_bait"] is True
    assert result["bomb_resolved"] is True


def test_hold_wipe_scores_premature_and_delayed_casts():
    scenario = SCENARIOS["s2_hold_the_wipe"]
    # Immediate cast wipes only the two injected creatures.
    premature = run_scenario_once(
        scenario, CastWhenLegalHero("Pyroclasm"), None, seed=4
    )
    assert premature["premature"] is True
    assert premature["wiped"] == 2
    assert premature["correct"] is False
    # Never casting is scored as neither correct nor premature.
    never = run_scenario_once(scenario, PassHero(), None, seed=4)
    assert never["never_cast"] is True
    assert never["correct"] is False


def test_hold_quench_lines():
    scenario = SCENARIOS["s5_hold_up_quench"]
    # Holding the main phase, then quenching the tapped-out threat.
    held = run_scenario_once(
        scenario, CastWhenLegalHero("It'll Quench Ya!"), None, seed=6
    )
    assert held["held_main"] is True
    assert held["quenched"] is True
    assert held["correct"] is True
    # Tapping out on the 2-drop forfeits the counter window.
    tapped = run_scenario_once(
        scenario, CastWhenLegalHero("Otter-Penguin"), None, seed=6
    )
    assert tapped["held_main"] is False
    assert tapped["threat_resolved"] is True
    assert tapped["correct"] is False


def test_race_block_full_hold_is_not_correct():
    scenario = SCENARIOS["s4_race_vs_block"]
    result = run_scenario_once(scenario, PassHero(), None, seed=7)
    assert result["full_hold"] is True
    assert result["correct"] is False


class RaceHero(PassHero):
    """Attacks with everything, never blocks."""

    def act(self, env, obs):
        raw = env.last_raw_obs
        if int(raw.action_space.action_space_type) == SPACE_ATTACKER:
            return 0
        return super().act(env, obs)


class BlockLineHero(PassHero):
    """S4's documented correct line: attack fliers, distribute ground blocks."""

    def __init__(self):
        self._blocks_this_turn: dict[int, int] = {}

    def act(self, env, obs):
        from manabot.verify.competency import _battlefield_pairs, _hero_view

        raw = env.last_raw_obs
        kind = int(raw.action_space.action_space_type)
        actions = raw.action_space.actions
        if kind == SPACE_ATTACKER:
            focus = int(actions[0].focus[0]) if actions[0].focus else -1
            hero_side, _ = _hero_view(raw)
            name = next(
                (
                    str(card.name)
                    for card, perm in _battlefield_pairs(raw, hero_side)
                    if int(perm.id) == focus
                ),
                "?",
            )
            return 0 if name == "Wind Drake" else 1
        if kind == SPACE_BLOCKER:
            turn = int(raw.turn.turn_number)
            n = self._blocks_this_turn.get(turn, 0)
            self._blocks_this_turn[turn] = n + 1
            return min(n, len(actions) - 1)  # blocker k blocks attacker k
        return super().act(env, obs)


def test_race_block_documented_math_holds_end_to_end():
    """The doc's combat math, verified in the engine: racing loses on the
    villain's second attack (game turn 4); the block line wins on hero turn
    5 (game turn 9) at low life."""

    scenario = SCENARIOS["s4_race_vs_block"]
    raced = run_scenario_once(scenario, RaceHero(), None, seed=2)
    assert raced["raced"] is True
    assert raced["hero_won"] is False
    assert raced["end_turn"] == 4
    blocked = run_scenario_once(scenario, BlockLineHero(), None, seed=2)
    assert blocked["correct"] is True
    assert blocked["hero_won"] is True
    assert blocked["end_turn"] == 9


def test_bolt_threat_never_cast_is_wrong():
    scenario = SCENARIOS["s3_bolt_the_threat"]
    result = run_scenario_once(scenario, PassHero(), None, seed=8)
    assert result["never_cast"] is True
    assert result["correct"] is False


def test_probed_games_smoke():
    out = play_probed_games(
        {"kind": "random"},
        {"kind": "random"},
        hero_deck=MICRO_CONTROL,
        villain_deck=MICRO_AGGRO,
        num_games=2,
        seed=11,
    )
    assert len(out["records"]) == 2
    assert len(out["probes"]) == 2
    # Seat balance: game 0 on the play, game 1 on the draw.
    assert out["records"][0]["hero_seat"] == 0
    assert out["records"][1]["hero_seat"] == 1
    for probe in out["probes"]:
        assert probe["counter_casts"] <= probe["counter_windows"]
        # bolt_biggest counts max-power picks among multi-creature boards
        # only, so it can never exceed the number of multi-choice decisions.
        assert probe["bolt_biggest"] <= probe["bolt_multi_choice"]
        assert probe["hold_breaks"] <= probe["hold_windows"]
