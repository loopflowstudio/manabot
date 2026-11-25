"""Exact counts for repeated offers and distinct declarations in replay."""

import json
from pathlib import Path

import pytest

from manabot.eval.behaviour import Play, PlayCounts, checkpoint_counts
from manabot.verify.text_observation import keep_opening_index
import managym
from managym.card_capture import CapturedCards
from managym.enums import ActionEnum as Action, ActionSpaceEnum as Space
from managym.recording import ScalarCapture, read_records
from managym.replay import RecordedDecision, ReplayStop
from managym.statistics import reduce


def test_play_counts_one_opportunity_per_own_turn() -> None:
    env = managym.Env(seed=3)
    env.reset([managym.PlayerConfig(name, {"Forest": 12}) for name in ("a", "b")])
    play = Play()
    play.start([])

    def choose(
        seat: int, turn: int, kind: Space, offers: list[Action], index: int
    ) -> None:
        view = json.loads(env.observation_for_player(seat).toJSON())
        view["turn"]["turn_number"] = turn
        view["agent"]["is_active"] = seat == 0
        view["action_space"] = {
            "type": kind,
            "actions": [
                {
                    "type": action,
                    "mode_index": None,
                    "continuation_index": None,
                    "ability_index": None,
                    "declared": i > 0 if kind != Space.PRIORITY else None,
                    "focus": [],
                }
                for i, action in enumerate(offers)
            ],
        }
        obs = managym.Observation.from_json(json.dumps(view))
        play.decision(RecordedDecision(0, obs, index, None))

    priority = [
        Action.PRIORITY_PASS_PRIORITY,
        Action.PRIORITY_PLAY_LAND,
        Action.PRIORITY_CAST_SPELL,
    ]
    for index in (0, 1, 2, 2):
        choose(0, 1, Space.PRIORITY, priority, index)
    choose(1, 1, Space.PRIORITY, priority[::2], 1)
    choose(0, 3, Space.PRIORITY, priority, 0)
    for index in (1, 0):
        choose(0, 3, Space.DECLARE_ATTACKER, [Action.DECLARE_ATTACKER] * 2, index)
    # Two legal block targets and a decline are one blocker opportunity.
    choose(1, 3, Space.DECLARE_BLOCKER, [Action.DECLARE_BLOCKER] * 3, 2)
    choose(1, 3, Space.DECLARE_BLOCKER, [Action.DECLARE_BLOCKER] * 2, 0)

    capture = CapturedCards.model_validate(
        {
            "turn": 13,
            "final_cards": [],
            "capture": {
                "version": 1,
                "content_digest": "0" * 64,
                "initial": [],
                "opening": [[], []],
                "moves": [],
                "exposure": {
                    "choices": [],
                    "resolutions": [],
                    "attackers_committed": [1, 0],
                    "attackers_lost_for_nothing": [1, 0],
                    "blockers_committed": [0, 1],
                    "blockers_lost_for_nothing": [0, 1],
                    "mana_turns": [
                        [
                            {"available": available, "paid": paid}
                            for available, paid in [
                                (0, 0),
                                (2, 1),
                                (3, 0),
                                (4, 4),
                                (5, 0),
                                (0, 0),
                                (2, 3),
                            ]
                        ],
                        [],
                    ],
                },
            },
        }
    )
    assert play.end(ReplayStop("complete", 10), capture) == [
        PlayCounts(
            0,
            land_turns=2,
            lands_played=1,
            cast_turns=2,
            casts_made=1,
            attackers_offered=2,
            attackers_declared=1,
            blockers_committed=0,
            attackers_committed=1,
            attackers_lost_for_nothing=1,
            mana_turns_early=2,
            mana_use_early=0.5,
            mana_turns_mid=2,
            mana_use_mid=1.0,
            mana_turns_late=1,
            mana_use_late=1.5,
            casts=2,
            games=1,
            turns=13,
        ),
        PlayCounts(
            1,
            blockers_offered=2,
            blockers_declared=1,
            attackers_committed=0,
            blockers_committed=1,
            blockers_lost_for_nothing=1,
            casts=1,
            casts_off_turn=1,
            games=1,
            turns=13,
        ),
    ]
    assert play.end(ReplayStop("interrupted", 10), capture) == []


@pytest.mark.parametrize("mandatory", [True, False], ids=["mandatory-block", "menace"])
def test_replayed_chump_blocks_count_committed_assignments(
    tmp_path: Path, mandatory: bool
) -> None:
    # Each deck casts one creature. The Spider attacks under Taunting Challenge;
    # the Inspector has menace and the lone Lions' chosen assignment is cleared.
    attack_turn, attacker, deck = (
        (9, "Giant Spider", {"Forest": 9, "Giant Spider": 3, "Taunting Challenge": 3})
        if mandatory
        else (5, "Callous Inspector", {"Swamp": 11, "Callous Inspector": 3})
    )
    setup = managym.GameSetup(
        [
            managym.PlayerConfig("attacker", deck),
            managym.PlayerConfig("blocker", {"Plains": 9, "Savannah Lions": 4}),
        ]
    )
    path = tmp_path / "combat.jsonl"
    env = managym.Env(seed=3)
    recording = ScalarCapture(seed=3, games_path=path)
    obs, _ = recording.reset(env, setup)
    while obs.turn.turn_number == 0:
        obs, *_ = recording.step(env, keep_opening_index(obs))
    while not env.is_game_over():
        actions = obs.action_space.actions
        turn = obs.turn.turn_number
        cards = {c.id: c.characteristics.name for c in obs.agent_cards}
        creatures = {p.characteristics.name: p.id for p in obs.agent_permanents}
        for i, action in enumerate(actions):
            kind = action.action_type
            if kind in (Action.PRIORITY_PLAY_LAND, Action.PAY_COST):
                break
            if kind == Action.PRIORITY_CAST_SPELL:
                name = cards[action.focus[0]]
                if name == "Taunting Challenge" and turn == attack_turn:
                    break
                if name in (attacker, "Savannah Lions") and name not in creatures:
                    break
            if kind == Action.CHOOSE_TARGET and action.focus == [
                creatures.get(attacker)
            ]:
                break
            if kind == Action.DECLARE_ATTACKER and action.declared == (
                turn == attack_turn
            ):
                break
            if kind == Action.DECLARE_BLOCKER and action.declared:
                break
            if kind == Action.PRIORITY_PASS_PRIORITY:
                break
        else:
            pytest.fail(
                f"unexpected fixture decision {obs.action_space.action_space_type}"
            )
        obs, *_ = recording.step(env, i)
    recording.close()

    [seats] = reduce(read_records(path), [Play()])
    attack, block = seats
    checkpoint = checkpoint_counts((attack, block), game_seat=1)
    assert attack.attackers_committed == attack.attackers_declared == 1
    assert checkpoint["seat"] == 0 and checkpoint["game_seat"] == 1
    assert block.blockers_offered == block.blockers_declared == int(not mandatory)
    assert checkpoint["blockers_committed"] == int(mandatory)
    assert checkpoint["chump_blocks"] == int(mandatory)
    if mandatory:
        assert checkpoint["chump_blocks"] / checkpoint["blockers_committed"] == 1
