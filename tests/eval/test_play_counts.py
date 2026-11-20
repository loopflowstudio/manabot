"""Exact counts for repeated offers and distinct declarations in replay."""

import json

from manabot.eval.behaviour import Play, PlayCounts
import managym
from managym.card_capture import CapturedCards
from managym.enums import ActionEnum as Action, ActionSpaceEnum as Space
from managym.replay import RecordedDecision, ReplayStop


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
            "turn": 4,
            "final_cards": [],
            "capture": {
                "version": 1,
                "content_digest": "0" * 64,
                "initial": [],
                "opening": [[], []],
                "moves": [],
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
            casts=2,
            games=1,
            turns=4,
        ),
        PlayCounts(
            1,
            blockers_offered=2,
            blockers_declared=1,
            casts=1,
            casts_off_turn=1,
            games=1,
            turns=4,
        ),
    ]
    assert play.end(ReplayStop("interrupted", 10), capture) == []
