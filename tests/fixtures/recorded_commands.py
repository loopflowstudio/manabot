"""Reach a native decision with an object precondition for replay tests."""

import managym
from managym.decision import DecisionFrame
from managym.recording import ScalarCapture


def record_until_object_candidate(
    env: managym.Env, capture: ScalarCapture
) -> tuple[DecisionFrame, int]:
    deck = {"Mountain": 10, "Gray Ogre": 20, "Lightning Bolt": 4}
    capture.reset(env, [managym.PlayerConfig(str(seat), deck) for seat in (0, 1)])
    for ordinal in range(64):
        frame = DecisionFrame.from_json(env.semantic_decision_frame_json())
        if frame.object_candidates:
            return frame, ordinal
        actions = env.observation_for_player(
            env.current_agent_index() or 0
        ).action_space.actions
        chosen = next(
            (
                index
                for kind in (
                    managym.ActionEnum.PRIORITY_PLAY_LAND,
                    managym.ActionEnum.PRIORITY_CAST_SPELL,
                )
                for index, action in enumerate(actions)
                if action.action_type == kind
            ),
            0,
        )
        capture.step(env, chosen)
    raise AssertionError("fixture did not reach a decision with an object candidate")
