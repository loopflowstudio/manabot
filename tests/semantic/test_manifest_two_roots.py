"""Two roots that differ only in a manifest's hidden printing look alike to the opponent.

Search builds such roots: it forks the live game and determinizes it from the
opponent's perspective, which redraws the controller's hidden cards, the
manifest's printing included. Every consumer of the opponent's view (native and
Python observations, v11 buffers, LLM text and capture, the semantic frame and
Academy's decision surface) must be byte-equal across the two, while the
controller's own view names the printing that differs. Each root replays from
its seed, actions and determinization. Python has no library control, so these
roots come from search's own determinization; closed disclosure refuses it.
"""

import json
import re

import numpy as np
import pytest

from manabot.env.observation import (
    ObservationConfig,
    ObservationSpace,
)
from manabot.semantic.text_observation import capture_text_decision, submit_response
import managym
from managym.disclosure import DecklistDisclosure
from managym.experience_projection import decision_surface_view
from tests.fixtures.observation_rows import observation_arrays

SPACE = ObservationSpace(ObservationConfig())
CONTROLLER, OPPONENT = 0, 1
# Preferred offers, in order, until the controller has manifested a card.
SCRIPT = (
    "Keep hand",
    "Native description: Manifest",
    "Play land",
    r"Cast .*Manifest Dread",
    "Pass priority",
)


def opponent_decision_after_manifest(decklists: DecklistDisclosure) -> managym.Env:
    """Play seed 4 until the controller manifests, then stop at the opponent's decision."""
    env = managym.Env(seed=4)
    env.reset(
        managym.GameSetup(
            [
                managym.PlayerConfig(
                    "hero", {"Forest": 32, "Manifest Dread": 4, "Llanowar Elves": 4}
                ),
                managym.PlayerConfig("villain", {"Plains": 40}),
            ],
            decklists=decklists,
        )
    )
    manifested = False
    for _ in range(60):
        if manifested and env.current_agent_index() == OPPONENT:
            return env
        decision = capture_text_decision(env)
        action = next(
            a
            for pattern in SCRIPT
            for a in decision.actions
            if re.search(pattern, a.description)
        )
        manifested |= "Native description: Manifest" in action.description
        submit_response(env, decision, action.reference)
    pytest.fail("the opponent never acted after a manifest")


def manifest_printing(env: managym.Env, viewer: int) -> int:
    """The registry key `viewer` sees for the controller's face-down 2/2."""
    data = json.loads(env.observation_for_player(viewer).toJSON())
    side = "agent_permanents" if viewer == CONTROLLER else "opponent_permanents"
    (key,) = (p["registry_key"] for p in data[side] if not p["card"]["name"])
    return int(key)


def opponent_views(env: managym.Env) -> dict[str, object]:
    observation = env.observation_for_player(OPPONENT)
    text = capture_text_decision(env)
    return {
        "observation": observation.toJSON(),
        "buffers": {
            key: np.asarray(value).tobytes()
            for key, value in observation_arrays(SPACE, observation).items()
        },
        "text": text.text,
        "capture": text.capture_json,
        "frame": env.semantic_decision_frame_json(),
        "academy": json.dumps(decision_surface_view(env, OPPONENT), sort_keys=True),
    }


def two_roots(env: managym.Env) -> tuple[managym.Env, managym.Env]:
    """Two opponent-perspective determinizations with different manifest printings."""
    roots: dict[int, managym.Env] = {}
    for seed in range(32):
        branch = env.clone_env()
        branch.determinize(seed, perspective=OPPONENT)
        roots.setdefault(manifest_printing(branch, CONTROLLER), branch)
        if len(roots) == 2:
            first, second = roots.values()
            return first, second
    pytest.fail(f"32 determinizations drew only {sorted(roots)}")


def test_two_manifest_roots_are_byte_equal_to_the_opponent() -> None:
    first, second = two_roots(opponent_decision_after_manifest("open"))
    assert first.state_digest() != second.state_digest()
    assert (
        manifest_printing(first, OPPONENT) == manifest_printing(second, OPPONENT) == -1
    )

    left, right = opponent_views(first), opponent_views(second)
    for consumer in left:
        assert left[consumer] == right[consumer], consumer
    assert "(face-down permanent)" in str(left["text"])

    for root in (first, second):
        replay = managym.GameSnapshot.from_replay_json(
            root.snapshot().to_replay_json()
        ).restore()
        assert replay.state_digest() == root.state_digest()
        assert (
            replay.semantic_decision_frame_json() == root.semantic_decision_frame_json()
        )
        assert manifest_printing(replay, CONTROLLER) == manifest_printing(
            root, CONTROLLER
        )


def test_closed_decklists_build_no_sampled_roots() -> None:
    """Search cannot redraw a closed opponent's cards; the native constructed
    comparison (`dsk::manifest_dread_hides_the_printing_at_opponent_priority`)
    covers closed disclosure."""
    env = opponent_decision_after_manifest("closed")
    assert manifest_printing(env, OPPONENT) == -1
    with pytest.raises(managym.AgentError, match="decklist closed"):
        env.clone_env().determinize(0, perspective=OPPONENT)
