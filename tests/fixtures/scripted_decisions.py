"""Encode visible life markers and one- or two-offer decisions for scripted streams."""

from collections.abc import Sequence

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.policy.spans import BatchRow
import managym


def marked_rows(
    space: ObservationSpace, markers: Sequence[int], counts: Sequence[int] | None = None
) -> list[BatchRow]:
    root = managym.Env(seed=0)
    root.reset(Match().to_rust())
    actor = root.current_agent_index()
    assert actor is not None
    scale = space.layout.arrays["agent_player"]["columns"]["life"]["scale"]
    observations = []
    for i, marker in enumerate(markers):
        raw = root.observation_for_player(actor)
        player = raw.agent
        player.life = marker * scale
        raw.agent = player
        if counts is not None:
            choices = raw.action_space
            choices.actions = choices.actions[: counts[i]]
            raw.action_space = choices
        observations.append(raw)
    batch = managym.encode_observation_batch(observations, space.layout)
    return [(batch, i) for i in range(len(markers))]
