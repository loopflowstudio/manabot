"""What a fresh model's numeric inputs look like on sampled games.

Rows are encoded natively (`Env.encode_observation_into`) from random play of
Elemental decks and prepared with the model's readings.
"""

from collections.abc import Iterator
import random

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.checkpoint import layout
from manabot.policy.prepare import Tables, prepare
from manabot.policy.readings import (
    Readings,
    legacy_readings,
    numeric_columns,
    published_columns,
)
import managym

WORLD = "elemental-world-v2"


def sampled_batches(
    space: ObservationSpace, *, games: int, stride: int
) -> Iterator[ObservationBuffers]:
    """One-row batches from random play, every `stride` decisions."""
    decks = list(managym.authored_deck_keys(WORLD))
    rng = random.Random(3)
    for game in range(games):
        env = managym.Env(seed=500 + game, skip_trivial=True)
        setups = [managym.authored_deck_setup(WORLD, d) for d in rng.sample(decks, 2)]
        raw, _ = env.reset(setups)
        for step in range(400):
            if step % stride == 0:
                buffers = space.encoder.allocate()
                env.encode_observation_into(raw, buffers)
                yield {key: value[None] for key, value in buffers.items()}
            raw, _, terminated, truncated, _ = env.step(
                rng.randrange(len(raw.action_space.actions))
            )
            if terminated or truncated:
                break


def creature_rows(
    space: ObservationSpace, agent: ActorCritic, readings: Readings
) -> tuple[np.ndarray, int, int]:
    """Prepared numeric rows of live creatures, with the power and toughness
    positions among the numeric columns."""
    tables = Tables.from_buffers(
        {name: value.detach().numpy() for name, value in agent.named_buffers()},
        layout(readings),
    )
    names = [column.name for column in published_columns(space)["agent_permanents"]]
    columns = numeric_columns(readings["agent_permanents"])
    index = {names[column]: position for position, column in enumerate(columns)}
    rows = []
    for batch in sampled_batches(space, games=4, stride=10):
        permanents = prepare(batch, tables).permanents
        flat = permanents.reshape(-1, permanents.shape[-1])
        rows.append(
            flat[(flat[:, index["valid"]] == 1) & (flat[:, index["toughness"]] > 0)]
        )
    return np.concatenate(rows), index["power"], index["toughness"]


@pytest.mark.parametrize(("fresh", "floor"), [(True, 0.25), (False, None)])
def test_power_and_toughness_move_a_creature_token(
    fresh: bool, floor: float | None
) -> None:
    """Setting a creature to 9/9 moves its numeric permanent embedding by at
    least a quarter at initialization; under the legacy readings the raw
    timestamp drowns it, so the floor is known to bite."""
    space = ObservationSpace()
    agent = ActorCritic(space, ModelConfig())
    readings = agent.readings if fresh else legacy_readings(space)
    rows, power, toughness = creature_rows(space, agent, readings)
    assert len(rows) > 50
    changes = []
    for seed in range(4):
        torch.manual_seed(seed)
        embedding = ActorCritic(space, ModelConfig()).perm_embedding
        base = torch.from_numpy(rows)
        large = base.clone()
        large[:, power] = large[:, toughness] = 0.9
        with torch.no_grad():
            before, after = embedding(base), embedding(large)
        changes.append(((after - before).norm(dim=1) / before.norm(dim=1)).numpy())
    median = float(np.median(np.concatenate(changes)))
    if floor is not None:
        assert median >= floor
    else:
        assert median < 0.25
