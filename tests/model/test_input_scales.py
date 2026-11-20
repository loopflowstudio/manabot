"""What a fresh model's numeric inputs look like on sampled games.

Rows are encoded natively (`Env.encode_observation_into`) from random play of
Elemental decks and prepared with the model's readings.
"""

from collections.abc import Iterator
import dataclasses
import random

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.checkpoint import layout
from manabot.policy.prepare import Prepared, Tables, prepare
from manabot.policy.readings import (
    Readings,
    legacy_readings,
    numeric_columns,
    published_columns,
)
import managym

WORLD = "elemental-world-v2"


def sampled_batches(
    space: ObservationSpace, *, games: int, stride: int, memory: bool = False
) -> Iterator[ObservationBuffers]:
    """One-row batches from random play, every `stride` decisions."""
    decks = list(managym.authored_deck_keys(WORLD))
    rng = random.Random(3)
    for game in range(games):
        env = managym.Env(seed=500 + game, skip_trivial=True, decision_memory=memory)
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


# -- scale bound ---------------------------------------------------------------

BOUND = 4.0
# Raw integers a backend reads through its own function before any projection.
BACKEND_READ = {
    "facts.programs.numbers": "program_number (signed log) in the backend",
    "effects.detail_numbers": "program_number (signed log) in the backend",
    "known_counts": "multiplies a definition embedding, then / 60",
    "events.ages": "log1p in the backend",
    "events.counts": "the divisor of a mean",
    "group_counts": "the divisor of a mean",
}
# Not yet read by kind: their preparation still applies the legacy readings.
UNRULED = {
    "relation_features",
    "facts.columns",
    "effects.numeric",
    "memory.detail_numbers",
}
CONFIGURATIONS = {
    "default": ({}, {}),
    "history": ({"policy_history_version": 1}, {"recent_events": True}),
    "memory": (
        {
            "max_decision_records": 512,
            "max_decision_reference_details": 8192,
            "max_decision_references": 8192,
            "max_decision_members": 65536,
        },
        {"decision_memory": "transformer", "attention_on": True, "policy_film": True},
    ),
}


def float_maxima(prepared: object, prefix: str = "") -> dict[str, float]:
    """Largest |value| of every float array a `Prepared` holds, by field path."""
    maxima: dict[str, float] = {}
    for field in dataclasses.fields(prepared):  # type: ignore[arg-type]
        value = getattr(prepared, field.name)
        name = prefix + field.name
        arrays = value if isinstance(value, tuple) else (value,)
        for array in arrays:
            if isinstance(array, np.ndarray) and array.dtype.kind == "f" and array.size:
                maxima[name] = max(maxima.get(name, 0.0), float(np.abs(array).max()))
        if dataclasses.is_dataclass(value):
            for key, top in float_maxima(value, name + ".").items():
                maxima[key] = max(maxima.get(key, 0.0), top)
    return maxima


def sampled_maxima(configuration: str, *, legacy: bool) -> dict[str, float]:
    observation, model = CONFIGURATIONS[configuration]
    space = ObservationSpace(ObservationConfig(**observation))
    agent = ActorCritic(
        space,
        ModelConfig.model_validate(
            {"world": managym.WORLD_SETUP, "hidden_dim": 16, **model}
        ),
        readings=legacy_readings(space) if legacy else None,
    )
    maxima: dict[str, float] = {}
    for batch in sampled_batches(
        space, games=3, stride=5, memory=configuration == "memory"
    ):
        prepared: Prepared[np.ndarray] = agent.prepare(batch)
        for key, top in float_maxima(prepared).items():
            maxima[key] = max(maxima.get(key, 0.0), top)
    return maxima


@pytest.mark.parametrize("configuration", list(CONFIGURATIONS))
def test_fresh_inputs_stay_within_the_bound(configuration: str) -> None:
    """Every float a fresh model's projections receive stays within |x| <= 4
    on sampled games; the fields a backend reads itself are named above."""
    maxima = sampled_maxima(configuration, legacy=False)
    assert "permanents" in maxima and "action_features" in maxima
    over = {
        key: top
        for key, top in maxima.items()
        if top > BOUND and key not in BACKEND_READ and key not in UNRULED
    }
    assert over == {}


def test_legacy_inputs_exceed_the_bound() -> None:
    """The same games under the legacy readings exceed it, so the bound bites."""
    maxima = sampled_maxima("default", legacy=True)
    assert maxima["permanents"] > BOUND
    assert maxima["action_features"] > BOUND
