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

from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.prepare import Layout, Prepared, Tables, prepare
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
                buffers = space.allocate()
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
        Layout(readings),
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


@pytest.mark.parametrize(("definition_names", "floor"), [(False, 0.15), (True, None)])
def test_power_and_toughness_move_a_whole_object_token(
    definition_names: bool, floor: float | None
) -> None:
    """Setting live creatures to 9/9 moves their whole object token (stats,
    definition, links, facts, shared effects and context, before attention)
    by a median of at least 15% at initialization in a world-bound model with
    names (measured about 20%). Adding each name's unit-variance vector onto
    definitions, as checkpoints written before the choice do, drowns the
    stats stream (about 7%), so the floor is known to bite."""
    space = ObservationSpace()
    changes = []
    for seed in range(2):
        torch.manual_seed(seed)
        agent = ActorCritic(
            space, ModelConfig(world=WORLD, definition_names=definition_names)
        ).eval()
        names = [c.name for c in published_columns(space)["agent_permanents"]]
        columns = numeric_columns(agent.readings["agent_permanents"])
        index = {names[column]: at for at, column in enumerate(columns)}
        for batch in sampled_batches(space, games=3, stride=10):
            prepared = on_device(agent.prepare(batch), torch.device("cpu"))
            permanents = prepared.permanents
            live = (permanents[:, 0, :, index["valid"]] == 1) & (
                permanents[:, 0, :, index["toughness"]] > 0
            )
            large = permanents.clone()
            for column in ("power", "toughness"):
                large[:, 0, :, index[column]][live] = 0.9
            with torch.no_grad():
                before, after = (
                    agent._gather_object_embeddings(
                        dataclasses.replace(prepared, permanents=rows)
                    ).objects
                    for rows in (permanents, large)
                )
            # Object order: two players, both seats' cards, then permanents.
            first = 2 + 2 * prepared.agent_cards.shape[1]
            seat = slice(first, first + permanents.shape[2])
            moved = after[:, seat][live] - before[:, seat][live]
            changes.append((moved.norm(dim=1) / before[:, seat][live].norm(dim=1)))
    change = torch.cat(changes)
    assert len(change) > 50
    median = float(change.median())
    if floor is not None:
        assert median >= floor
    else:
        assert median < 0.15


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
# Fields decoded by their own preparation (`OWN_PREPARATION`) that exceed the
# common bound, with the bound each keeps. Changing their preparation would
# change what stored memory checkpoints compute: it records no readings.
OWN_BOUNDS = {
    # Program ordinal / 128 (`decision_memory.prepare_memory`); ordinals reach
    # about 800 on sampled games.
    "memory.detail_numbers": 16.0,
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
    on sampled games; the fields a backend reads itself, and the few decoded by
    their own preparation with a wider bound, are named above."""
    maxima = sampled_maxima(configuration, legacy=False)
    assert "permanents" in maxima and "action_features" in maxima
    over = {
        key: top
        for key, top in maxima.items()
        if top > OWN_BOUNDS.get(key, BOUND) and key not in BACKEND_READ
    }
    assert over == {}


def test_legacy_inputs_exceed_the_bound() -> None:
    """The same games under the legacy readings exceed it, so the bound bites."""
    maxima = sampled_maxima("default", legacy=True)
    assert maxima["permanents"] > BOUND
    assert maxima["action_features"] > BOUND
    assert maxima["relation_features"] > BOUND
