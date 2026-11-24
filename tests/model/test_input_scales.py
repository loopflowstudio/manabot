"""What a fresh model's numeric inputs look like on sampled games.

Rows are encoded natively (`encode_observation`) from random play of Elemental
decks and prepared with the model's readings.
"""

from collections.abc import Iterator
import dataclasses
import random

import numpy as np
import pytest
import torch

from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import (
    ObservationBuffers,
    ObservationSpace,
    encode_observation,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.prepare import Prepared
from manabot.policy.readings import (
    READ_BY_READINGS,
    fresh_readings,
    input_width,
    legacy_readings,
    published_columns,
)
from manabot.verify.text_observation import keep_opening_hands
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
        raw = keep_opening_hands(env, raw)
        for step in range(400):
            if step % stride == 0:
                yield {
                    key: value[None]
                    for key, value in encode_observation(space, raw).items()
                }
            raw, _, terminated, truncated, _ = env.step(
                rng.randrange(len(raw.action_space.actions))
            )
            if terminated or truncated:
                break


def nine_nine(
    batch: ObservationBuffers, space: ObservationSpace, keys: tuple[str, ...]
) -> tuple[ObservationBuffers, np.ndarray]:
    """The batch with every live creature of `keys` set to 9/9 as the encoder
    stores it (tenths), and which rows of each key [B, len(keys), P] those are."""
    names = [c.name for c in published_columns(space)["agent_permanents"]]
    valid, power, toughness = (
        names.index(name) for name in ("valid", "power", "toughness")
    )
    large = dict(batch)
    live = []
    for key in keys:
        rows = batch[key].copy()
        creature = (rows[..., valid] == 1) & (rows[..., toughness] > 0)
        rows[..., power][creature] = rows[..., toughness][creature] = 0.9
        large[key] = rows
        live.append(creature)
    return large, np.stack(live, 1)


@pytest.mark.parametrize(("fresh", "floor"), [(True, 0.25), (False, None)])
def test_power_and_toughness_move_a_creature_token(
    fresh: bool, floor: float | None
) -> None:
    """Setting a creature to 9/9 moves its numeric permanent embedding by at
    least a quarter at initialization; under the legacy readings the raw
    timestamp drowns it, so the floor is known to bite."""
    space = ObservationSpace()
    readings = None if fresh else legacy_readings(space)
    keys = ("agent_permanents", "opponent_permanents")
    pairs = []
    agent = ActorCritic(space, ModelConfig(), readings=readings)
    for batch in sampled_batches(space, games=4, stride=10):
        large, live = nine_nine(batch, space, keys)
        if live.any():
            pairs.append(
                tuple(
                    torch.from_numpy(agent.prepare(rows).permanents[live])
                    for rows in (batch, large)
                )
            )
    assert sum(len(base) for base, _ in pairs) > 50
    changes = []
    for seed in range(4):
        torch.manual_seed(seed)
        embedding = ActorCritic(space, ModelConfig(), readings=readings).perm_embedding
        with torch.no_grad():
            for base, large in pairs:
                before, after = embedding(base), embedding(large)
                changes.append(
                    ((after - before).norm(dim=1) / before.norm(dim=1)).numpy()
                )
    median = float(np.median(np.concatenate(changes)))
    if floor is not None:
        assert median >= floor
    else:
        assert median < 0.25


@pytest.mark.parametrize(
    ("definition_names", "legacy", "floor"),
    [(False, False, 0.4), (True, False, 0.15), (False, True, None)],
)
def test_power_and_toughness_move_a_whole_object_token(
    definition_names: bool, legacy: bool, floor: float | None
) -> None:
    """Setting live creatures to 9/9 moves their whole object token (stats,
    definition, links, facts, shared effects and context, before attention)
    at initialization in a world-bound model: by a median of at least 40%
    (measured 60%), and at least 15% with each name's unit-variance vector
    added onto definitions (measured 21%). Read as stored tenths, as the
    legacy readings do, the stats move it by about 6%, so the floor bites."""
    space = ObservationSpace()
    changes = []
    for seed in range(2):
        torch.manual_seed(seed)
        agent = ActorCritic(
            space,
            ModelConfig(world=WORLD, definition_names=definition_names),
            readings=legacy_readings(space) if legacy else None,
        ).eval()
        for batch in sampled_batches(space, games=3, stride=10):
            large, live = nine_nine(batch, space, ("agent_permanents",))
            live = torch.from_numpy(live[:, 0])
            with torch.no_grad():
                before, after = (
                    agent._gather_object_embeddings(
                        on_device(agent.prepare(rows), torch.device("cpu"))
                    ).objects
                    for rows in (batch, large)
                )
            # Object order: two players, both seats' cards, then permanents.
            first = 2 + 2 * batch["agent_cards"].shape[1]
            seat = slice(first, first + live.shape[1])
            moved = after[:, seat][live] - before[:, seat][live]
            changes.append((moved.norm(dim=1) / before[:, seat][live].norm(dim=1)))
    change = torch.cat(changes)
    assert len(change) > 50
    median = float(change.median())
    if floor is not None:
        assert median >= floor
    else:
        assert median < 0.15


def test_player_mana_is_scalar_and_combat_stats_keep_their_thresholds() -> None:
    space = ObservationSpace()
    readings = fresh_readings(space)
    columns = published_columns(space)
    for key, width, pools in (("agent_player", 99, 31), ("player_turn_state", 16, 6)):
        mana = [
            r
            for col, r in zip(columns[key], readings[key], strict=True)
            if col.kind == "mana"
        ]
        assert mana == ["as_is"] * pools
        assert sum(map(input_width, readings[key])) == width
    for key, names in (
        (
            "agent_permanents",
            ("power", "toughness", "damage", "mana_value", "ward_cost"),
        ),
        ("actions", ("x",)),
    ):
        by_name = {
            col.name: str(r) for col, r in zip(columns[key], readings[key], strict=True)
        }
        assert all(by_name[name] == "steps:8" for name in names)


def test_no_category_or_selector_is_a_magnitude() -> None:
    """A fresh model reads every category and selector as flags, a join, a
    lookup or a mask; none reaches a projection as one number."""
    space = ObservationSpace()
    readings = fresh_readings(space)
    scalar = [
        f"{key}.{column.name}: {reading}"
        for key, columns in published_columns(space).items()
        if key in READ_BY_READINGS
        for column, reading in zip(columns, readings[key], strict=True)
        if column.kind in ("category", "selector") and input_width(reading) == 1
    ]
    assert scalar == []


@pytest.mark.parametrize(("legacy", "apart"), [(False, True), (True, False)])
def test_a_two_two_and_a_three_three_are_a_flag_apart(
    legacy: bool, apart: bool
) -> None:
    """On sampled creatures, making each a 2/2 and then a 3/3 moves its
    prepared row by at least one flag flip (distance 1); stored tenths move it
    by 0.14."""
    space = ObservationSpace()
    agent = ActorCritic(
        space, ModelConfig(), readings=legacy_readings(space) if legacy else None
    )
    names = [c.name for c in published_columns(space)["agent_permanents"]]
    power, toughness = names.index("power"), names.index("toughness")
    distances = []
    for batch in sampled_batches(space, games=2, stride=10):
        _, live = nine_nine(batch, space, ("agent_permanents",))
        rows = []
        for stat in (0.2, 0.3):
            sized = dict(batch)
            sized["agent_permanents"] = batch["agent_permanents"].copy()
            for column in (power, toughness):
                sized["agent_permanents"][..., column][live[:, 0]] = stat
            rows.append(agent.prepare(sized).permanents[:, 0][live[:, 0]])
        distances.append(np.linalg.norm(rows[1] - rows[0], axis=-1))
    distance = np.concatenate(distances)
    assert len(distance) > 10
    assert bool((distance >= 1).all()) is apart


def test_an_x_decision_offers_numbers_apart_from_modes() -> None:
    """Choosing X for United Front with three spare mana offers 0..3. Each
    offer is marked `has_x` and carries its value in `x`, so a fresh model
    reads X = 1 and X = 3 as different amounts rather than only as mode flags;
    a spell's cast offer at priority carries neither."""
    space = ObservationSpace()
    columns = [c.name for c in published_columns(space)["actions"]]
    has_x, x = columns.index("has_x"), columns.index("x")
    env = managym.Env(seed=7, skip_trivial=True)
    deck = {"Plains": 36, "United Front": 4}
    raw, _ = env.reset([managym.PlayerConfig(name, dict(deck)) for name in ("a", "b")])
    keep_opening_hands(env, raw)
    env.scenario_clear_hand(0)
    env.scenario_force_card_in_hand(0, "United Front")
    for _ in range(5):
        env.scenario_force_battlefield(0, "Plains", ready=True)
    obs = env.scenario_refresh()

    priority = encode_observation(space, obs)
    assert not priority["actions"][:, [has_x, x]].any()
    cast = next(
        index
        for index, action in enumerate(obs.action_space.actions)
        if action.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
    )
    obs, *_ = env.step(cast)
    assert [a.mode_index for a in obs.action_space.actions] == [0, 1, 2, 3]
    batch = encode_observation(space, obs)
    rows = batch["actions"][batch["actions_valid"].astype(bool)]
    np.testing.assert_array_equal(rows[:, has_x], 1.0)
    np.testing.assert_allclose(rows[:, x], [0.0, 0.1, 0.2, 0.3])

    model = ActorCritic(space, ModelConfig())
    inputs = {key: value[None] for key, value in batch.items()}
    features = model.prepare(inputs).action_features
    one, three = features[1], features[3]
    inputs["actions"] = inputs["actions"].copy()
    inputs["actions"][..., [has_x, x]] = 0
    modes_only = model.prepare(inputs).action_features
    # X adds distance between offers beyond what their mode flags give.
    assert np.linalg.norm(three - one) > np.linalg.norm(modes_only[3] - modes_only[1])


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
