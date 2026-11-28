"""Combat metrics on collected decisions and boards with known log-odds."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.prepare import Prepared
from manabot.policy.readings import (
    Reading,
    fresh_readings,
    input_width,
    legacy_readings,
    read_numeric,
)
from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.combat_probe import (
    combat_columns,
    combat_probe,
    combat_rows,
    gap_shifts,
)
from manabot.training.loops.self_play import (
    create,
    device_settings,
    self_play_collector,
)
from manabot.training.rollout.collection import RolloutBatch, close_collector, collect


@pytest.fixture(scope="module")
def rollout() -> tuple[RolloutBatch, ObservationSpace]:
    settings = gray_ogre_self_play(horizon=2)
    with device_settings(settings):
        learner, _ = create(settings, seed=1)
        collector = self_play_collector(settings, seed=1, start=0)
        try:
            batch = collect(collector, learner.agent, settings.transitions)
        finally:
            close_collector(collector)
    return batch, learner.agent.observation_space


def scoring_policy(space: ObservationSpace, board_aware: bool) -> Mock:
    """Declare with gap 3, optionally adding my power minus their untapped power."""

    columns = combat_columns(space)

    readings = dict(fresh_readings(space))
    for key in ("agent_permanents", "opponent_permanents"):
        readings[key] = tuple(
            Reading.AS_IS
            if i in (columns.mine.power, columns.mine.toughness)
            else reading
            for i, reading in enumerate(readings[key])
        )
    policy = Mock(
        wraps=ActorCritic(
            space, ModelConfig(hidden_dim=8, num_attention_heads=2), readings=readings
        ),
        observation_space=space,
        readings=readings,
    )

    def index(key: str, column: int) -> int:
        return sum(map(input_width, readings[key][:column]))

    def evaluate(rows: Prepared[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        gap = np.full(rows.batch, 3.0)
        if board_aware:
            for seat, (key, cols, sign) in enumerate(
                (
                    ("agent_permanents", columns.mine, 1),
                    ("opponent_permanents", columns.theirs, -1),
                )
            ):
                board = rows.permanents[:, seat]
                power = board[..., index(key, cols.power)] * cols.power_scale
                available = board[..., index(key, cols.is_creature)] > 0
                if seat == 1:
                    available &= board[..., index(key, cols.tapped)] == 0
                gap += sign * (available * power).sum(axis=1)
        logits = np.zeros(rows.legal.shape)
        logits[rows.legal] = (
            rows.action_features[:, index("actions", columns.declared)]
            * gap[rows.legal_batches]
        )
        return logits, np.zeros(rows.batch)

    policy.evaluate.side_effect = evaluate
    return policy


def test_played_combat_distinguishes_lock_from_board_dependence(
    rollout: tuple[RolloutBatch, ObservationSpace],
) -> None:
    batch, space = rollout
    columns = combat_columns(space)
    blind = combat_probe(scoring_policy(space, False), batch, limit=16)
    aware = combat_probe(scoring_policy(space, True), batch, limit=16)
    for kind, column in (
        ("attack", columns.declare_attacker),
        ("block", columns.declare_blocker),
    ):
        prefix = f"combat/{kind}/"
        probed = len(combat_rows(batch.action_data, column, columns.declared, 16))
        assert probed > 0
        assert blind[prefix + "decisions"] == probed
        assert blind[prefix + "gap"] == 3
        assert 0 < blind[prefix + "shift_decisions/theirs"] <= probed
        assert blind[prefix + "shift_decisions/mine"] == probed
        for label, sign in (
            ("my_creatures_0_1", -1),
            ("my_creatures_9_9", 1),
            ("their_creatures_0_1_tapped", 1),
            ("their_creatures_9_9_untapped", -1),
        ):
            key = prefix + "shift/" + label
            assert blind[key] == 0
            assert aware[key] * sign > 0


def test_no_combat_omits_metrics(
    rollout: tuple[RolloutBatch, ObservationSpace],
) -> None:
    batch, space = rollout
    packed = batch.action_data
    cols = combat_columns(space)
    # Two-offer decisions that are neither an attack nor a block declaration.
    two = np.flatnonzero(np.diff(packed.offsets) == 2)
    kinds = packed.action_columns(
        np.concatenate([packed.offsets[two], packed.offsets[two] + 1]),
        (cols.declare_attacker, cols.declare_blocker),
    ).reshape(2, len(two), 2)
    [candidates] = np.nonzero((kinds == 0).all(axis=(0, 2)))
    assert len(candidates)
    row = packed.row(int(two[candidates[0]]))
    noncombat = type(packed).concatenate([row] * len(np.diff(packed.offsets)))
    empty = replace(batch, action_data=noncombat)
    assert combat_probe(Mock(observation_space=space), empty) == {}


@pytest.mark.parametrize("historical", [False, True])
def test_prepared_edits_match_reencoding_with_saved_readings(
    rollout: tuple[RolloutBatch, ObservationSpace],
    historical: bool,
) -> None:
    batch, space = rollout
    cols = combat_columns(space)
    indexes = combat_rows(batch.action_data, cols.declare_attacker, cols.declared, 4)
    rows = batch.observations(indexes).observation
    model = ActorCritic(
        space,
        ModelConfig(hidden_dim=16, num_attention_heads=2),
        readings=legacy_readings(space) if historical else None,
        bucketed_shapes=True,
    )
    policy = Mock(wraps=model, readings=model.readings)
    metrics = gap_shifts(
        policy, [rows.select(slice(0, 3)), rows.select(slice(3, None))], cols
    )
    assert policy.prepare.call_count == 2

    # Independent oracle: edit the stored values, then read them with the
    # checkpoint's readings into the prepared rows they occupy.
    original = {key: rows.project(key) for key in space.shapes}
    yes = original["actions"][..., cols.declared] > 0
    no = (original["actions_valid"] > 0) & ~yes
    prepared = model.prepare(rows)

    def gaps(inputs: Prepared[np.ndarray]) -> np.ndarray:
        logits, _ = model.evaluate(inputs)
        return logits[yes] - logits[no]

    base = gaps(prepared)
    assert metrics["gap"] == pytest.approx(base.mean(), abs=1e-5)
    assert metrics["saturated"] == (np.abs(base) > 9.2).mean()
    capacity = prepared.permanents.shape[2]
    for seat, (key, side, prefix) in enumerate(
        (
            ("agent_permanents", cols.mine, "my"),
            ("opponent_permanents", cols.theirs, "their"),
        )
    ):
        creatures = original[key][:, :capacity, side.is_creature] > 0
        for power, toughness in ((0, 1), (9, 9)):
            board = original[key][:, :capacity].copy()
            board[..., side.power][creatures] = power / side.power_scale
            board[..., side.toughness][creatures] = toughness / side.toughness_scale
            label = f"shift/{prefix}_creatures_{power}_{toughness}"
            if prefix == "their":
                board[..., side.tapped][creatures] = power == 0
                label += "_tapped" if power == 0 else "_untapped"
            edited = replace(prepared, permanents=prepared.permanents.copy())
            edited.permanents[:, seat][creatures] = read_numeric(
                board[creatures], model.readings[key]
            )
            present = creatures.any(axis=1)
            if present.any():
                assert metrics[label] == pytest.approx(
                    (gaps(edited) - base)[present].mean(), abs=1e-5
                )
            else:
                assert label not in metrics


def test_bucketed_and_live_token_policy_matches_padded_probe(
    rollout: tuple[RolloutBatch, ObservationSpace],
) -> None:
    batch, space = rollout
    config = ModelConfig(
        world="elemental-world-v2",
        hidden_dim=16,
        attention_layers=1,
        num_attention_heads=2,
        attention_on=True,
        policy_film=True,
        focus_film=True,
        stream_norm=True,
    )
    torch.manual_seed(7)
    padded = ActorCritic(space, config)
    packed = ActorCritic(space, config, bucketed_shapes=True, live_tokens=True)
    packed.load_state_dict(padded.state_dict())
    sampled = Mock(wraps=batch, action_data=batch.action_data)
    assert combat_probe(packed, sampled, limit=4, execution_rows=2) == pytest.approx(
        combat_probe(padded, batch, limit=4), abs=1e-5
    )
    assert max(len(call.args[0]) for call in sampled.observations.call_args_list) <= 2


def test_combat_probe_under_bfloat16_forward_precision(
    rollout: tuple[RolloutBatch, ObservationSpace],
) -> None:
    batch, space = rollout
    policy = ActorCritic(space, ModelConfig(hidden_dim=16, num_attention_heads=2))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        metrics = combat_probe(policy, batch, limit=2)
    assert np.isfinite(metrics["combat/attack/gap"])
