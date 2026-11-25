"""Combat metrics on collected decisions and boards with known log-odds."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.prepare import Prepared
from manabot.policy.readings import (
    Reading,
    fresh_readings,
    input_width,
    legacy_readings,
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
    assert columns is not None

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
    assert columns is not None
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


def test_no_combat_or_legacy_layout_omits_metrics(
    rollout: tuple[RolloutBatch, ObservationSpace],
) -> None:
    batch, space = rollout
    packed = batch.action_data
    noncombat = replace(
        packed,
        fields={
            "actions": np.zeros(
                (len(packed.slots), space.shapes["actions"][-1]), np.float32
            )
        },
        native=None,
    )
    empty = replace(batch, action_data=noncombat)
    assert combat_probe(Mock(observation_space=space), empty) == {}
    for version in (9, 10):
        with pytest.warns(DeprecationWarning):
            legacy = ObservationSpace(ObservationConfig(policy_input_version=version))
        assert combat_probe(Mock(observation_space=legacy), batch) == {}


@pytest.mark.parametrize("historical", [False, True])
def test_prepared_edits_match_reencoding_with_saved_readings(
    rollout: tuple[RolloutBatch, ObservationSpace],
    historical: bool,
) -> None:
    batch, space = rollout
    cols = combat_columns(space)
    assert cols is not None
    indexes = combat_rows(batch.action_data, cols.declare_attacker, cols.declared, 4)
    rows = batch.policy_rows(indexes).observation
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

    # Independent oracle edits the stored values, before the checkpoint reads them.
    original = {key: rows.project(key) for key in space.shapes}
    yes = original["actions"][..., cols.declared] > 0
    no = (original["actions_valid"] > 0) & ~yes

    def gaps(inputs: dict[str, np.ndarray]) -> np.ndarray:
        logits, _ = model.evaluate(model.prepare(inputs))
        return logits[yes] - logits[no]

    base = gaps(original)
    assert metrics["gap"] == pytest.approx(base.mean(), abs=1e-5)
    assert metrics["saturated"] == (np.abs(base) > 9.2).mean()
    for key, side, prefix in (
        ("agent_permanents", cols.mine, "my"),
        ("opponent_permanents", cols.theirs, "their"),
    ):
        creatures = original[key][..., side.is_creature] > 0
        for power, toughness in ((0, 1), (9, 9)):
            board = original[key].copy()
            board[..., side.power][creatures] = power / side.power_scale
            board[..., side.toughness][creatures] = toughness / side.toughness_scale
            label = f"shift/{prefix}_creatures_{power}_{toughness}"
            if prefix == "their":
                board[..., side.tapped][creatures] = power == 0
                label += "_tapped" if power == 0 else "_untapped"
            present = creatures.any(axis=1)
            if present.any():
                assert metrics[label] == pytest.approx(
                    (gaps({**original, key: board}) - base)[present].mean(), abs=1e-5
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
    assert max(len(call.args[0]) for call in sampled.policy_rows.call_args_list) <= 2


def test_combat_probe_under_bfloat16_forward_precision(
    rollout: tuple[RolloutBatch, ObservationSpace],
) -> None:
    batch, space = rollout
    policy = ActorCritic(space, ModelConfig(hidden_dim=16, num_attention_heads=2))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        metrics = combat_probe(policy, batch, limit=2)
    assert np.isfinite(metrics["combat/attack/gap"])
