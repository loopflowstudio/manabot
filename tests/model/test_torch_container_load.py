"""The torch backend loads both the legacy `.pt` file and the shared container."""

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.policy.checkpoint import (
    NotAContainer,
    read_container,
    read_metadata,
)
import managym
from tests.model.checkpoint_fixture import read_checkpoint
from tests.model.prepared_routes import device_input


def _observation(space: ObservationSpace) -> dict[str, torch.Tensor]:
    env = managym.Env(seed=7)
    env.reset(
        [
            managym.PlayerConfig("Hero", {"Mountain": 20, "Gray Ogre": 20}),
            managym.PlayerConfig("Villain", {"Plains": 20, "Gray Ogre": 20}),
        ]
    )
    encoded = space.encode(env.observation_for_player(0))
    return {key: torch.from_numpy(value)[None] for key, value in encoded.items()}


@pytest.mark.parametrize("optimized", [False, True])
def test_pt_and_container_load_the_same_policy(tmp_path: Path, optimized: bool) -> None:
    torch.set_num_threads(1)
    space = ObservationSpace()
    torch.manual_seed(0)
    model = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8, value_aggregation="masked_mean", world=managym.WORLD_SETUP
        ),
    ).eval()
    pt = tmp_path / "policy.pt"
    container = tmp_path / "policy.safetensors"
    save_checkpoint(model, space, container, player_configs=Match().to_rust())
    # Retain a real historical-format fixture: only the legacy reader uses pickle.
    torch.save(read_checkpoint(container), pt)

    legacy, legacy_space = load_checkpoint(
        str(pt), bucketed_shapes=optimized, live_tokens=optimized
    )
    shared, shared_space = load_checkpoint(
        str(container), bucketed_shapes=optimized, live_tokens=optimized
    )
    assert shared_space.shapes == legacy_space.shapes == space.shapes
    assert shared.world_binding == legacy.world_binding
    expected = model.state_dict()
    for loaded in (legacy, shared):
        assert loaded.bucketed_shapes is optimized
        assert loaded.live_tokens is optimized
        state = loaded.state_dict()
        assert state.keys() == expected.keys()
        for name, value in expected.items():
            assert torch.equal(state[name], value), name

    observation = _observation(space)
    with torch.no_grad():
        reference = model(device_input(model, observation))
        for loaded in (legacy, shared):
            for actual, wanted in zip(
                loaded(device_input(loaded, observation)), reference, strict=True
            ):
                torch.testing.assert_close(
                    actual, wanted, atol=1e-6 if optimized else 0, rtol=0
                )


def test_a_file_that_is_neither_format_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "policy.bin"
    path.write_bytes(np.arange(64, dtype=np.int64).tobytes())
    with pytest.raises(NotAContainer, match="neither a PyTorch checkpoint"):
        load_checkpoint(str(path))


@pytest.mark.parametrize(
    "configuration",
    ["hashed", "no-world", "v10", "history", "categorical", "film", "no-attention"],
)
def test_container_preserves_general_policy_and_metadata(
    tmp_path: Path, configuration: str
) -> None:
    torch.set_num_threads(1)
    model = ModelConfig(hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP)
    observation = ObservationConfig()
    if configuration == "hashed":
        model.record_vocabulary = "hashed"
    elif configuration == "no-world":
        model.world = None
    elif configuration == "v10":
        observation.policy_input_version = 10
    elif configuration == "history":
        observation.policy_history_version = 1
        model.recent_events = True
    elif configuration == "categorical":
        model.value_kind = "categorical_wdl"
    elif configuration == "film":
        model.policy_film = True
    elif configuration == "no-attention":
        model.attention_on = False
    space = ObservationSpace(observation)
    agent = ActorCritic(space, model)
    path = tmp_path / "policy.safetensors"
    extra = {
        "value_semantic": "signed_outcome",
        "weights": "ema",
        "averaging": {"rate": 0.99, "iteration": 2},
    }
    save_checkpoint(agent, space, path, player_configs=Match().to_rust(), extra=extra)
    loaded, loaded_space = load_checkpoint(str(path))
    metadata = read_metadata(path)
    assert metadata["bc"] == extra
    assert metadata["global_step"] == 0
    assert "converted_from" not in metadata
    assert metadata["parameter_names"] == list(dict(agent.named_parameters()))
    assert loaded_space.encoder.config == observation
    assert loaded.value_semantic == "signed_outcome"
    assert loaded.config == model
    for name, tensor in agent.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, atol=0, rtol=0)
    container = read_container(path)
    assert set(container.buffers) == set(agent.state_dict()) - set(
        dict(agent.named_parameters())
    )


def test_container_refuses_parameter_casts(tmp_path: Path) -> None:
    space = ObservationSpace()
    agent = ActorCritic(space, ModelConfig(hidden_dim=8)).double()
    with pytest.raises(TypeError, match="parameter .*expected float32"):
        save_checkpoint(
            agent,
            space,
            tmp_path / "policy.safetensors",
            player_configs=Match().to_rust(),
        )
