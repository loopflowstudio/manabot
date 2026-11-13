"""The torch backend loads both the legacy `.pt` file and the shared container."""

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.admission import admit_observation
from manabot.policy.checkpoint import (
    NotAContainer,
    checkpoint_metadata,
    from_torch_state,
    write_container,
)
import managym


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


def _convert(pt: Path, target: Path) -> None:
    record = torch.load(pt, map_location="cpu", weights_only=False)
    hypers = record["hypers"]
    container = from_torch_state(
        ModelConfig.from_checkpoint(hypers["agent_hypers"]),
        ObservationConfig.from_checkpoint(hypers["observation_hypers"]),
        {name: value.numpy() for name, value in record["model_state_dict"].items()},
        checkpoint_metadata(record, pt),
    )
    write_container(target, container)


@pytest.mark.parametrize("optimized", [False, True])
def test_pt_and_container_load_the_same_policy(tmp_path: Path, optimized: bool) -> None:
    space = ObservationSpace()
    torch.manual_seed(0)
    model = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8, value_aggregation="masked_mean", world=managym.WORLD_SETUP
        ),
    ).eval()
    pt = tmp_path / "policy.pt"
    save_checkpoint(model, space, pt, player_configs=None)
    container = tmp_path / "policy.safetensors"
    _convert(pt, container)

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
        reference = model(admit_observation(model, observation))
        for loaded in (legacy, shared):
            for actual, wanted in zip(
                loaded(admit_observation(loaded, observation)), reference, strict=True
            ):
                torch.testing.assert_close(
                    actual, wanted, atol=1e-6 if optimized else 0, rtol=0
                )


def test_a_file_that_is_neither_format_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "policy.bin"
    path.write_bytes(np.arange(64, dtype=np.int64).tobytes())
    with pytest.raises(NotAContainer, match="neither a PyTorch checkpoint"):
        load_checkpoint(str(path))
