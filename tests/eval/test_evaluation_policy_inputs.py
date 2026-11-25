"""Policy fixtures and array transfers shared by scalar and Etude evaluation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.backends.torch.arrays import array_fields, transfer
from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic, sample_rows
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.eval.models import EvaluationKey
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.sim.matchup import SELECTED_SUITE, selected_match
from manabot.sim.registration import PlayerRegistration
from tests.eval.arena_fixture import arena_fixture
from tests.model.test_inference import _policy


def fixture_policy(
    directory: Path,
    seed: int = 83,
) -> tuple[EvaluationKey, PlayerRegistration, PlayerRegistration, Path]:
    contract = arena_fixture()
    space = ObservationSpace(ObservationConfig())
    torch.manual_seed(seed)
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8, num_attention_heads=2, world="ur-lessons-vs-gw-allies"
        ),
    )
    path = directory / f"policy-{seed}.pt"
    save_checkpoint(
        agent, space, path, player_configs=Match(selected_match()).to_rust()
    )
    runtime = runtime_fingerprints(
        match_config=selected_match(), observation_space=space
    )
    key = contract.key.model_copy(update={"content_suite": SELECTED_SUITE})
    anchor = contract.anchors[0].model_copy(
        update={
            "content_suite": SELECTED_SUITE,
            "matchup_sha256": canonical_sha256(selected_match().model_dump()),
        }
    )
    player = PlayerRegistration.model_validate(
        {
            **anchor.model_dump(),
            "player_id": f"policy-{seed}",
            "runner_kind": "checkpoint",
            "role": "challenger",
            "source_sha256": None,
            "checkpoint_sha256": file_sha256(path),
            "checkpoint_bytes": path.stat().st_size,
            "parameter_count": sum(p.numel() for p in agent.parameters()),
            "training_seed": seed,
            "artifact_id": f"fixture-{seed}",
            "evidence_class": "fixture",
            "player_spec": {
                "kind": "checkpoint",
                "deterministic": False,
                "device": "cpu",
                "batch_size": 1,
            },
            "observation_abi_sha256": runtime["observation_abi_sha256"],
            "action_abi_sha256": runtime["action_abi_sha256"],
        }
    )
    return key, player, anchor, path


def test_row_sampling_is_independent_of_neighbors_and_preserves_rng() -> None:
    logits = torch.tensor([[1.0, 2.0, -1e8], [3.0, 4.0, -1e8]])
    masks = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    before = torch.random.get_rng_state()
    batched = sample_rows((logits).numpy(), (masks).numpy(), [5, 7], False)
    assert batched == [
        sample_rows(
            (logits[i : i + 1]).numpy(), (masks[i : i + 1]).numpy(), [seed], False
        )[0]
        for i, seed in enumerate([5, 7])
    ]
    assert torch.equal(before, torch.random.get_rng_state())
    assert batched[::-1] == sample_rows(
        (logits.flip(0)).numpy(), (masks.flip(0)).numpy(), [7, 5], False
    )
    with pytest.raises(ValueError, match="invalid legal"):
        sample_rows((logits).numpy(), (torch.zeros_like(masks)).numpy(), [5, 7], False)


def test_packed_transfer_makes_one_copy_per_dtype() -> None:
    case = _policy()
    case.agent.bucketed_shapes = True
    host = case.agent.prepare(case.obs)
    arrays = array_fields(host, np.ndarray)
    device, transferred, copies = transfer(host, "meta")
    assert host.bucket_needs is not None and device.bucket_needs is host.bucket_needs
    assert transferred == sum(value.nbytes for value in arrays.values())
    moved = array_fields(device, torch.Tensor)
    assert {
        key: (value.shape, value.dtype, value.device.type)
        for key, value in moved.items()
    } == {
        key: (torch.Size(value.shape), torch.from_numpy(value).dtype, "meta")
        for key, value in arrays.items()
    }
    # Every dtype group shares exactly one transferred allocation.
    assert copies == len({value.dtype for value in moved.values()})
    assert copies == len({value.untyped_storage()._cdata for value in moved.values()})
    shared, transferred, copies = transfer(host, "cpu")
    assert (transferred, copies) == (0, 0)
    cpu = array_fields(shared, torch.Tensor)
    for key, value in cpu.items():
        source = arrays[key]
        np.testing.assert_array_equal(value.numpy(), source)
        if source.size:
            assert np.shares_memory(value.numpy(), source) == source.flags.writeable
