"""Explicit native catalog binding and shared encoder dimensions, not strength."""

from copy import deepcopy
from pathlib import Path
from typing import Literal

from naming_reference import packed_definitions
import pytest
import torch

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint
from manabot.model.semantic_cards import SemanticCardEncoder
from manabot.model.world import checkpoint_world
from manabot.semantic.learning import ContentPackBindingError
from manabot.semantic.policy_catalog import bind_policy_catalog
from manabot.sim.matchup import AuthoredMatchup
import managym


@pytest.mark.parametrize("selection", ["full", "reordered", "empty"])
@pytest.mark.parametrize("width", [16, 64])
def test_program_execution_preserves_values_and_parameter_gradients(
    selection: Literal["full", "reordered", "empty"],
    width: int,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(243)
    # Use double precision for the unnormalized whole-catalog gradient oracle:
    # chunked input-weight reductions change float32 summation order. Actual
    # float32 policy losses/gradients are checked in test_catalog_naming.py.
    encoder = SemanticCardEncoder("world", width).double()
    reference = deepcopy(encoder)
    rows = torch.arange(1, len(encoder.tokens) + 1)
    if selection == "reordered":
        # Include the longest program, short programs and noncatalog ordering.
        rows = torch.argsort(encoder.lengths, descending=True)[::5] + 1
    elif selection == "empty":
        rows = rows[:0]
    actual = encoder.definitions(None if selection == "full" else rows)
    expected = packed_definitions(reference, None if selection == "full" else rows)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    if not len(rows):
        assert not actual.any()
        return
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for (name, parameter), (refname, old) in zip(
        encoder.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == refname
        torch.testing.assert_close(
            parameter.grad, old.grad, atol=1e-11, rtol=1e-9, msg=name
        )
    # Padding contributes no learning signal, even after long real programs.
    assert encoder.token_embedding.weight.grad is not None
    assert not encoder.token_embedding.weight.grad[0].any()


def test_catalog_uses_actual_native_ids_and_shared_encoder() -> None:
    key = "ur-lessons-vs-gw-allies"
    pack = bind_policy_catalog(key)
    encoder = SemanticCardEncoder(key, 8)
    shape = encoder.catalog_shape
    assert shape.definitions == pack.definition_count
    assert shape.unique_tokens == len(encoder.token_vocabulary)
    assert shape.expanded_tokens == int(encoder.lengths.sum())
    assert shape.padded_tokens == encoder.tokens.numel()
    assert encoder.token_embedding.num_embeddings == shape.unique_tokens + 1
    definitions = encoder.definitions()
    assert definitions.shape == (shape.definitions + 1, 8)
    for definition, row in pack.definition_row_by_card_def_id.items():
        assert int(encoder._rows(torch.tensor([definition + 1.0]))[0]) == row + 1
    definitions[1:].square().sum().backward()
    assert encoder.token_embedding.weight.grad is not None
    assert encoder.encoder.weight_ih_l0.grad is not None
    assert torch.isfinite(encoder.encoder.weight_ih_l0.grad).all()


def test_unknown_catalog_is_not_a_name_alias() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        SemanticCardEncoder("simple-subset", 8)


def test_native_pool_digest_must_match_the_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = managym.content_pool("world")
    pool["content_digest"] = "0" * 64
    monkeypatch.setattr(managym, "content_pool", lambda key: pool)
    with pytest.raises(ContentPackBindingError, match="differs from the live pack"):
        SemanticCardEncoder("ur-lessons-vs-gw-allies", 8)


def test_every_registered_name_binds_the_one_world_catalog() -> None:
    world = bind_policy_catalog("world")
    assert world.definition_count == managym.CARD_DEFINITION_COUNT
    for key in ("ur-lessons-vs-gw-allies", "elemental-world-v2"):
        pack = bind_policy_catalog(key)
        assert pack.content_pack_hash == world.content_pack_hash
        assert pack.binding_hash == world.binding_hash


@pytest.mark.parametrize(
    "roles",
    [("l1_swift", "l1_heavy"), ("l2_swift", "l2_heavy"), ("ur_lessons", "gw_allies")],
)
def test_research_checkpoint_binds_common_catalog(
    roles: tuple[str, str], tmp_path: Path
) -> None:
    key = "mtg122-complexity-v1"
    roster = AuthoredMatchup(pack_key=key, role_a=roles[0], role_b=roles[1])
    space = ObservationSpace()
    torch.manual_seed(12251)
    agent = ActorCritic(
        space, ModelConfig(hidden_dim=8, num_attention_heads=2, world=key)
    )
    payload = {
        "model_state_dict": agent.state_dict(),
        "hypers": {
            "agent_hypers": agent.config.model_dump(),
            "observation_hypers": space.encoder.config.model_dump(),
        },
        "world_binding": checkpoint_world(Match(roster.match()).to_rust(), space),
    }
    path = tmp_path / "research.pt"
    torch.save(payload, path)
    loaded, _ = load_checkpoint(str(path))
    assert loaded.world_binding == payload["world_binding"]
    for name, tensor in agent.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])
    catalog = bind_policy_catalog(key).catalog
    assert managym.learning_token_value("Giant") in set(catalog.token_value.tolist())
