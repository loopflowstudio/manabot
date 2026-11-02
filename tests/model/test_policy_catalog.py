"""Explicit native catalog binding and shared encoder dimensions, not strength."""

from pathlib import Path

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
    pool = managym.content_pool("ur-lessons-vs-gw-allies")
    pool["content_digest"] = "0" * 64
    monkeypatch.setattr(managym, "content_pool", lambda key: pool)
    with pytest.raises(ContentPackBindingError, match="differs from the live pack"):
        SemanticCardEncoder("ur-lessons-vs-gw-allies", 8)


def test_policy_catalog_requires_its_registered_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = managym.content_pool("ur-lessons-vs-gw-allies")
    pool["pack_key"] = "world"
    monkeypatch.setattr(managym, "content_pool", lambda key: pool)
    with pytest.raises(ContentPackBindingError, match="registered pack"):
        bind_policy_catalog("ur-lessons-vs-gw-allies")


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
