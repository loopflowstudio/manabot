"""A CPU learner that owns its update and uses only the published library."""

import ast
from functools import partial
from importlib import import_module
import json
from pathlib import Path

import numpy as np
import torch

from manabot.backends.torch.bot import evaluate_observation
from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rollout import AdmittedRollout
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.training.rollout.collection import close_collector, collect, open_collector
from manabot.training.state.game_errors import GameErrorPolicy, settle
import managym


def test_external_policy_gradient_update(tmp_path: Path) -> None:
    space = ObservationSpace(ObservationConfig())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        policy = ActorCritic(space, ModelConfig(hidden_dim=16, num_attention_heads=2))
    match = Match(MatchConfig())
    errors = GameErrorPolicy(recover_capacity=True)
    collector = open_collector(
        space,
        match,
        Reward(RewardConfig()),
        num_envs=2,
        seed=17,
        opponent_mode="self",
        learning_seats="one",
        batch_closure="exact-quota",
        numerics=CollectionSession(device="cpu", seed=17),
        game_errors=errors,
        record_attempt=partial(settle, policy=errors),
    )
    try:
        batch = collect(collector, policy, 8)
    finally:
        close_collector(collector)
    assert batch.actions.shape == (8, 2)

    device = torch.device("cpu")
    admitted = AdmittedRollout(policy, batch)
    with torch.no_grad():
        tail = torch.cat(
            [
                policy.get_value(p).reshape(-1)
                for p in admitted.next_observations(device, size=2)
            ]
        )
    # One-step TD targets in [time, stream] order. The terminal action has
    # no bootstrap value, including when its stream already holds a new game.
    next_values = np.concatenate([batch.values[1:], tail.numpy()[None, :]])
    targets = batch.rewards + 0.99 * (~batch.dones) * next_values
    advantages = (targets - batch.values).reshape(-1)
    order = np.arange(batch.actions.size, dtype=np.int64)
    admitted.admit_rows(order, chunk_rows=8)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    before = [p.detach().clone() for p in policy.parameters()]
    with admitted.minibatches(order, size=8, device=device) as batches:
        for chunk, prepared, _rows in batches:
            logits, values = policy.forward_distribution(prepared)
            legal = admitted.legal(prepared).bool()
            actions = torch.as_tensor(
                batch.actions.reshape(-1)[chunk], dtype=torch.long
            )
            assert legal[torch.arange(len(chunk)), actions].all()
            distribution = torch.distributions.Categorical(
                logits=logits.masked_fill(~legal, -torch.inf)
            )
            actor = -distribution.log_prob(actions) * torch.as_tensor(advantages[chunk])
            critic = (
                values.squeeze(-1) - torch.as_tensor(targets.reshape(-1)[chunk])
            ).square()
            loss = (actor + critic).sum() / len(order)
            assert loss.isfinite()
            loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, policy.parameters(), strict=True)
    )

    path = tmp_path / "policy.safetensors"
    save_checkpoint(policy, space, path, player_configs=match.to_rust())
    loaded, loaded_space = load_checkpoint(str(path))
    obs, _ = managym.Env(seed=17).reset(match.to_rust())
    with policy.inference_scope() as scope:
        probabilities, value = evaluate_observation(scope, space, obs)
    with loaded.inference_scope() as scope:
        restored_probs, restored_value = evaluate_observation(scope, loaded_space, obs)
    np.testing.assert_allclose(restored_probs, probabilities, atol=1e-6)
    np.testing.assert_allclose(restored_value, value, atol=1e-6)
    assert len(probabilities) == len(obs.action_space.actions)
    np.testing.assert_allclose(sum(probabilities), 1.0, atol=1e-6)

    catalog = managym.catalog_tokens()
    keys = [
        key or name
        for key, name in zip(catalog["keys"], catalog["registry_names"], strict=True)
    ]
    # Read the portable safetensors header without a training-library loader.
    with path.open("rb") as checkpoint:
        length = int.from_bytes(checkpoint.read(8), "little")
        metadata = json.loads(checkpoint.read(length))["__metadata__"]
    assert keys == json.loads(metadata["writer_catalog"])["definition_keys"]


def test_consumer_uses_published_imports() -> None:
    path = Path(__file__)
    modules = json.loads((path.parents[1] / "public_api.json").read_bytes())["modules"]
    nodes = list(ast.walk(ast.parse(path.read_text())))
    engine_names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in {"managym", "manabot"}:
                    assert alias.name == "managym", (
                        "import manabot symbols from their published module"
                    )
                    engine_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and (module := node.module):
            if module.split(".")[0] in {"manabot", "managym"}:
                for alias in node.names:
                    assert alias.name in modules[module], f"{module}.{alias.name}"
                    if module != "managym":
                        assert alias.name in import_module(module).__all__
    for node in nodes:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in engine_names
        ):
            assert node.attr in modules["managym"]
