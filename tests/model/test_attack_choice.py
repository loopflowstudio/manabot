"""Attack/decline must remain learnable through the native policy input path."""

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.env import Match, ObservationSpace
from manabot.env.observation import ACTION_DECLARED_INDEX, ACTION_TYPE_DIM
from manabot.infra.hypers import AgentSpec
from manabot.model import Agent
from manabot.model.world import checkpoint_world
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.execution import execute_regime
from manabot.training.experiments import Experiment
from manabot.training.presets import ataraxos_mtg_v1
from manabot.training.references import reference_distribution
from manabot.verify.store import VerifyStore
import managym


@pytest.fixture
def attack_observation() -> dict[str, torch.Tensor]:
    env = managym.Env(seed=190, skip_trivial=False)
    env.reset(
        [
            managym.PlayerConfig("a", {"Gray Ogre": 36, "Mountain": 4}),
            managym.PlayerConfig("b", {"Gray Ogre": 36, "Mountain": 4}),
        ]
    )
    env.scenario_clear_hand(0)
    env.scenario_force_battlefield(0, "Gray Ogre", ready=True)
    raw = env.scenario_refresh()
    for _ in range(100):
        if (
            raw.action_space.action_space_type
            == managym.ActionSpaceEnum.DECLARE_ATTACKER
        ):
            break
        index = next(
            (
                i
                for i, a in enumerate(raw.action_space.actions)
                if a.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
            ),
            0,
        )
        raw, _, done, _, _ = env.step(index)
        assert not done
    else:
        pytest.fail("fixture did not reach an attack decision")
    assert [a.declared for a in raw.action_space.actions] == [True, False]
    native = env.encode_observation(raw)
    scalar = ObservationSpace().encode(raw)
    for key in native:
        np.testing.assert_array_equal(native[key], scalar[key])
    np.testing.assert_array_equal(native["action_focus"][0], native["action_focus"][1])
    np.testing.assert_array_equal(native["actions"][:2, ACTION_DECLARED_INDEX], [1, 0])
    np.testing.assert_array_equal(
        native["actions"][:2, :ACTION_TYPE_DIM].sum(-1), [1, 1]
    )
    return {key: torch.as_tensor(value).unsqueeze(0) for key, value in native.items()}


@pytest.mark.parametrize("preferred", [0, 1])
def test_native_attack_choice_can_learn_either_preference(
    attack_observation: dict[str, torch.Tensor],
    preferred: int,
) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(17)
            agent = Agent(
                ObservationSpace(),
                AgentSpec(hidden_dim=8, num_attention_heads=2, attention_on=False),
            )
        # Erasing the declaration reproduces the old, weight-independent tie.
        aliased = {k: v.clone() for k, v in attack_observation.items()}
        aliased["actions"][..., ACTION_DECLARED_INDEX] = 0
        logits, _ = agent(aliased)
        torch.testing.assert_close(logits[0, 0], logits[0, 1], rtol=0, atol=0)

        optimizer = torch.optim.Adam(agent.parameters(), lr=0.02)
        for _ in range(40):
            logits, _ = agent(attack_observation)
            loss = -logits.log_softmax(-1)[0, preferred]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits, _ = agent(attack_observation)
            assert float(logits.softmax(-1)[0, preferred]) > 0.95
            permuted = {k: v.clone() for k, v in attack_observation.items()}
            for key in ("actions", "actions_valid", "action_focus"):
                permuted[key][:, :2] = permuted[key][:, [1, 0]]
            swapped, _ = agent(permuted)
            torch.testing.assert_close(swapped[:, :2], logits[:, [1, 0]])
        for kind in ("uniform", "action_type_uniform"):
            prior = reference_distribution(attack_observation, kind)
            torch.testing.assert_close(prior[0, :2], torch.tensor([0.5, 0.5]))
            assert not prior[0, 2:].any()
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("old_world,width", [("w4", 17), ("w5", 18)])
def test_old_checkpoint_is_rejected_before_loading_weights(
    tmp_path: Path, old_world: str, width: int
) -> None:
    space = ObservationSpace()
    binding = checkpoint_world(Match().to_rust(), space)
    assert binding["world"] == "w6"
    binding["world"] = old_world
    binding["input_schema"]["actions"]["shape"][-1] = width
    path = tmp_path / "w4.pt"
    torch.save(
        {
            "world_binding": binding,
            "hypers": {
                "agent_hypers": AgentSpec().model_dump(),
                "observation_hypers": space.encoder.hypers.model_dump(),
            },
            "model_state_dict": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="compatible w6 world binding"):
        load_checkpoint_agent(str(path))


def test_historical_recipe_resolves_but_cannot_execute(tmp_path: Path) -> None:
    baseline = ataraxos_mtg_v1()
    recipe = Experiment("historical", baseline).resolve().cases[0].regime
    assert recipe.world == baseline.regime().world == "w4"
    out = tmp_path / "must-not-start"
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(ValueError, match="world differs from native runtime"):
            execute_regime(recipe, 17, out, store)
    assert not out.exists()
