"""FiLM policy conditioning: historical identity at rest, interaction when trained.

New runs train FiLM; an omitted field stays additive. FiLM must not perturb the
additive model's initialization, receipts or permutation contracts, and must
give state-conditioned offer choices a gradient.
"""

import pytest
import torch

<<<<<<< HEAD
from manabot.config.load import load_train_config
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentSpec, Hypers
from manabot.model import Agent
from manabot.training.baseline import baseline
=======
from manabot.env import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.model import ActorCritic
from manabot.recipes.baseline import baseline
from manabot.recipes.local import local_smoke
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
from manabot.verify.action_cases import decision_case, reorder_objects, reorder_offers

SPEC = {"hidden_dim": 32, "num_attention_heads": 2, "attention_feedforward_dim": 64}


<<<<<<< HEAD
def _agent(seed: int, film: bool) -> Agent:
    torch.manual_seed(seed)
    return Agent(ObservationSpace(), AgentSpec(**SPEC, policy_film=film))


def test_disabled_film_preserves_serialized_spec() -> None:
    assert "policy_film" not in AgentSpec(**SPEC).model_dump()
    assert AgentSpec(**SPEC, policy_film=True).model_dump()["policy_film"] is True
    with pytest.raises(ValueError, match="compound"):
        AgentSpec(compound_decisions=True, policy_film=True)


def test_new_runs_train_film_and_omitted_fields_stay_additive() -> None:
    assert Hypers().agent.policy_film
    assert load_train_config("local").agent.policy_film
    recipe = baseline(1)
    assert recipe.version == "baseline-v2" and recipe.regime.agent.policy_film
    assert not AgentSpec.model_validate({"hidden_dim": 64}).policy_film
    assert not Hypers.model_validate({"agent": {}}).agent.policy_film
=======
def _agent(seed: int, film: bool) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(ObservationSpace(), ModelConfig(**SPEC, policy_film=film))


def test_disabled_film_preserves_serialized_spec() -> None:
    assert "policy_film" not in ModelConfig(**SPEC).model_dump()
    assert ModelConfig(**SPEC, policy_film=True).model_dump()["policy_film"] is True
    with pytest.raises(ValueError, match="compound"):
        ModelConfig(compound_decisions=True, policy_film=True)


def test_new_runs_train_film_and_omitted_fields_stay_additive() -> None:
    assert local_smoke().agent.policy_film
    recipe = baseline(1)
    assert recipe.version == "baseline-v3" and recipe.regime.agent.policy_film
    assert not ModelConfig.model_validate({"hidden_dim": 64}).policy_film
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f


def test_film_starts_as_the_additive_policy() -> None:
    additive, film = _agent(17, False), _agent(17, True)
    shared = additive.state_dict()
    for name, value in film.state_dict().items():
        if name.startswith("policy_film."):
            assert not value.any()
        else:
            torch.testing.assert_close(value, shared[name], rtol=0, atol=0)
    # skip_init draws nothing: later RNG consumers see the historical stream.
    after_additive = torch.rand(1)
    _agent(17, True)
    torch.testing.assert_close(torch.rand(1), after_additive)
    obs = decision_case("mode").observation
    with torch.no_grad():
        torch.testing.assert_close(film(obs)[0], additive(obs)[0], rtol=0, atol=0)


def test_film_gradient_couples_offer_and_state() -> None:
    """Opposite labels on two states give the state code a nonzero gradient."""
    film = _agent(4, True)
    cases = [decision_case("mode", preference=p) for p in (0, 1)]
    batch = {
        k: torch.cat([c.observation[k] for c in cases]) for k in cases[0].observation
    }
    labels = torch.tensor([c.preferred for c in cases])
    logits, _ = film(batch)
    torch.nn.functional.cross_entropy(logits, labels).backward()
    assert film.policy_film is not None and film.policy_film.weight.grad is not None
    assert film.policy_film.weight.grad.abs().max() > 1e-6


def test_film_preserves_offer_and_object_order_invariance() -> None:
    film = _agent(29, True)
    assert film.policy_film is not None
    with torch.no_grad():
        film.policy_film.weight.normal_()
        film.policy_film.bias.normal_()
    film.eval()
    obs = decision_case("learn").observation
    count = int(obs["actions_valid"].sum())
    order = torch.randperm(count, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        expected = film(obs)[0].softmax(-1)
        offers = film(reorder_offers(obs, order))[0].softmax(-1)
        objects = film(reorder_objects(obs))[0].softmax(-1)
    torch.testing.assert_close(offers[:, :count], expected[:, order], atol=1e-6, rtol=0)
    torch.testing.assert_close(objects, expected, atol=1e-6, rtol=0)
