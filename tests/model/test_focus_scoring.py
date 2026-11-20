"""`focus_film`: offers naming one object differ by a function of it at step 0.

"Declare" and "decline" on one attacker share their focus and differ in one
flag. The additive head lets the attacker reach their gap only through ReLUs;
scaling the offer by its focused object makes the gap a function of it from
initialization. The tests by construction hold whatever the parameter draws;
the fit compares learning on one decision.
"""

import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.readings import published_columns
from manabot.verify.action_cases import decision_case
from tests.model.prepared_routes import device_input

SPACE = ObservationSpace()
COLUMNS = [column.name for column in published_columns(SPACE)["agent_permanents"]]
POWER, TOUGHNESS = COLUMNS.index("power"), COLUMNS.index("toughness")
SPEC = {
    "hidden_dim": 64,
    "num_attention_heads": 4,
    "attention_layers": 2,
    "policy_film": True,
    "value_aggregation": "masked_mean",
}


def _agent(seed: int, **fields: object) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(SPACE, ModelConfig.model_validate({**SPEC, **fields}))


def _attacker(
    obs: ObservationBuffers, power: int | torch.Tensor, toughness: int | torch.Tensor
) -> None:
    """Set the attack case's one creature (agent permanent 0) in place, one
    value for every row or one per row."""
    obs["agent_permanents"][:, 0, POWER] = power / 10
    obs["agent_permanents"][:, 0, TOUGHNESS] = toughness / 10


def test_field_is_omitted_when_off_and_keeps_every_other_draw() -> None:
    assert "focus_film" not in ModelConfig().model_dump()
    assert ModelConfig(focus_film=True).model_dump()["focus_film"] is True
    additive = _agent(5).state_dict()
    focused = _agent(5, focus_film=True).state_dict()
    assert {name for name in focused if name not in additive} == {
        "focus_film.weight",
        "focus_film.bias",
    }
    for name, value in additive.items():
        torch.testing.assert_close(focused[name], value, rtol=0, atol=0)


def test_an_unfocused_offer_is_scaled_by_exactly_one() -> None:
    obs = {
        key: value.clone() for key, value in decision_case("mode").observation.items()
    }
    obs["action_focus"].fill_(-1)
    with torch.no_grad():
        additive, focused = _agent(7), _agent(7, focus_film=True)
        torch.testing.assert_close(
            focused(device_input(focused, obs))[0],
            additive(device_input(additive, obs))[0],
            rtol=0,
            atol=0,
        )


def test_a_zero_focus_weight_is_the_additive_model() -> None:
    """With `focus_film.weight` zero (its bias starts zero) every offer is
    scaled by exactly one, focused or not: the field adds the term and
    nothing else."""
    obs = decision_case("attack").observation
    additive, focused = _agent(3), _agent(3, focus_film=True)
    assert focused.focus_film is not None
    with torch.no_grad():
        focused.focus_film.weight.zero_()
        torch.testing.assert_close(
            focused(device_input(focused, obs))[0],
            additive(device_input(additive, obs))[0],
            rtol=0,
            atol=0,
        )


def test_only_a_focused_offer_trains_the_focus_weight() -> None:
    """The attack case's two offers name the attacker; with the second's
    focus removed, its logit sends no gradient to `focus_film.weight` and the
    first's does."""
    obs = {
        key: value.clone() for key, value in decision_case("attack").observation.items()
    }
    obs["action_focus"][0, 1] = -1
    agent = _agent(3, focus_film=True)
    assert agent.focus_film is not None
    prepared = device_input(agent, obs)

    def weight_gradient(offer: int) -> torch.Tensor:
        agent.zero_grad()
        logits, _ = agent(prepared)
        logits[0, offer].backward()
        assert agent.focus_film is not None and agent.focus_film.weight.grad is not None
        return agent.focus_film.weight.grad.clone()

    assert weight_gradient(0).abs().max() > 0
    assert not weight_gradient(1).any()


def _steps_to_fit(seed: int, **fields: bool) -> int:
    """Full-batch Adam updates on "attack iff power >= 3" over 100 attackers
    until every board's label has probability at least 0.9 (at most 300)."""
    power, toughness = torch.cartesian_prod(torch.arange(10), torch.arange(10)).T
    case = decision_case("attack").observation
    batch = {key: torch.cat([value] * len(power)) for key, value in case.items()}
    _attacker(batch, power, toughness)
    labels = (power < 3).long()
    agent = _agent(seed, **fields)
    prepared = device_input(agent, batch)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    for step in range(300):
        logits, _ = agent(prepared)
        chosen = logits[:, :2].log_softmax(-1)[torch.arange(len(labels)), labels]
        if float(chosen.detach().exp().min()) >= 0.9:
            return step
        optimizer.zero_grad()
        (-chosen.mean()).backward()
        optimizer.step()
    return 300


def test_a_stat_dependent_attack_rule_fits_sooner() -> None:
    """Summed over three seeds, `focus_film` alone fits in under 0.9 of the
    historical layout's updates (measured 277 against 333) and the new layout
    in under 0.85 (measured 242)."""
    seeds = range(3)
    historical = sum(_steps_to_fit(seed) for seed in seeds)
    focused = sum(_steps_to_fit(seed, focus_film=True) for seed in seeds)
    new = sum(_steps_to_fit(seed, stream_norm=True, focus_film=True) for seed in seeds)
    assert focused < 0.9 * historical, (focused, historical)
    assert new < 0.85 * historical, (new, historical)
