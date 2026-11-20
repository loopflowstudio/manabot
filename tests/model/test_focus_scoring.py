"""`focus_film`: offers naming one object differ by a function of it at step 0.

"Declare" and "decline" on one attacker share their focus and differ in one
flag. The additive head lets the attacker reach their gap only through ReLUs,
so it barely depends on the attacker; scaling the offer by its focused object
makes the gap a function of it from initialization.
"""

import numpy as np
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


def _gap(agent: ActorCritic, obs: ObservationBuffers) -> float:
    with torch.no_grad():
        logits, _ = agent(device_input(agent, obs))
    return float(logits[0, 0] - logits[0, 1])


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


def test_the_attack_gap_moves_with_the_attacker() -> None:
    """Over 64 seeds, 2/2 to 9/9 moves the declare/decline gap further with
    `focus_film` than without: the median move is larger (measured 1.25-2.2x
    across 32-seed windows) and so is the move on most seeds.

    `focus_film` is built last, so each pair shares every other weight and
    differs only by the focus term."""
    case = decision_case("attack").observation
    big = {key: value.clone() for key, value in case.items()}
    _attacker(big, 9, 9)

    def moves(**fields: object) -> np.ndarray:
        return np.array(
            [
                abs(_gap(agent, big) - _gap(agent, case))
                for agent in (_agent(seed, **fields) for seed in range(64))
            ]
        )

    additive, focused = moves(), moves(focus_film=True)
    assert np.median(focused) >= 1.2 * np.median(additive)
    assert (focused > additive).mean() > 0.55


def _steps_to_fit(**fields: bool) -> int:
    """Full-batch Adam updates on "attack iff power >= 3" over 100 attackers
    until every board's label has probability at least 0.9 (at most 300)."""
    power, toughness = torch.cartesian_prod(torch.arange(10), torch.arange(10)).T
    case = decision_case("attack").observation
    batch = {key: torch.cat([value] * len(power)) for key, value in case.items()}
    _attacker(batch, power, toughness)
    labels = (power < 3).long()
    agent = _agent(0, **fields)
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
    """The new layout fits in under 0.85 of the historical layout's updates
    (measured 91/131 on this seed; 0.62-0.78 across three seeds)."""
    new, historical = _steps_to_fit(stream_norm=True, focus_film=True), _steps_to_fit()
    assert new < 0.85 * historical, (new, historical)
