"""`focus_film`: offers naming one object differ by a function of it at step 0.

"Declare" and "decline" on one attacker share their focus and differ in one
flag. The additive head lets the attacker reach their gap only through ReLUs,
so it barely depends on the attacker; scaling the offer by its focused object
makes the gap a function of it from initialization.
"""

import itertools

import numpy as np
import pytest
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


def _attacker(obs: ObservationBuffers, power: int, toughness: int) -> None:
    """Set the attack case's one creature (agent permanent 0) in place."""
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
    """Median over seeds: 2/2 to 9/9 moves the declare/decline gap at least
    twice as far with `focus_film` (measured 2.6x)."""
    case = decision_case("attack").observation
    big = {key: value.clone() for key, value in case.items()}
    _attacker(big, 9, 9)

    def moves(**fields: object) -> float:
        agents = [_agent(seed, **fields) for seed in range(8)]
        return float(np.median([abs(_gap(a, big) - _gap(a, case)) for a in agents]))

    assert moves(focus_film=True) >= 2 * moves()


@pytest.mark.parametrize(
    ("fields", "fits"),
    [({"stream_norm": True, "focus_film": True}, True), ({}, False)],
)
def test_a_stat_dependent_attack_rule_fits_sooner(
    fields: dict[str, bool], fits: bool
) -> None:
    """Full-batch Adam on "attack iff power >= 3" over 100 attackers: the new
    layout fits in 120 steps (measured 81-92); the historical one does not
    (measured 149-158)."""
    boards = list(itertools.product(range(10), range(10)))
    case = decision_case("attack").observation
    batch = {key: torch.cat([value] * len(boards)) for key, value in case.items()}
    for row, (power, toughness) in enumerate(boards):
        batch["agent_permanents"][row, 0, POWER] = power / 10
        batch["agent_permanents"][row, 0, TOUGHNESS] = toughness / 10
    labels = torch.tensor([0 if power >= 3 else 1 for power, _ in boards])
    agent = _agent(0, world="elemental-world-v2", **fields)
    prepared = device_input(agent, batch)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    for _ in range(120):
        logits, _ = agent(prepared)
        chosen = logits[:, :2].log_softmax(-1)[torch.arange(len(labels)), labels]
        if float(chosen.detach().exp().min()) >= 0.9:
            break
        optimizer.zero_grad()
        (-chosen.mean()).backward()
        optimizer.step()
    assert (float(chosen.detach().exp().min()) >= 0.9) == fits
