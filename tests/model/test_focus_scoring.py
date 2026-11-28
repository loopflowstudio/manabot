"""`focus_film`: offers naming one object differ by a function of it at step 0.

"Declare" and "decline" on one attacker share their focus and differ in one
flag. The additive head lets the attacker reach their gap only through ReLUs;
scaling the offer by its focused object makes the gap a function of it from
initialization. The tests by construction hold whatever the parameter draws;
the fit compares learning on one decision.
"""

from dataclasses import replace

import numpy as np
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace, encode_policy_batch
from manabot.infra.config import ModelConfig
from manabot.policy.prepare import Prepared
from manabot.policy.readings import published_columns, write_numeric_column
from manabot.verify.action_cases import decision_case
from manabot.verify.choice_inputs import encode
import managym
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


def _agent(
    seed: int, *, bucketed_shapes: bool = False, **fields: object
) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(
        SPACE,
        ModelConfig.model_validate({**SPEC, **fields}),
        bucketed_shapes=bucketed_shapes,
    )


def _attacker(
    agent: ActorCritic,
    obs: managym.Observation,
    power: torch.Tensor,
    toughness: torch.Tensor,
) -> Prepared[np.ndarray]:
    """Copies of `obs` whose one creature (agent permanent 0) has the given
    power and toughness per row, set in the prepared numeric readings."""
    batch = encode_policy_batch(SPACE, [obs] * len(power))
    prepared = agent.prepare(batch)
    board = prepared.permanents.copy()
    readings = agent.readings["agent_permanents"]
    for row, (p, t) in enumerate(zip(power.tolist(), toughness.tolist(), strict=True)):
        write_numeric_column(board[row, 0, 0], slice(None), readings, POWER, p / 10)
        write_numeric_column(board[row, 0, 0], slice(None), readings, TOUGHNESS, t / 10)
    return replace(prepared, permanents=board)


def test_field_is_omitted_when_off_and_keeps_every_other_draw() -> None:
    assert "focus_film" not in ModelConfig().model_dump()
    assert ModelConfig(focus_film=True).model_dump()["focus_film"] is True
    additive = _agent(5).state_dict()
    focused = _agent(5, focus_film=True).state_dict()
    assert {name for name in focused if name not in additive} == {
        "focus_film.weight",
    }
    for name, value in additive.items():
        torch.testing.assert_close(focused[name], value, rtol=0, atol=0)


def test_an_unfocused_offer_is_scaled_by_exactly_one() -> None:
    observation = decision_case("mode").observation
    with torch.no_grad():
        additive, focused = _agent(7), _agent(7, focus_film=True)
        assert focused.focus_film is not None
        focused.focus_film.weight.add_(1)

        def unfocused(agent: ActorCritic) -> Prepared[np.ndarray]:
            prepared = agent.prepare(encode(observation, SPACE), inference=True)
            return replace(prepared, focus_mask=np.zeros_like(prepared.focus_mask))

        torch.testing.assert_close(
            focused(device_input(focused, unfocused(focused)))[0],
            additive(device_input(additive, unfocused(additive)))[0],
            rtol=0,
            atol=0,
        )


def test_a_zero_focus_weight_is_the_additive_model() -> None:
    """With `focus_film.weight` zero every offer is
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
    agent = _agent(3, focus_film=True)
    assert agent.focus_film is not None
    with torch.no_grad():
        agent.focus_film.weight.add_(1)
    host = agent.prepare(encode(decision_case("attack").observation, SPACE))
    mask = host.focus_mask.copy()
    mask[host.logit_positions[0, 1]] = 0
    prepared = device_input(agent, replace(host, focus_mask=mask))

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
    labels = (power < 3).long()
    agent = _agent(seed, bucketed_shapes=True, **fields)
    prepared = device_input(
        agent, _attacker(agent, decision_case("attack").observation, power, toughness)
    )
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
    historical layout's updates (measured 259 against 329) and the new layout
    in under 0.85 (measured 254)."""
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        seeds = range(3)
        historical = sum(_steps_to_fit(seed) for seed in seeds)
        focused = sum(_steps_to_fit(seed, focus_film=True) for seed in seeds)
        new = sum(
            _steps_to_fit(seed, stream_norm=True, focus_film=True) for seed in seeds
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert focused < 0.9 * historical, (focused, historical)
    assert new < 0.85 * historical, (new, historical)
