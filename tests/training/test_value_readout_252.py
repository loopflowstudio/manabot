"""MTG-252 readout arms vary only the critic readout of the v10 recipe."""

import pytest

from experiments import value_readout_252
from manabot.recipes.builders import with_agent


def test_arms_change_only_the_value_readout() -> None:
    baseline = value_readout_252.base()
    for aggregation in value_readout_252.ARMS:
        regime = value_readout_252.arm(aggregation)
        assert regime.agent.value_aggregation == aggregation
        assert regime.stages == baseline.stages
        assert regime.observation == baseline.observation
    wider = with_agent(
        baseline,
        id="wider",
        agent=baseline.agent.model_copy(
            update={"value_aggregation": "value_token", "hidden_dim": 128}
        ),
    )
    with pytest.raises(ValueError, match="more than agent.value_aggregation"):
        value_readout_252.require_single_axis(baseline, wider)


def test_static_costs_and_initialization_sharing() -> None:
    costs = {cost.aggregation: cost for cost in value_readout_252.costs()}
    masked, token, readout = (costs[arm] for arm in value_readout_252.ARMS)
    # v10: two player objects, 128 cards and 64 permanents per seat.
    assert masked.objects == 2 + 2 * 128 + 2 * 64
    assert masked.added_parameters == 0 and masked.shares_initialization
    assert token.added_parameters == 64
    assert token.attention_pairs == 2 * (masked.objects + 1) ** 2
    # Query 64 + in-projection 3*64*64+3*64 + out-projection 64*64+64.
    assert readout.added_parameters == 16_704
    assert readout.attention_pairs == masked.attention_pairs + masked.objects
    assert readout.shares_initialization
    # The token's draw precedes the v7 encoders, shifting their initial values.
    assert not token.shares_initialization
