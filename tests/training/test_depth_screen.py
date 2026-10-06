"""Depth screen admission and cost calibration fixtures; no training."""

import pytest

from experiments.runners import depth_screen as depth
from experiments.runners.run_history_input import history_plan
from experiments.runners.training_protocol import ResolvedStudy
from manabot.arena.models import canonical_sha256
from tests.training.test_history_input import calibration


def test_depth_controls_and_timing_only_allocation() -> None:
    values = depth.recipes()
    controls = []
    for recipe in values:
        data = recipe.model_dump(mode="json")
        data.pop("id")
        data["agent"].pop("attention_layers")
        controls.append(data)
    assert controls[0] == controls[1]
    assert values[0].agent.value_kind == "scalar"
    assert values[0].agent.value_aggregation == "value_token"
    assert not values[0].agent.recent_events
    assert calibration(2, depth).admitted_updates() == 1400
    assert calibration(4, depth).admitted_updates() == 700
    with pytest.raises(ValueError, match="minimum 400"):
        calibration(8, depth).admitted_updates()
    plan = history_plan(calibration(spec=depth), spec=depth)
    assert ResolvedStudy.model_validate_json(plan.model_dump_json()) == plan
    assert plan.allocation_seconds == 28800
    assert plan.protocol.training_seeds == depth.SEEDS
    assert plan.protocol.evaluation_variants == ("raw",)
    assert len(plan.protocol.anchor_deals) * 4 == 100


@pytest.mark.parametrize(
    "field",
    [
        "hidden_dim",
        "value_kind",
        "recent_events",
        "filter_scope",
        "updates",
        "input",
        "seed",
        "budget",
    ],
)
def test_depth_rejects_other_treatments_even_with_new_digests(field: str) -> None:
    data = history_plan(calibration(spec=depth), spec=depth).model_dump(mode="json")
    arm = data["recipes"][1]
    if field == "hidden_dim":
        arm["agent"][field] = 128
    elif field == "value_kind":
        arm["agent"][field] = "categorical"
    elif field == "recent_events":
        arm["agent"][field] = True
    elif field == "filter_scope":
        arm["stages"][0]["learning"][field] = "actor_only"
    elif field == "updates":
        arm["stages"][0][field] += 1
    elif field == "input":
        data["input_bindings"][1]["input_schema_sha256"] = "f" * 64
    elif field == "seed":
        data["protocol"]["training_seeds"][0] += 100
    else:
        data["allocation_seconds"] += 1
    data["protocol"]["regime_digests"] = [canonical_sha256(r) for r in data["recipes"]]
    with pytest.raises(ValueError):
        ResolvedStudy.model_validate(data)
