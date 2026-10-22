"""Frozen pre-change weights/receipts and default plan captured at fd7437df."""

import hashlib
import json
from pathlib import Path

import torch

from experiments.runners.calibrate_training import capacity_plan
from manabot.arena.models import canonical_sha256
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentSpec
from manabot.model.agent import Agent
from manabot.model.architecture import architecture_receipt


def test_frozen_capacity_contract() -> None:
    frozen = json.loads(
        (Path(__file__).parent / "fixtures/capacity_base.json").read_text()
    )
    torch.set_num_threads(1)
    # Frozen w4 evidence remains immutable; w6 must have a distinct receipt.
    for row in frozen["models"]:
        agent = Agent(
            ObservationSpace(),
            AgentSpec(attention_layers=row["depth"], value_aggregation=row["pooling"]),
        )
        assert architecture_receipt(agent).model_dump(mode="json") != row["receipt"]
    # Recipes follow the engine's current world; the plan was frozen in w4, so
    # compare it with the world (and the regime digests that cover it) put back.
    planned = capacity_plan().model_dump(mode="json")
    for recipe in planned["recipes"]:
        recipe["world"] = "w4"
    planned["protocol"]["regime_digests"] = [
        canonical_sha256(recipe) for recipe in planned["recipes"]
    ]
    plan = json.dumps(planned, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(plan.encode()).hexdigest() == frozen["plan_sha256"]
