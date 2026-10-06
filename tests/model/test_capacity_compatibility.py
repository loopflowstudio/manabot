"""Frozen pre-change weights/receipts and default plan captured at fd7437df."""

import hashlib
import json
import os
from pathlib import Path
import platform

import torch

from experiments.runners.calibrate_training import capacity_plan
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentSpec
from manabot.model.agent import Agent
from manabot.model.architecture import architecture_receipt


def test_frozen_capacity_contract() -> None:
    frozen = json.loads(
        (Path(__file__).parent / "fixtures/capacity_base.json").read_text()
    )
    torch.set_num_threads(1)
    for row in frozen["models"]:
        torch.manual_seed(115)
        agent = Agent(
            ObservationSpace(),
            AgentSpec(attention_layers=row["depth"], value_aggregation=row["pooling"]),
        )
        digest = hashlib.sha256()
        for name, value in agent.state_dict().items():
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
        # QR initialization differs across backends and across Apple CPUs with the
        # same system/machine/torch triple, so hosted runners never match the
        # capture host; the receipt and plan checks below run everywhere.
        if not os.environ.get("CI") and frozen["weight_runtime"] == {
            "system": platform.system(),
            "machine": platform.machine(),
            "torch": torch.__version__,
        }:
            assert digest.hexdigest() == row["weights"]
        assert architecture_receipt(agent).model_dump(mode="json") == row["receipt"]
    plan = json.dumps(
        capacity_plan().model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    assert hashlib.sha256(plan.encode()).hexdigest() == frozen["plan_sha256"]
