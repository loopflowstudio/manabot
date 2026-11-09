"""Require the C1 two-update CUDA worker's admitted, reloadable raw endpoint."""

import json
import os
from pathlib import Path

import torch

from manabot.infra.artifacts import verify_file
from manabot.model.checkpoint import load_checkpoint
from manabot.training.records import TrainingRun
from manabot.training.regime import TrainSelfPlay


def check(output: Path) -> dict[str, object]:
    receipt = TrainingRun.model_validate_json(
        (output / "training/run.json").read_bytes()
    )
    stage = receipt.stages[0]
    recipe = receipt.regime.stages[0]
    if not (
        receipt.status == "completed"
        and receipt.seed == 26400
        and isinstance(recipe, TrainSelfPlay)
        and recipe.streams == 256
        and recipe.execution.precision == "bf16"
        and stage.updates == 2
        and stage.actual_device == "cuda"
        and stage.optimizer_updates > 0
        and stage.numerical_failure is None
    ):
        raise ValueError(
            "C1 admission worker did not complete two healthy CUDA BF16 updates"
        )
    reference = stage.artifacts["raw"]
    path = output / "training" / Path(reference.path).name
    verify_file(path, reference.sha256, reference.bytes)
    agent, space = load_checkpoint(str(path))
    if space.encoder.config.policy_input_version != 6 or agent.config.catalog_records(
        6
    ):
        raise ValueError("C1 admission endpoint is not hashed input v6")
    if not all(torch.isfinite(p).all() for p in agent.parameters()):
        raise ValueError("C1 admission endpoint contains nonfinite weights")
    return {
        "status": "passed",
        "checkpoint_sha256": reference.sha256,
        "updates": stage.updates,
        "optimizer_updates": stage.optimizer_updates,
        "effective_records": "hashed",
        "seed": receipt.seed,
    }


if __name__ == "__main__":
    output = Path(os.environ["ETUDE_OUTPUT_DIR"])
    result = check(output)
    (output / "cuda-admission.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
