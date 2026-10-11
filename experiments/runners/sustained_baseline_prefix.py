"""Bounded source comparison for ETU-118, using each source's ordinary executor.

Run this same file with each frozen source on PYTHONPATH and one shared runtime.
Two linked two-update stages exercise the continuation boundary, with the exact
256-row ETU-105 recipe. Hashes cover batches, Adam, learner and sampling state;
this is a mechanism check, not a full 1,240-update reproduction or strength run.
"""

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from manabot.sim.net_opponent import NetOpponentTrainer, RolloutBatch
from manabot.training import execution
from manabot.training.models import AtaraxosMoveLearning, TrainingRegime
from manabot.verify.store import VerifyStore


def digest(value: Any) -> str:
    """Hash local Torch/native boundary state with type, shape and exact bytes."""
    result = hashlib.sha256()

    def add(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            result.update(repr((item.dtype.str, item.shape)).encode())
            result.update(item.tobytes())
        elif is_dataclass(item):
            add(asdict(item))
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(type(item).__name__.encode())
            for child in item:
                add(child)
        else:
            result.update(repr((type(item).__name__, item)).encode())
        result.update(b"\0")

    add(value)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--recovery", action="store_true")
    args = parser.parse_args()
    data = TrainingRegime.model_validate_json(args.recipe.read_text()).model_dump(
        mode="json"
    )
    data["wall_seconds"] = 240
    for stage in data["stages"]:
        stage["updates"] = 2
        stage["execution"]["wall_seconds"] = 240
    if args.recovery:
        data["recovery"] = {"checkpoint_updates": 1}
    regime = TrainingRegime.model_validate(data)
    original = execution.update_iteration
    rows: list[dict[str, str | int]] = []

    def measured(
        trainer: NetOpponentTrainer,
        batch: RolloutBatch,
        learning: AtaraxosMoveLearning,
        fraction: float,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> dict[str, Any]:
        row: dict[str, str | int] = {
            "iteration": kwargs["iteration"],
            "batch": digest(batch),
            "learner_before": digest(trainer.agent.state_dict()),
            "adam_before": digest(trainer.optimizer.state_dict()),
            "torch_before": digest(torch.get_rng_state()),
            "minibatch_before": digest(rng.bit_generator.state),
        }
        diagnostic = original(trainer, batch, learning, fraction, rng, **kwargs)
        row.update(
            learner_after=digest(trainer.agent.state_dict()),
            adam_after=digest(trainer.optimizer.state_dict()),
            torch_after=digest(torch.get_rng_state()),
            minibatch_after=digest(rng.bit_generator.state),
            diagnostic=digest(diagnostic),
        )
        rows.append(row)
        return diagnostic

    execution.update_iteration = measured
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with VerifyStore(args.out.with_suffix(".sqlite")) as store:
        options = {"checkpoint_updates": (1, 3)} if args.recovery else {}
        run = execution.execute_regime(
            regime, 11840, args.out, store, checkpoint_seconds=3600, **options
        )
    execution.atomic_json(
        args.out.with_suffix(".prefix.json"),
        {
            "status": run.status,
            "seconds": run.seconds,
            "identities": run.identities,
            "rows": rows,
        },
    )
    if run.status != "completed" or len(rows) != 4:
        raise RuntimeError(run.error or "prefix did not complete")


if __name__ == "__main__":
    main()
