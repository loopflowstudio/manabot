"""Etude's model-loading entry point; only this module chooses Torch and device.

The returned forward is deterministic and returns masked logits in input order.
Etude owns its lifetime, residency, request batching and process allocation.
"""

from collections.abc import Callable
import os
from typing import cast

from etude.checkpoints import CheckpointRef
import numpy as np

from manabot.policy.batching import concatenate
from manabot.policy.prepare import Prepared


def load(checkpoint: CheckpointRef) -> Callable[[list[object]], list[object]]:
    import torch

    from manabot.backends.torch.arrays import on_device
    from manabot.backends.torch.checkpoint import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    model, _ = load_checkpoint(str(checkpoint.path))
    model.to(device).eval()
    with torch.inference_mode():
        model.cache_programs()

    def forward(inputs: list[object]) -> list[object]:
        if not inputs:
            return []
        if any(not isinstance(row, Prepared) or row.batch != 1 for row in inputs):
            raise TypeError("Manabot inference takes one host Prepared row per input")
        rows = cast(list[Prepared[np.ndarray]], inputs)
        batch = concatenate(rows)
        with torch.inference_mode():
            logits, _ = model(on_device(batch, device))
            values = logits.float().cpu().numpy()
            values[~batch.legal] = -np.inf
        return [row.copy() for row in values]

    return forward
