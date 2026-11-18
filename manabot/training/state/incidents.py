"""Bounded private numerical incidents retained by the self-play loop.

These are debugging artifacts, never admitted policies or process-recovery
snapshots. Keep a separate directory per failure, at most 32 MiB of evidence and
512 MiB of model/Adam/EMA state. No environment variables or arbitrary locals.
The backend's `TrainingState` captures and serializes its framework tensors.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from manabot.model.world import checkpoint_source
from manabot.policy.protocol import BackendPolicy
from manabot.training.state.backend import NumericalIncident, TrainingState
from manabot.training.updates.backend import Learner

if TYPE_CHECKING:
    from manabot.training.loops.self_play import Counts
    from manabot.training.rollout.collection import Collector, RolloutBatch

EVIDENCE_BYTES = 32 * 1024**2
STATE_BYTES = 512 * 1024**2
MAX_ROWS = 16


def save_incident(
    directory: Path,
    error: NumericalIncident,
    learner: Learner,
    ema: BackendPolicy | None,
    counts: "Counts",
    collector: "Collector",
    batch: "RolloutBatch | None",
    rng: np.random.Generator,
    *,
    out: Path,
    settings_digest: str,
    backend: TrainingState,
    framework: dict[str, str | None],
) -> list[Path]:
    """Write unique terminal evidence; propagate capture errors to caller notes.

    RNG is captured at failure, not reconstructed at collection time. Exact
    offending rows and the full current learner/Adam are enough to inspect or
    repeat a failing loss; this is NOT a promise to replay an entire iteration.
    `framework` names the framework versions the source record carries.
    """
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    captured = backend.capture_incident(
        error,
        learner,
        collector,
        batch,
        ema,
        rng,
        evidence_bytes=EVIDENCE_BYTES,
        max_rows=MAX_ROWS,
    )
    evidence_path = directory / "tensors.pt"
    with evidence_path.open("wb") as output:
        backend.save(captured.evidence, output)
    size = captured.state_tensor_bytes
    paths = [evidence_path]
    metadata = {
        "schema": 2,
        "invariant": error.invariant,
        "health": error.health,
        "run": out.name,
        "recipe": settings_digest,
        "stage": "self-play",
        "iteration": counts.updates + 1,
        "completed_updates": counts.updates,
        "source": {**checkpoint_source(), **framework},
        "tensors": captured.shapes,
        "state_tensor_bytes": size,
        "state_limit_bytes": STATE_BYTES,
        "evidence_limit_bytes": EVIDENCE_BYTES,
        "full_batch_captured": captured.full_batch_captured,
        "sampling_contract": "masked-log-softmax-v1",
        "attempted_collection": collector.stats.to_dict(),
        "last_valid_exports": [
            str(path) for path in sorted((out / "policies").glob("*.safetensors"))
        ],
        "recovery": "diagnostic only; no engine journal or collection-time RNG",
    }
    if size <= STATE_BYTES:
        state_path = directory / "state.pt"
        with state_path.open("wb") as output:
            backend.save(captured.state, output)
        paths.append(state_path)
        metadata["state"] = "captured at failure (may itself be invalid)"
    else:
        metadata["state"] = "omitted: exceeds explicit state byte budget"
    metadata_path = directory / "incident.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    paths.append(metadata_path)
    return paths
