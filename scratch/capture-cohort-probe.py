"""Capture and audit one matching-shape tape under the existing command supervisor."""

import hashlib
from pathlib import Path

import torch

from experiments.runners.pipeline_benchmark import capture, replay
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRegime


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path("scratch/accelerator-probes/cohort-t8-s64")
    root.mkdir(exist_ok=False)
    source = Path(
        "scratch/gpu-probes/l4-004/evidence/shape-t8-s64-reference-a/w64-d1-loop-b512-s64/regime.json"
    )
    recipe = TrainingRegime.model_validate_json(source.read_text())
    capture(recipe, root, 64, 8, 10349)
    audited = replay(root, verify_inputs=True)
    samples = [
        replay(root, verify_inputs=False, request_timings=True) for _ in range(3)
    ]
    atomic_json(
        root / "replay.json",
        {
            "tape_sha256": hashlib.sha256(
                (root / "tape.json").read_bytes()
            ).hexdigest(),
            "audited": audited.model_dump(mode="json"),
            "samples": [s.model_dump(mode="json") for s in samples],
            "recipe_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "capture_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "limits": "Untrained CPU tape; no claim of exact CUDA trajectories or changing-policy traffic.",
        },
    )
    print(
        "Audited and three timed replays passed:",
        [(s.seconds, s.demand.learner_transitions) for s in samples],
    )


if __name__ == "__main__":
    main()
