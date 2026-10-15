"""Compare scoped preparation with ordinary inputs through the same TrainingRun.

The reference changes only preparation to the original CPU-to-device dictionary
transfer. Checkpoint witnesses are computed after training timing completes and
include raw/EMA parameters and Adam state, not serialization-specific file bytes.
"""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.runners.cuda_performance import Cell, child
from manabot.model.inference import PolicyInference


def _ordinary_inputs(
    scope: PolicyInference, obs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {key: tensor.to(scope.device) for key, tensor in obs.items()}


@contextmanager
def reference_inference(enabled: bool) -> Iterator[None]:
    if enabled:
        with patch.object(PolicyInference, "prepare", _ordinary_inputs):
            yield
    else:
        yield


def state_digest(value: object) -> str:
    """Canonical tensor/optimizer payload hash, independent of torch.save IDs."""
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(
                json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)]).encode()
            )
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
            digest.update(b"end")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for element in item:
                visit(element)
            digest.update(b"end")
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(json.dumps([type(item).__name__, item]).encode())
        else:
            raise TypeError(
                f"unsupported checkpoint witness type: {type(item).__name__}"
            )

    visit(value)
    return digest.hexdigest()


def checkpoint_witnesses(directory: Path) -> dict[str, str]:
    """Hash exports, leaving separately typed recovery envelopes to recovery tests."""
    witnesses: dict[str, str] = {}
    for path in sorted(directory.glob("*.pt")):
        if not path.name.endswith(("-raw.pt", "-ema.pt", "-optimizer.pt")):
            continue
        # These are the experiment's own ordinary TrainingRun exports.
        payload: object = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("training export must be a dictionary")
        if "model_state_dict" in payload:
            payload = payload["model_state_dict"]
        witnesses[path.name] = state_digest(payload)
    for suffix in ("-raw.pt", "-ema.pt", "-optimizer.pt"):
        if not any(name.endswith(suffix) for name in witnesses):
            raise ValueError(f"training produced no {suffix} witness")
    return witnesses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cell", type=Cell.model_validate_json, required=True)
    parser.add_argument("--reference-policy", action="store_true")
    args = parser.parse_args()
    if args.cell.kind != "loop":
        parser.error("preparation comparison requires an ordinary training loop")
    with reference_inference(args.reference_policy):
        child(args.root, args.cell)
    out = args.root / args.cell.name
    witnesses = checkpoint_witnesses(out / "run")
    (out / "policy-witness.json").write_text(json.dumps(witnesses, indent=2) + "\n")


if __name__ == "__main__":
    main()
