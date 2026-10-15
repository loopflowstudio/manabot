"""Compare retained TrainingRun tensors exactly, without inventing a tolerance.

Raw and EMA comparisons select model_state_dict, excluding run metadata. Adam
comparisons include all state and parameter groups. Every discrepancy is retained;
finite absolute differences describe it but never turn it into an exact match.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch


@dataclass(frozen=True)
class Difference:
    path: str
    kind: str
    shape: list[int] | None = None
    differing_elements: int | None = None
    max_absolute: float | None = None
    finite: bool | None = None


def compare(left: object, right: object, path: str = "") -> list[Difference]:
    if isinstance(left, torch.Tensor):
        if (
            not isinstance(right, torch.Tensor)
            or left.dtype != right.dtype
            or left.shape != right.shape
        ):
            return [Difference(path, "tensor structure")]
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        if torch.equal(left, right) and finite:
            return []
        maximum = (
            float((left.double() - right.double()).abs().max())
            if finite and left.numel()
            else None
        )
        return [
            Difference(
                path,
                "tensor values",
                list(left.shape),
                int(torch.count_nonzero(left != right)),
                maximum,
                finite,
            )
        ]
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return [Difference(path, "mapping keys")]
        return [
            d
            for key, value in left.items()
            for d in compare(value, right[key], f"{path}/{key}")
        ]
    if isinstance(left, (list, tuple)):
        if (
            type(left) is not type(right)
            or not isinstance(right, (list, tuple))
            or len(left) != len(right)
        ):
            return [Difference(path, "sequence structure")]
        return [
            d
            for index, (a, b) in enumerate(zip(left, right, strict=True))
            for d in compare(a, b, f"{path}/{index}")
        ]
    if not isinstance(left, (str, bool, int, float, type(None))):
        raise TypeError(f"unrecognized checkpoint value at {path}")
    return (
        []
        if type(left) is type(right) and left == right
        else [Difference(path, "scalar value")]
    )


def state(path: Path, variant: str) -> object:
    loaded: object = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError("checkpoint must be a mapping")
    return loaded if variant == "optimizer" else loaded["model_state_dict"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeated-streams", type=int, nargs="+", default=[64])
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("fresh comparison output required")
    comparisons: list[dict[str, object]] = []
    if not set(args.repeated_streams) <= {64, 512, 4096}:
        raise ValueError("unsupported repeated shape")
    pairs = [
        (streams, first, second)
        for streams in (64, 512, 4096)
        for first, second in (
            [
                ("reference-a", "reference-b"),
                ("cached-a", "cached-b"),
                ("reference-a", "cached-a"),
                ("reference-b", "cached-b"),
            ]
            if streams in args.repeated_streams
            else [("reference-a", "cached-a")]
        )
    ]
    for streams, first, second in pairs:
        base = f"w64-d1-loop-b{streams * 8}-s{streams}/run"
        left = args.root / f"shape-t8-s{streams}-{first}" / base
        right = args.root / f"shape-t8-s{streams}-{second}" / base
        item: dict[str, object] = {"streams": streams, "first": first, "second": second}
        statuses = [
            json.loads((p / "run.json").read_text())["status"]
            if (p / "run.json").exists()
            else "missing"
            for p in (left, right)
        ]
        item["run_statuses"] = statuses
        artifacts: list[dict[str, object]] = []
        for variant in ("initial-raw", "raw", "ema", "optimizer"):
            paths = [p / f"policy-0-{variant}.pt" for p in (left, right)]
            result: dict[str, object] = {"variant": variant}
            if not all(p.exists() for p in paths) or (
                variant != "initial-raw" and statuses != ["completed", "completed"]
            ):
                result.update(
                    status="unavailable",
                    missing=[str(p) for p in paths if not p.exists()],
                )
            else:
                differences = compare(
                    state(paths[0], variant), state(paths[1], variant)
                )
                result.update(
                    status="exact" if not differences else "different",
                    differences=[asdict(d) for d in differences],
                    files={
                        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in paths
                    },
                )
            artifacts.append(result)
        item["artifacts"] = artifacts
        comparisons.append(item)
        print(streams, first, second, [(a["variant"], a["status"]) for a in artifacts])
    args.out.write_text(
        json.dumps(
            {
                "comparison": "strict, no numerical tolerance",
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "comparisons": comparisons,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
