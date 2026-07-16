"""Generate or byte-check the W2-214 structural semantic kata artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from manabot.semantic.structural_katas import (
    CONTRACT_PATH,
    SOURCE_PATH,
    SUITE_PATH,
    artifact_bytes,
)


def _check(path: Path, expected: bytes) -> None:
    if not path.exists():
        raise SystemExit(f"missing generated artifact: {path}")
    actual = path.read_bytes()
    if actual != expected:
        raise SystemExit(f"generated artifact is stale: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and fail unless every checked artifact is byte-identical",
    )
    args = parser.parse_args()
    source, suite, contract = artifact_bytes()
    artifacts = (
        (SOURCE_PATH, source),
        (SUITE_PATH, suite),
        (CONTRACT_PATH, contract),
    )
    if args.check:
        for path, expected in artifacts:
            _check(path, expected)
        print("structural semantic kata artifacts are byte-identical")
        return
    for path, value in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        print(path)


if __name__ == "__main__":
    main()
