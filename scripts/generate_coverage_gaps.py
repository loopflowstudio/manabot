"""Generate the checked curated-card coverage and kernel-change artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from manabot.semantic.coverage import (
    default_paths,
    enforce_coverage_gate,
    generate_paths,
    write_or_check,
)


def main() -> None:
    root, source, evidence, output = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if output is stale")
    parser.add_argument("--source", type=Path, default=source)
    parser.add_argument("--evidence", type=Path, default=evidence)
    parser.add_argument("--output", type=Path, default=output)
    args = parser.parse_args()

    artifact = generate_paths(args.source, args.evidence, root=root)
    write_or_check(args.output, artifact, check=args.check)
    enforce_coverage_gate(artifact)


if __name__ == "__main__":
    main()
