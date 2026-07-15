"""Compile the reviewed two-deck semantic source into checked-in IR."""

from __future__ import annotations

import argparse

from manabot.semantic.compiler import compile_paths, default_paths, write_or_check


def main() -> None:
    source, fixtures, ir_path, coverage_path = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if outputs are stale")
    parser.add_argument("--source", type=type(source), default=source)
    parser.add_argument("--fixtures", type=type(fixtures), default=fixtures)
    parser.add_argument("--ir-output", type=type(ir_path), default=ir_path)
    parser.add_argument("--coverage-output", type=type(coverage_path), default=coverage_path)
    args = parser.parse_args()

    ir, coverage = compile_paths(args.source, args.fixtures)
    write_or_check(args.ir_output, ir, check=args.check)
    write_or_check(args.coverage_output, coverage, check=args.check)


if __name__ == "__main__":
    main()
