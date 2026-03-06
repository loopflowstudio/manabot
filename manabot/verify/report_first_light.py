"""Generate a markdown report for one recorded first-light run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .first_light import default_report_path, write_report
from .store import VerifyStore


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=".runs/verify.sqlite")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output) if args.output else default_report_path(args.run_id)

    with VerifyStore(args.db) as store:
        markdown, written_path, _ = write_report(
            store,
            run_id=args.run_id,
            output_path=output,
            report_kind="summary",
        )

    if written_path is not None:
        print(f"Wrote {written_path}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
