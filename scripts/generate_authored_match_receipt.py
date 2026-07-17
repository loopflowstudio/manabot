"""Generate or check the deterministic authored-match authority receipt.

Run from the repository root with:

    uv run --extra dev python scripts/generate_authored_match_receipt.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from etude.authored_match_receipt import (
    DEFAULT_RECEIPT_PATH,
    write_or_check_authored_match_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if receipt is stale")
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT_PATH)
    args = parser.parse_args()
    write_or_check_authored_match_receipt(args.output, check=args.check)


if __name__ == "__main__":
    main()
