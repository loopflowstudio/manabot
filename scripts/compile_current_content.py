"""Export typed Rust cards and generate checked-in native/learning projections."""

import argparse
import json
from pathlib import Path
import subprocess

from manabot.semantic.compiler import write_or_check
from manabot.semantic.content import CONTENT_ROOT, REGENERATE_COMMAND, generate


def export_cards(*, check: bool) -> None:
    """Compile the authoring source before checking any downstream projection."""
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--jobs",
            "2",
            "--manifest-path",
            "managym/Cargo.toml",
            "--example",
            "export_cards",
        ],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    catalog = json.loads(result.stdout)
    write_or_check(
        CONTENT_ROOT / "generated" / "cards.json",
        catalog,
        check=check,
        regenerate_command=REGENERATE_COMMAND,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    export_cards(check=args.check)
    generate(check=args.check)


if __name__ == "__main__":
    main()
