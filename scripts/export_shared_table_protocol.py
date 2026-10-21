"""Export the Game-owned shared-table control schema (frozen identifier `testing-house-v1`)."""

from __future__ import annotations

import json
from pathlib import Path

from academy.shared_table_protocol import shared_table_schema


def main() -> None:
    destination = (
        Path(__file__).parents[1] / "protocol" / "testing-house-v1.schema.json"
    )
    destination.write_text(
        json.dumps(shared_table_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
