"""Export the Game-owned testing-house-v1 control schema."""

from __future__ import annotations

import json
from pathlib import Path

from etude.testing_house_protocol import testing_house_schema


def main() -> None:
    destination = (
        Path(__file__).parents[1] / "protocol" / "testing-house-v1.schema.json"
    )
    destination.write_text(
        json.dumps(testing_house_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
