"""Frozen first GPU validation allocation; explicit HEAD excludes unrelated scratch."""

from decimal import Decimal
from pathlib import Path

from etude.launch.records import Job


def runs() -> Job:
    return Job(
        name="fantasia-gpu-benchmark-l4-002",
        repo=str(Path.cwd()),
        revision="HEAD",
        command='timeout 900 bash scratch/gpu-validation-setup.sh && uv run --no-sync python scratch/gpu-validation.py --root "$ETUDE_OUTPUT_DIR/evidence"',
        config=Path("scratch/cpu-probes/update-recipe.json").read_text(),
        max_hours=0.5,
        max_usd=Decimal("1.50"),
        max_attempts=1,
    )
