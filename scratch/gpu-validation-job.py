"""Frozen learner-shape validation with exact source-bundle admission."""

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from etude.launch.records import Job


def runs() -> Job:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    files: dict[str, str] = {}
    with tarfile.open(
        Path(".runs/gpu-validation/sources") / f"{commit}.tar"
    ) as archive:
        for entry in archive:
            if entry.isfile():
                source = archive.extractfile(entry)
                assert source is not None
                with source:
                    files[entry.name] = hashlib.file_digest(
                        source, "sha256"
                    ).hexdigest()
    return Job(
        name="fantasia-gpu-highend-h100-005",
        repo=str(Path.cwd()),
        revision="HEAD",
        command='timeout 900 bash scratch/gpu-validation-setup.sh && uv run --no-sync python scratch/gpu-validation.py --experiment highend --expected-gpu "NVIDIA H100 80GB HBM3" --root "$ETUDE_OUTPUT_DIR/evidence"',
        config=json.dumps(
            {
                "recipe": json.loads(
                    Path("scratch/cpu-probes/update-recipe.json").read_text()
                ),
                "source": {"commit": commit, "files": files},
            }
        ),
        max_hours=0.5,
        max_usd=Decimal("3.00"),
        max_attempts=1,
    )
