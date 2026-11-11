"""Preparation keeps the frozen scientific recipe and bounds the rental."""

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from manabot.experiment.job import TrainingJob
from scripts.collection_comparison import (
    FINALIZATION_SECONDS,
    RECIPE,
    RENTAL_SECONDS,
    SOURCES,
    STARTUP_SECONDS,
    TRAINING_SECONDS,
    prepared_runs,
)
from scripts.prepare_collection_comparison import prepare

ROOT = Path(__file__).resolve().parents[2]


def test_prepared_pair_preserves_recipe_sources_and_rental_bounds(
    tmp_path: Path,
) -> None:
    out = tmp_path / "prepared"
    prepare(ROOT, out)
    contract, runs = prepared_runs(out)
    assert contract.supply.setup_seconds == 0
    assert contract.supply.startup_seconds == 900
    assert contract.supply.delivery_seconds == 600
    assert contract.supply.max_machines == contract.supply.slots == 1
    assert contract.supply.max_usd_per_hour == Decimal("0.60")
    for arm, run in runs.items():
        assert run.job.max_hours * 3600 == RENTAL_SECONDS
        assert RENTAL_SECONDS + 300 == 12 * 3600
        assert run.job.max_attempts == 1
        assert run.job.max_usd == Decimal("7.20")
        spec = TrainingJob.model_validate_json(run.job.config or "")
        original = json.loads(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"{SOURCES[arm]}:{RECIPE}"]
            )
        )
        original.update(
            id=f"collection-growth-281-{arm}", wall_seconds=TRAINING_SECONDS
        )
        original["stages"][0]["updates"] = 500
        original["stages"][0]["execution"]["wall_seconds"] = TRAINING_SECONDS
        assert spec.regime.model_dump(mode="json") == type(spec.regime).model_validate(
            original
        ).model_dump(mode="json")
        assert spec.finalization_seconds == FINALIZATION_SECONDS
        assert spec.validate_numerics
        assert spec.source_bundle is not None
        with tarfile.open(out / f"{arm}-source.tar.gz", "r:gz") as archive:
            hashes: dict[str, str] = {}
            for member in archive:
                if member.isfile():
                    stream = archive.extractfile(member)
                    assert stream is not None
                    hashes[member.name] = hashlib.sha256(stream.read()).hexdigest()
        assert hashes == spec.source_bundle.files
        # Full bootstrap + setup cutoff is derived from rental origin. A
        # controlled date/timeout fixture exercises that shell admission without
        # installing tools, sleeping or launching training.
        bin_dir = tmp_path / arm
        bin_dir.mkdir()
        fake_date = bin_dir / "date"
        fake_date.write_text(
            '#!/bin/sh\nif [ "$1" = -d ]; then echo 42300; else echo "$FIXTURE_NOW"; fi\n'
        )
        fake_timeout = bin_dir / "timeout"
        fake_timeout.write_text('#!/bin/sh\nprintf "%s\\n" "$3"\n')
        fake_date.chmod(0o755)
        fake_timeout.chmod(0o755)
        for now, success in ((100, True), (STARTUP_SECONDS, False)):
            result = subprocess.run(
                ["sh", "-c", run.job.setup or ""],
                env={
                    "PATH": f"{bin_dir}:/usr/bin:/bin",
                    "FIXTURE_NOW": str(now),
                    "ETUDE_DEADLINE": "fixture",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            assert (result.returncode == 0) is success
            if success:
                assert result.stdout.strip() == "790s"
    with pytest.raises(FileExistsError):
        prepare(ROOT, out)
