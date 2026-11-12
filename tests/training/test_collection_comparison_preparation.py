"""Preparation keeps the frozen scientific recipe and bounds the rental."""

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from etude.launch.records import Job
import pytest

from manabot.data import DataFile, file_entry
from manabot.experiment.job import TrainingJob
from manabot.infra.source import bundle_at
from manabot.training.execution import validate_regime
from manabot.training.regime import TrainingRegime
from scripts.collection_comparison import (
    FINALIZATION_SECONDS,
    RECIPE,
    RENTAL_SECONDS,
    SOURCES,
    STARTUP_SECONDS,
    TRAINING_SECONDS,
    prepared_runs,
)
import scripts.prepare_collection_comparison as preparation
from scripts.prepare_collection_comparison import _bound_rental, prepare

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def compatible_preparation_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic declaration admission only; never play, encode or train here.

    Frozen source digests stand in for local source reads. Output files still
    receive real hashes, and job bundles/archives still come from pinned Git.
    The real runtime's rejection is tested without this fixture below.
    """
    bundle = bundle_at(ROOT, SOURCES["before"])
    protocol = json.loads(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{bundle.commit}:experiments/protocols/tla-frontier-c5fba4-v1.json",
            ]
        )
    )

    def declared_regime(regime: TrainingRegime) -> TrainingRegime:
        return validate_regime(regime, require_current_world=False)

    def declared_source(path: Path) -> DataFile:
        if path.is_relative_to(ROOT):
            name = path.relative_to(ROOT).as_posix()
            if name in bundle.files:
                return DataFile(sha256=bundle.files[name], bytes=0)
        return file_entry(path)

    monkeypatch.setattr(preparation, "validate_regime", declared_regime)
    monkeypatch.setattr(
        preparation.managym, "world_identity", lambda: protocol["runtime_world"]
    )
    monkeypatch.setattr(preparation, "file_entry", declared_source)


def test_prepared_pair_preserves_recipe_sources_and_rental_bounds(
    tmp_path: Path,
    compatible_preparation_runtime: None,
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
    with pytest.raises(FileExistsError):
        prepare(ROOT, out)


@pytest.mark.parametrize("now, success", [(100, True), (STARTUP_SECONDS, False)])
def test_setup_uses_original_startup_cutoff(
    tmp_path: Path, now: int, success: bool
) -> None:
    # Exercise the shared shell wrapper without archives, native admission,
    # installing tools, sleeping or launching training.
    job = _bound_rental(Job(command="unused", setup="unused", max_hours=12))
    fake_date = tmp_path / "date"
    fake_date.write_text(
        '#!/bin/sh\nif [ "$1" = -d ]; then echo 42300; else echo "$FIXTURE_NOW"; fi\n'
    )
    fake_timeout = tmp_path / "timeout"
    fake_timeout.write_text('#!/bin/sh\nprintf "%s\\n" "$3"\n')
    fake_date.chmod(0o755)
    fake_timeout.chmod(0o755)
    result = subprocess.run(
        ["sh", "-c", job.setup or ""],
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "FIXTURE_NOW": str(now),
            "ETUDE_DEADLINE": "fixture",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert (result.returncode == 0) is success
    if success:
        assert result.stdout.strip() == "790s"


def test_preparation_rejects_current_runtime(tmp_path: Path) -> None:
    """Merged native changes cannot refresh admission of the frozen pair."""
    out = tmp_path / "rejected"
    with pytest.raises(
        ValueError,
        match="regime world differs|local native world|local native/recipe admission",
    ):
        prepare(ROOT, out)
    assert not (out / "local-admission.json").exists()
    assert not (out / "preparation.json").exists()


def test_preparation_rejects_changed_world(
    tmp_path: Path,
    compatible_preparation_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preparation.managym, "world_identity", lambda: {})
    out = tmp_path / "changed-world"
    with pytest.raises(ValueError, match="local native world or declared capacities"):
        prepare(ROOT, out)
    assert not (out / "preparation.json").exists()
