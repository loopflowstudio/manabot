"""Relocation preserves evidence and rejects unsafe or corrupt paths."""

from pathlib import Path

import pytest

import manabot.remote.bundle as bundles
from manabot.remote.bundle import Bundle, BundleFile, make_bundle
from manabot.remote.plan import digest
from manabot.training.models import StageRecord, TrainingRegime, TrainingRun
from manabot.verify.store import VerifyStore
from tests.remote.test_compile import ROOT


def test_relocated_bytes_and_corruption(tmp_path: Path) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    artifact = producer / "policy.pt"
    artifact.write_bytes(b"immutable")
    manifest = make_bundle(producer)
    destination = tmp_path / "returned"
    producer.rename(destination)
    assert manifest.resolve(destination, str(artifact)).read_bytes() == b"immutable"
    (destination / "policy.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="digest"):
        manifest.verify(destination)


def test_traversal_and_symlink_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "private"
    artifact.write_bytes(b"x")
    root = tmp_path / "bundle"
    root.mkdir()
    for relative in ("../private", str(artifact), "link"):
        (root / "link").unlink(missing_ok=True)
        (root / "link").symlink_to(artifact)
        bundle = Bundle(
            files=(
                BundleFile(
                    producer_path="/original",
                    relative_path=relative,
                    size=1,
                    sha256=digest(b"x"),
                ),
            )
        )
        with pytest.raises(ValueError):
            bundle.verify(root)


@pytest.mark.parametrize(
    "relative", ["", ".", "dir/../file", "dir//file", "./file", "a b", "a\nb"]
)
def test_destination_rejects_unsafe_or_aliased_paths(
    tmp_path: Path, relative: str
) -> None:
    item = BundleFile(
        producer_path="/original", relative_path=relative, size=0, sha256=digest(b"")
    )
    with pytest.raises(ValueError, match="invalid bundle path"):
        item.destination(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["relative_path", "producer_path"])
def test_duplicate_manifest_paths_fail_before_download(field: str) -> None:
    first = BundleFile(
        producer_path="/first", relative_path="first", size=0, sha256=digest(b"")
    )
    second = BundleFile(
        producer_path="/second", relative_path="second", size=0, sha256=digest(b"")
    ).model_copy(update={field: getattr(first, field)})
    with pytest.raises(ValueError, match="duplicate bundle paths"):
        Bundle(files=(first, second))


@pytest.mark.parametrize("mismatch", [None, "path", "bytes", "sha256"])
def test_verified_bundle_must_match_run_artifact_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str | None
) -> None:
    recipe = TrainingRegime.model_validate_json(
        (ROOT / "experiments/regimes/direct-self-play.json").read_text()
    )
    policy = tmp_path / "raw.pt"
    policy.write_bytes(b"policy-loader-fixture")
    artifact = {
        "path": str(policy),
        "bytes": policy.stat().st_size,
        "sha256": digest(policy.read_bytes()),
    }
    if mismatch is not None:
        artifact[mismatch] = {"path": "/missing", "bytes": 0, "sha256": "0" * 64}[
            mismatch
        ]
    run = TrainingRun(
        id="bundle-fixture",
        regime=recipe,
        regime_digest=digest(recipe.model_dump_json().encode()),
        seed=197,
        seed_streams={},
        identities={},
        status="completed",
        stages=[StageRecord(id="train", artifacts={"raw": artifact})],
    )
    (tmp_path / "run").mkdir()
    (tmp_path / "run/run.json").write_text(run.model_dump_json())
    with VerifyStore(tmp_path / "training.sqlite") as store:
        store.save_training_run(run)
    bundle = make_bundle(tmp_path)
    loaded: list[str] = []
    monkeypatch.setattr(bundles, "load_checkpoint_agent", loaded.append)
    if mismatch is None:
        assert bundles.verify_training_bundle(tmp_path, bundle) == [policy]
        assert loaded == [str(policy)]
    else:
        with pytest.raises(ValueError, match="artifact"):
            bundles.verify_training_bundle(tmp_path, bundle)
        assert not loaded
    bundle.verify(tmp_path)
