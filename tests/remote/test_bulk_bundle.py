"""Bulk transport retains manifest admission; archive names never own placement."""

import io
from pathlib import Path
import tarfile

import pytest

from manabot.remote.bundle import make_bundle, unpack_bundle


def test_bulk_return_preserves_bytes_and_rejects_unlisted_links(tmp_path: Path) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "a").write_bytes(b"first")
    (producer / "nested").mkdir()
    (producer / "nested/b").write_bytes(b"second")
    bundle = make_bundle(producer)
    archive = tmp_path / "evidence.tar"
    with tarfile.open(archive, "w") as packed:
        packed.add(producer, arcname=".")
    unpack_bundle(archive, tmp_path / "returned", bundle)
    assert (tmp_path / "returned/nested/b").read_bytes() == b"second"
    with tarfile.open(archive, "a") as packed:
        member = tarfile.TarInfo("../../escape")
        member.size = 4
        packed.addfile(member, io.BytesIO(b"oops"))
    with pytest.raises(ValueError, match="differs"):
        unpack_bundle(archive, tmp_path / "rejected", bundle)
    assert not (tmp_path.parent / "escape").exists()


def test_bulk_return_rejects_duplicate_and_missing_files(tmp_path: Path) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "a").write_bytes(b"a")
    bundle = make_bundle(producer)
    for duplicate in (False, True):
        archive = tmp_path / f"{duplicate}.tar"
        with tarfile.open(archive, "w") as packed:
            if duplicate:
                packed.add(producer / "a", arcname="./a")
                packed.add(producer / "a", arcname="./a")
        with pytest.raises(ValueError):
            unpack_bundle(archive, tmp_path / f"out-{duplicate}", bundle)
