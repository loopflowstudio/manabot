"""A launched source archive names its commit through Etude, without git."""

from pathlib import Path

import pytest

from manabot.infra import source as source_bundle


def test_launched_archive_uses_etudes_commit_without_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        source_bundle, "__file__", str(tmp_path / "manabot/infra/source.py")
    )
    monkeypatch.setenv("ETUDE_SOURCE_DIR", str(tmp_path))
    monkeypatch.setenv("ETUDE_COMMIT", "b" * 40)
    source_bundle.source_commit.cache_clear()
    source_bundle.source_dirty.cache_clear()
    try:
        assert source_bundle.source_commit() == "b" * 40
        assert source_bundle.source_dirty() is None
    finally:
        source_bundle.source_commit.cache_clear()
        source_bundle.source_dirty.cache_clear()
