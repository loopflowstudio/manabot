"""SDK source archives carry exact source and locally built native identities."""

import hashlib
import json
from pathlib import Path
import subprocess
import sysconfig

import pytest

from manabot.infra import source as source_bundle


@pytest.mark.parametrize("altered", [False, True])
def test_archive_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, altered: bool
) -> None:
    monkeypatch.setattr(
        source_bundle, "__file__", str(tmp_path / "manabot/infra/source.py")
    )
    (tmp_path / "manabot" / "managym").mkdir(parents=True)
    (tmp_path / "source.py").write_bytes(b"changed" if altered else b"source")
    native = (
        tmp_path
        / "manabot"
        / "managym"
        / ("_managym" + sysconfig.get_config_var("EXT_SUFFIX"))
    )
    native.write_bytes(b"native fixture")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "source_bundle": {
                    "commit": "a" * 40,
                    "files": {"source.py": hashlib.sha256(b"source").hexdigest()},
                }
            }
        )
    )
    monkeypatch.setenv("ETUDE_CONFIG", str(config))
    source_bundle.source_commit.cache_clear()
    try:
        if altered:
            with pytest.raises(ValueError, match="archive differs"):
                source_bundle.bind_etude_source()
            assert not (tmp_path / ".manabot-source.json").exists()
        else:
            source_bundle.bind_etude_source()
            assert source_bundle.source_commit() == "a" * 40
            bound = source_bundle.SourceBundle.model_validate_json(
                (tmp_path / ".manabot-source.json").read_text()
            )
            assert (
                bound.files[str(native.relative_to(tmp_path))]
                == hashlib.sha256(b"native fixture").hexdigest()
            )
    finally:
        source_bundle.source_commit.cache_clear()


def test_bundle_reads_committed_bytes_not_the_working_tree(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init", "-q")
    (tmp_path / "a.py").write_bytes(b"committed")
    git("add", "a.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "fixture")
    (tmp_path / "a.py").write_bytes(b"edited later")
    bundle = source_bundle.bundle_at(tmp_path, "HEAD")
    assert len(bundle.commit) == 40
    assert bundle.files == {"a.py": hashlib.sha256(b"committed").hexdigest()}
