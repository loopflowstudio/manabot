"""SDK source archives carry exact source and locally built native identities."""

import hashlib
import json
from pathlib import Path
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
    (tmp_path / "managym").mkdir()
    (tmp_path / "source.py").write_bytes(b"changed" if altered else b"source")
    native = (
        tmp_path / "managym" / ("_managym" + sysconfig.get_config_var("EXT_SUFFIX"))
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
