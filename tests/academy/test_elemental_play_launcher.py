"""The Elemental handoff pins a published update and refuses changed bytes."""

import os
from pathlib import Path
import subprocess

from etude.run.records import ArtifactRef
from etude.train.records import CheckpointRecord
import pytest

from scripts.play_elemental import latest_checkpoint, prepare_checkpoint


@pytest.mark.parametrize("prepare_status", [0, 7])
def test_wrapper_prepares_native_before_importing_checkpoint_admission(
    tmp_path: Path, prepare_status: int
) -> None:
    uv = tmp_path / "uv"
    uv.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$LAUNCH_LOG"\n'
        'case "$*" in *ensure_native*) exit "$PREPARE_STATUS";; esac\n'
    )
    uv.chmod(0o755)
    log = tmp_path / "launch.log"
    wrapper = Path(__file__).resolve().parents[2] / "scripts" / "play-elemental"
    result = subprocess.run(
        [str(wrapper), "--offline"],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "LAUNCH_LOG": str(log),
            "PREPARE_STATUS": str(prepare_status),
        },
        check=False,
        timeout=10,
    )
    commands = log.read_text().splitlines()
    assert "ensure_native" in commands[0]
    assert result.returncode == prepare_status
    if prepare_status:
        assert len(commands) == 1
    else:
        assert len(commands) == 2
        assert commands[1].endswith("scripts/play_elemental.py --offline")


def checkpoint(update: int, wall_ms: int = 1) -> CheckpointRecord:
    return CheckpointRecord(
        ordinal=1,
        artifact=ArtifactRef(uri="file:///unused.pt", sha256="a" * 64, size_bytes=4),
        at={"updates": update},
        wall_ms=wall_ms,
    )


def test_latest_uses_update_before_upload_time() -> None:
    newest = checkpoint(5950, 10)
    assert latest_checkpoint((newest, checkpoint(5900, 20))) == newest
    with pytest.raises(ValueError, match="no published checkpoint"):
        latest_checkpoint(())


def test_offline_refuses_changed_checkpoint(tmp_path: Path) -> None:
    record = checkpoint(5950)
    (tmp_path / "checkpoint.json").write_text(record.model_dump_json())
    (tmp_path / "MTG-183-update-5950-aaaaaaaaaaaa.pt").write_bytes(b"oops")
    with pytest.raises(ValueError, match="artifact bytes differ from receipt"):
        prepare_checkpoint(tmp_path, "r2://unused", "unused", offline=True)


def test_offline_never_downloads_missing_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.json").write_text(checkpoint(5950).model_dump_json())
    with pytest.raises(ValueError, match="pinned checkpoint is missing"):
        prepare_checkpoint(tmp_path, "r2://unused", "unused", offline=True)


def test_offline_without_a_pin_explains_how_to_prepare(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No pinned checkpoint; run once"):
        prepare_checkpoint(tmp_path, "r2://unused", "unused", offline=True)
