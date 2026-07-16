"""Integration coverage for durable, fail-closed teacher datagen."""

import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "experiments/runners/run_distill_datagen.py"


def _run(out_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--games",
            "2",
            "--workers",
            "1",
            "--sims",
            "1",
            "--games-per-shard",
            "1",
            "--seed",
            "41",
            "--out-dir",
            str(out_dir),
            *extra,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_datagen_resumes_incremental_shards_and_rejects_stale_contract(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "dataset"
    first = _run(out_dir)
    assert first.returncode == 0, first.stderr

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["progress"] == {
        "decisions_completed": manifest["decisions"],
        "games_completed": 2,
        "shards_completed": 2,
        "shards_total": 2,
    }
    first_shard = out_dir / "shard_00000.npz"
    first_summary = out_dir / "shard_00000.json"
    first_bytes = first_shard.read_bytes()

    # Model an abrupt stop after the first parent progress update. The first
    # shard and sidecar are durable; the second task never completed.
    (out_dir / "shard_00001.npz").unlink()
    (out_dir / "shard_00001.json").unlink()
    manifest["status"] = "running"
    manifest["shards"] = [json.loads(first_summary.read_text())]
    manifest["progress"] = {
        "decisions_completed": manifest["shards"][0]["decisions"],
        "games_completed": 1,
        "shards_completed": 1,
        "shards_total": 2,
    }
    manifest_path.write_text(json.dumps(manifest))

    resumed = _run(out_dir, "--resume")
    assert resumed.returncode == 0, resumed.stderr
    assert "resume: 1/2 shards durable; 1 pending" in resumed.stdout
    assert first_shard.read_bytes() == first_bytes
    resumed_manifest = json.loads(manifest_path.read_text())
    assert resumed_manifest["status"] == "completed"
    assert resumed_manifest["progress"]["games_completed"] == 2

    stale = _run(out_dir, "--resume", "--seed", "42")
    assert stale.returncode != 0
    assert "run fingerprint mismatch" in stale.stderr
