import json
from pathlib import Path

import pytest

from experiments.runners import run_int7_value_target_comparison as runner


def test_verify_only_is_byte_preserving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "result"
    out_dir.mkdir()
    (out_dir / "artifact.json").write_text('{"fixed": true}\n')
    (out_dir / "manifest.json").write_text('{"manifest": true}\n')
    before = runner._tree_snapshot(out_dir)
    monkeypatch.setattr(
        runner,
        "verify_output",
        lambda path: {"state": "verified", "path": str(path), "no_generation": True},
    )

    result = runner.verify_only(out_dir)

    assert result["no_generation"] is True
    assert runner._tree_snapshot(out_dir) == before


def test_verify_only_rejects_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "result"
    out_dir.mkdir()
    (out_dir / "artifact.json").write_text('{"fixed": true}\n')
    (out_dir / "manifest.json").write_text('{"manifest": true}\n')

    def generating_verifier(path: Path) -> dict[str, object]:
        (path / "generated.json").write_text("{}\n")
        return {"state": "verified"}

    monkeypatch.setattr(runner, "verify_output", generating_verifier)

    with pytest.raises(runner.Int7Error, match="changed output bytes"):
        runner.verify_only(out_dir)


def test_manifest_digest_detects_tamper() -> None:
    manifest = {"schema_version": 1, "decision": "fixed"}
    stored = {**manifest, "manifest_sha256": runner.manifest_digest(manifest)}

    assert stored["manifest_sha256"] == runner.manifest_digest(stored)
    stored["decision"] = "tampered"
    assert stored["manifest_sha256"] != runner.manifest_digest(stored)


def test_preregistration_identity_stays_frozen_after_later_result_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    contract = repo_root / "experiments/contracts/frozen.json"
    contract.parent.mkdir(parents=True)
    contract.write_text('{"frozen": true}\n')
    frozen_commit = "0" * 40
    later_result_commit = "1" * 40
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> object:
        calls.append(command)
        if command[1] == "rev-parse":
            assert command[-1] == f"{frozen_commit}^{{commit}}"
            return runner.subprocess.CompletedProcess(
                command, 0, stdout=f"{frozen_commit}\n"
            )
        if command[1] == "show":
            assert command[2] == (f"{frozen_commit}:experiments/contracts/frozen.json")
            return runner.subprocess.CompletedProcess(
                command, 0, stdout=contract.read_bytes()
            )
        if command[1] == "log":
            return runner.subprocess.CompletedProcess(
                command, 0, stdout=f"{later_result_commit}\n"
            )
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(runner, "REPO_ROOT", repo_root)
    monkeypatch.setattr(runner, "FROZEN_PREREGISTRATION_COMMIT", frozen_commit)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner._preregistration_commit(contract) == frozen_commit
    assert all(command[1] != "log" for command in calls)


def test_cli_separates_run_and_verify_only() -> None:
    verify = runner.parse_args(["--out-dir", "result", "--verify-only"])
    assert verify.verify_only is True
    with pytest.raises(SystemExit):
        runner.parse_args(["--out-dir", "result"])
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--out-dir",
                "result",
                "--verify-only",
                "--input-manifest",
                "input.json",
            ]
        )


def test_tree_receipts_close_every_non_manifest_file(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/evidence.json").write_text(json.dumps({"ok": True}) + "\n")
    (tmp_path / "manifest.json").write_text("{}\n")

    receipts = runner._tree_receipts(tmp_path)

    assert [row["path"] for row in receipts] == ["nested/evidence.json"]
