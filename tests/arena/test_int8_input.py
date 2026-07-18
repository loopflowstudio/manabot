import json
from pathlib import Path
import shutil

import pytest

import manabot.arena.int8_input as int8_input

ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = (
    ROOT
    / "experiments/data/int-8-retained-int-4-smoke-v1/sha256"
    / "13868767846b7004f140cfade3652909347bdbb6708b69cb8c10b36ec2756eb0"
)


def test_retained_input_passes_exact_current_loaders() -> None:
    receipt = int8_input.verify_retained_input(INPUT_ROOT / "input-manifest.json")

    assert receipt["status"] == "compatible"
    assert receipt["dataset"] == {
        "rows": 507,
        "games": 4,
        "finite": True,
        "legal": True,
    }
    assert len(receipt["checkpoints"]) == 4
    assert all(row["deterministic_repeated_bytes"] for row in receipt["checkpoints"])


def test_retained_input_tamper_fails_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / "retained"
    shutil.copytree(INPUT_ROOT, copied)
    shard = copied / "payload/dataset/shard_000.npz"
    shard.write_bytes(shard.read_bytes() + b"tamper")

    def loader_must_not_run(paths: list[str | Path]) -> dict[str, object]:
        raise AssertionError(f"loader ran before the byte gate: {paths}")

    monkeypatch.setattr(int8_input, "load_shards", loader_must_not_run)
    failure = tmp_path / "failure.json"
    with pytest.raises(int8_input.RetainedInputError, match="payload_leaf_identity"):
        int8_input.verify_retained_input(
            copied / "input-manifest.json", failure_receipt=failure
        )

    evidence = json.loads(failure.read_text())
    assert evidence["status"] == "input_incompatible"
    assert evidence["boundary"] == "payload_leaf_identity"
    assert evidence["expected"]["bytes"] + len(b"tamper") == evidence["actual"]["bytes"]
