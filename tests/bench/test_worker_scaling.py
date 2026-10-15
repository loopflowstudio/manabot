"""Exercise actual worker barriers and transport, without asserting speedups."""

import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from experiments.runners.pipeline_benchmark import capture
from experiments.runners.run_training_regimes import smoke_recipe
from experiments.runners.worker_inference import Exchange


@pytest.mark.parametrize("fresh_capture", [False, True])
def test_persistent_workers_preserve_fixed_work(tmp_path: Path, fresh_capture: bool) -> None:
    torch.set_num_threads(1)
    tape_dir = tmp_path / "tape"
    tape_dir.mkdir()
    tape = capture(smoke_recipe("direct-self-play"), tape_dir, 2, 8, 10831)
    if fresh_capture:
        changed = tape.model_copy(update={"batch_sha256": "incompatible"})
        (tape_dir / "tape.json").write_text(changed.model_dump_json())
    out = tmp_path / "workers"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.runners.worker_scaling",
            "--tape",
            str(tape_dir),
            "--out",
            str(out),
            "--actors",
            "4",
            "--seconds",
            "10",
            *(["--capture-replay"] if fresh_capture else []),
        ],
        capture_output=True,
        text=True,
        timeout=12,
    )
    assert result.returncode == 0, (
        result.stdout + result.stderr + (out / "stderr.log").read_text()
    )
    assert json.loads((out / "attempt.json").read_text())["status"] == "completed"
    if fresh_capture:
        portability = json.loads((out / "portability.json").read_text())
        assert portability["status"] == "failed"
        assert '"batch_sha256"' in portability["error"]
        assert '"buffers_sha256"' not in portability["error"]
    for active in (1, 2, 4):
        collected = json.loads((out / f"r0-w{active}.collect.json").read_text())
        verified = json.loads((out / f"r0-w{active}.verify.json").read_text())
        assert sum(w["count"] for w in collected["workers"]) == 4
        assert sum(w["count"] > 0 for w in collected["workers"]) == active
        actors = [a for w in verified["workers"] for a in w["actors"]]
        assert len(actors) == 4
        assert all(a["batch_sha256"] == tape.batch_sha256 for a in actors)
        assert sum(a["demand"]["learner_transitions"] for a in actors) == 64


def test_exchange_slot_is_released_only_by_matching_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = {"actions_valid": torch.ones(2, 3)}
    outputs = (torch.ones(2, 3), torch.ones(2, 1))
    server = Exchange.create(tmp_path / "slot", obs, outputs)
    client = Exchange(tmp_path / "slot")
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(
        sys, "stdin", io.StringIO('{"kind":"inference_done","sequence":1}\n')
    )
    server.inputs["actions_valid"][:2] = 1

    def policy(values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return values["actions_valid"] * 2, torch.ones(2, 1) * 3

    server.serve(2, 1, policy)
    actual = client.client(0)(obs)
    assert torch.equal(actual[0], outputs[0] * 2)
    assert torch.equal(actual[1], outputs[1] * 3)
    assert json.loads(stdout.getvalue()) == {
        "kind": "inference",
        "worker": 0,
        "sequence": 1,
        "rows": 2,
    }
    with pytest.raises(ValueError, match="sequence"):
        server.serve(2, 1, policy)
    monkeypatch.setattr(
        sys, "stdin", io.StringIO('{"kind":"inference_done","sequence":1}\n')
    )
    with pytest.raises(ValueError, match="acknowledgment"):
        client.client(0)(obs)
