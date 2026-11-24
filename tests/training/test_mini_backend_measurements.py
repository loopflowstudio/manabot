"""Acceptance and pooling for the mini's retained baseline measurements."""

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.mini_baseline_backends import (
    ARMS,
    EXPERIMENT,
    rates,
    verdict,
    verify_live_run,
    write,
)


def test_only_finite_half_precision_tolerance_misses_can_be_timed(
    tmp_path: Path,
) -> None:
    arm = ARMS[2]
    evidence: dict[str, Any] = {
        "backend": "torch",
        "requested_device": "mps",
        "actual_device": "mps",
        "requested_bucketed_shapes": False,
        "bucketed_shapes": False,
        "threads": 4,
        "precision": "bfloat16",
        "observed_forward_dtype": "bfloat16",
        "step_checked": True,
        "checkpoint_sha256": "same",
        "execution_error": None,
        "minibatch_indices": list(range(8)),
        "collection_indices": list(range(8)),
        "failures": ["learner loss"],
        "minibatches": [
            {
                "index": 0,
                "loss_relative": 0.02,
                "loss_torch": 0.01,
                "loss_backend": 0.0102,
            }
        ],
    }
    write(tmp_path / "parity-attempt.json", {"status": "exit 1"})
    write(tmp_path / "parity.json", {"torch": evidence})
    gate, detail = verdict(tmp_path, "padded", arm, "same")
    assert gate == "approximate" and "2.00%" in detail
    assert verdict(tmp_path, "bucketed", arm, "same")[0] == "rejected"
    assert verdict(tmp_path, "padded", arm, "other")[0] == "rejected"
    evidence.update(precision="float32", observed_forward_dtype="float32")
    write(tmp_path / "parity.json", {"torch": evidence})
    assert verdict(tmp_path, "padded", ARMS[1], "same")[0] == "rejected"


def test_warmup_counts_only_toward_whole_invocation_wall_rate() -> None:
    rows = []
    cumulative = 0
    for index, (count, seconds) in enumerate([(1000, 50), (2, 1), (4, 1), (6, 4)], 1):
        cumulative += count
        rows.append(
            {
                "update": index,
                "rows": count,
                "coordinates/learner_transitions": cumulative,
                "execution/backend": "torch",
                "execution/device": "cpu",
                "execution/precision": "float32",
                "time/collection": seconds,
                "time/learning": seconds,
                "time/collection/observation_preparation": 0.2,
                "time/collection/policy_inference_and_action_selection": 0.2,
                "time/collection/environment_step_and_encoding": 0.1,
                "time/learning/targets": 0.2,
                "time/learning/minibatches": 0.3,
                "retained": 1,
                "optimizer_exposures": 2,
                "numerical/optimizer_steps": 1,
            }
        )
    result = rates(rows, 200, ARMS[0])
    assert result.collection == 2 and result.active == result.exposures == 1
    assert result.wall == 1012 / 200
    assert result.active_range == (0.75, 2)
    with pytest.raises(ValueError, match="four finite updates"):
        rates(rows[:-1], 200, ARMS[0])


def test_report_refuses_live_evidence_from_another_layout_or_run(
    tmp_path: Path,
) -> None:
    run = "mini-run"
    directory = tmp_path / "track" / EXPERIMENT / run
    directory.mkdir(parents=True)
    write(tmp_path / "worker.json", {"run": run})
    write(
        directory / "run.json",
        {
            "id": f"{EXPERIMENT}/{run}",
            "seed": 25201,
            "config": json.dumps(
                {
                    "device": "cpu",
                    "precision": "float32",
                    "threads": 4,
                    "model": {"hidden_dim": 64},
                    "bucketed_shapes": False,
                }
            ),
        },
    )
    verify_live_run(tmp_path, {"run": run}, 64, "padded", ARMS[0])
    with pytest.raises(ValueError, match="run settings"):
        verify_live_run(tmp_path, {"run": run}, 64, "bucketed", ARMS[0])
    with pytest.raises(ValueError, match="another run"):
        verify_live_run(tmp_path, {"run": "another"}, 64, "padded", ARMS[0])
