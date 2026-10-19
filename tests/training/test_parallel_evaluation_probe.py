"""Equivalence compares decisions and identities, never merely final scores."""

import gzip
import json
from pathlib import Path

import pytest

from experiments.runners.parallel_evaluation_probe import (
    DEAL_SEED,
    cpu_allowance,
    panel_evidence,
)


def panel(
    directory: Path, *, latency: float, action: int = 1, complete: bool = True
) -> Path:
    root = directory / "transitions-0"
    trace = root / "deal-000/traces/games.gz"
    trace.parent.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    games: list[dict[str, object]] = []
    for leg in range(8):
        rows.append(
            {
                "deal_seed": DEAL_SEED,
                "leg": leg,
                "replay_passed": complete,
                "terminated": True,
                "failure": None,
                "score_a": 1,
                "game_seconds": latency,
                "latency": {"seconds": latency},
                "trace_path": "deal-000/traces/games.gz",
                "trace_sha256": str(latency),
                "trace_shard_sha256": str(latency),
                "game_trace_sha256": str(latency),
            }
        )
        games.append(
            {
                "deal_seed": DEAL_SEED,
                "leg": leg,
                "winner": 0,
                "game_trace_sha256": str(latency),
                "decisions": [
                    {
                        "command": {"offer": action},
                        "latency_seconds": latency,
                        "pre_state_digest": "before",
                        "post_state_digest": "after",
                    }
                ],
            }
        )
    trace.write_bytes(gzip.compress(b"\n".join(json.dumps(g).encode() for g in games)))
    result = {
        key: "same"
        for key in (
            "run_id",
            "regime_digest",
            "training_seed",
            "artifact",
            "coordinates",
            "protocol",
            "key",
            "candidate",
            "opponent",
        )
    }
    (root / "monitor.json").write_text(
        json.dumps(
            {
                **result,
                "status": "completed",
                "expected_games": 8,
                "rows": rows,
                "evaluation_seconds": latency * 8,
            }
        )
    )
    return directory


def test_timing_changes_do_not_hide_decision_changes(tmp_path: Path) -> None:
    serial = panel_evidence(panel(tmp_path / "serial", latency=1), "random")
    parallel = panel_evidence(panel(tmp_path / "parallel", latency=3), "random")
    changed = panel_evidence(panel(tmp_path / "changed", latency=3, action=2), "random")
    assert serial.decision_digest == parallel.decision_digest
    assert serial.result_sha256 != parallel.result_sha256
    # The score is identical in all three panels, but a different Command fails.
    assert serial.decision_digest != changed.decision_digest


def test_incomplete_replay_cannot_supply_equivalence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="complete replayed"):
        panel_evidence(
            panel(tmp_path / "incomplete", latency=1, complete=False), "random"
        )


@pytest.mark.parametrize("quota,affinity", [("300000 100000", 8), ("max 100000", 3)])
def test_quota_and_affinity_both_bound_concurrency(quota: str, affinity: int) -> None:
    with pytest.raises(ValueError, match="at least four"):
        cpu_allowance(quota, affinity)


def test_cpu_admission_retains_fractional_quota() -> None:
    assert cpu_allowance("1437500 100000", 120) == 14.375
    assert cpu_allowance("max 100000", 8) == 8
