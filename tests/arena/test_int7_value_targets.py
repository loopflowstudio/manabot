import json
from pathlib import Path

import numpy as np
import pytest

from experiments.runners import run_int7_value_target_comparison as runner
from manabot.arena.int7_value_targets import (
    RESOURCE_CAPS,
    CumulativeResourceLedger,
    ResourceCapExceeded,
    phase_indices,
    verify_resource_ledger,
)
from manabot.sim.distill import load_shards

ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = (
    ROOT
    / "experiments/data/int-8-retained-int-4-smoke-v1/sha256"
    / "13868767846b7004f140cfade3652909347bdbb6708b69cb8c10b36ec2756eb0"
)


def test_retained_phase_strata_are_decoded_without_new_metadata() -> None:
    dataset = load_shards(sorted((INPUT_ROOT / "payload/dataset").glob("shard_*.npz")))

    phases = phase_indices(dataset)

    assert {name: len(rows) for name, rows in phases.items()} == {
        "beginning": 35,
        "precombat_main": 283,
        "combat": 123,
        "postcombat_main": 63,
        "ending": 3,
    }
    assert np.array_equal(
        np.sort(np.concatenate(list(phases.values()))), np.arange(507)
    )


def test_resource_ledger_is_chained_and_requires_worker_preflights(
    tmp_path: Path,
) -> None:
    ledger = CumulativeResourceLedger(tmp_path, started=0.0, caps=RESOURCE_CAPS)
    ledger.started = __import__("time").perf_counter()
    for worker in range(1, 5):
        ledger.check(
            "arena",
            projected_wall_seconds=1.0,
            projected_workers=4,
            worker_launch=worker,
        )
    ledger.finish("arena", elapsed_seconds=0.01, workers=4)
    ledger.complete({"games": 544})

    receipt = verify_resource_ledger(ledger.path, RESOURCE_CAPS)

    assert receipt["events"] == 6
    events = ledger.path.read_text().splitlines()
    tampered = json.loads(events[1])
    tampered["payload"]["stage"] = "changed"
    events[1] = json.dumps(tampered, sort_keys=True)
    ledger.path.write_text("\n".join(events) + "\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_resource_ledger(ledger.path, RESOURCE_CAPS)


def test_resource_cap_fails_closed_before_launch(tmp_path: Path) -> None:
    ledger = CumulativeResourceLedger(tmp_path, started=0.0, caps=RESOURCE_CAPS)
    ledger.started = __import__("time").perf_counter()

    with pytest.raises(ResourceCapExceeded) as caught:
        ledger.check(
            "arena",
            projected_wall_seconds=RESOURCE_CAPS["wall_hours"] * 3600 + 1,
            projected_workers=4,
            worker_launch=1,
        )

    assert caught.value.evidence["status"] == "resource_cap_exceeded"
    assert "resource_cap_exceeded" in ledger.path.read_text()


def test_registration_keeps_neutral_variant_profile_only() -> None:
    checkpoint = (
        INPUT_ROOT / "payload/training/visit_policy_value-seed-197-b97a4796a6cbcad0.pt"
    )
    receipt = {
        "arm": "visit_terminal",
        "seed": 197,
        "checkpoint_sha256": runner.file_sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "parameter_count": 102722,
    }
    runtime = runner.arena_runtime_fingerprints()

    learned = runner._registration(
        receipt, runtime, value_mode="learned", profile_only=False
    )
    neutral = runner._registration(
        receipt, runtime, value_mode="neutral", profile_only=True
    )

    assert learned.value_mode == "learned"
    assert learned.profile_only is False
    assert neutral.value_mode == "neutral"
    assert neutral.profile_only is True
    with pytest.raises(ValueError, match="joint gameplay candidates"):
        runner._registration(receipt, runtime, value_mode="neutral", profile_only=False)
