"""Injected clocks and spend exercise C1 without training or rented hosts."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.training_throughput_readings import (
    PHASES,
    Budget,
    Run,
    Update,
    _packing_comparisons,
    admit_cost,
    analyze,
    compare,
    load_run,
    precision_check,
    read_cell,
)
from experiments.training_throughput_sizes import C1_LEGS, leg_recipe, limits
from manabot.infra.identity import canonical_sha256
from manabot.training.records import TrainingRun
from scripts.diagnose_training_transfers import diagnose


def _run(seed: int, origin: float = 0, seconds: float = 10, count: int = 8) -> Run:
    return Run(
        str(seed),
        seed,
        origin,
        tuple(
            Update(
                n,
                origin + (n - 1) * seconds,
                origin + n * seconds,
                100,
                {key: 1.0 for key in PHASES.values()},
            )
            for n in range(2, count + 1)
        ),
        "source",
        "world",
        "host",
        "control",
    )


def test_overlap_keeps_whole_updates_and_reports_exact_ordinals() -> None:
    leg = C1_LEGS["c1-l4-w160l8-k2-float32-u8"]
    runs = [_run(26401), _run(26402, origin=15)]
    read = read_cell(leg, runs)
    assert read.status == "read"
    assert read.ordinals == {"26401": [3, 4, 5, 6, 7, 8], "26402": [2, 3, 4, 5, 6]}
    result = compare(_run(26401, seconds=5), read, runs)
    assert result.latency_gain == 1
    assert result.decision_gain == 1
    assert result.solo_ordinals == read.ordinals


def test_insufficient_overlap_does_not_become_a_gain_or_pool_retries() -> None:
    leg = C1_LEGS["c1-l4-w160l8-k2-float32-u8"]
    runs = [_run(26401), _run(26402, origin=65)]
    read = read_cell(leg, runs)
    assert read.status == "insufficient_overlap" and not read.seeds
    with pytest.raises(ValueError, match="admitted"):
        compare(_run(26401), read, runs)
    with pytest.raises(ValueError, match="exactly once"):
        read_cell(leg, runs + [_run(26401)])


def test_longer_cell_requires_matching_solo_ordinals() -> None:
    leg = C1_LEGS["c1-l4-w160l8-k2-float32-u12-overlap"]
    runs = [_run(26401, count=12), _run(26402, count=12)]
    read = read_cell(leg, runs)
    with pytest.raises(ValueError, match="matching ordinals"):
        compare(_run(26401), read, runs)
    assert compare(_run(26401, count=12), read, runs).latency_gain == 2
    with pytest.raises(ValueError, match="identities"):
        compare(replace(_run(26401, count=12), source="old"), read, runs)
    with pytest.raises(ValueError, match="identities"):
        read_cell(leg, [runs[0], replace(runs[1], host="other")])


@pytest.mark.parametrize("insufficient_overlap", [False, True])
def test_packing_uses_first_valid_attempt_and_matching_solo(
    insufficient_overlap: bool,
) -> None:
    prefix = "c1-l4-w160l8-"
    first = prefix + "k2-float32-u8"
    retry = prefix + "k2-float32-u12-overlap"
    solo = prefix + "k1-float32-u8"
    longer_solo = prefix + "k1-float32-u12-overlap"
    loaded = {
        first: [
            _run(26401, seconds=20),
            _run(26402, origin=145 if insufficient_overlap else 0, seconds=20),
        ],
        retry: [_run(26401, count=12), _run(26402, count=12)],
        solo: [_run(26401, seconds=5)],
        longer_solo: [_run(26401, seconds=5, count=12)],
    }
    readings = {name: read_cell(C1_LEGS[name], runs) for name, runs in loaded.items()}
    result = _packing_comparisons(loaded, readings)[first]
    assert result["cell"] == (retry if insufficient_overlap else first)
    assert result["solo"] == (longer_solo if insufficient_overlap else solo)


def test_cost_reserves_watchdog_cleanup_and_all_caps() -> None:
    leg = C1_LEGS["c1-l4-w160l8-k2-float32-u8"]
    limit = limits(leg, 37)
    budget = Budget(1.8, 3.9, 0, 0, 0)
    options = dict(
        rate=0.6, remaining_seconds=14400, line_remaining_usd=2.4, packed=2, updates=8
    )
    liability = admit_cost(budget, limit, **options)
    assert liability == (limit.max_hours + 300 / 3600) * 0.6
    for change in (
        {"committed": 8},
        {"prior_step3": 9.9, "prior_total": 12.0},
        {"prior_total": 49.9},
        {"gpu_hours": 14.9},
        {"updates": 1590},
    ):
        with pytest.raises(ValueError, match="budget/window"):
            admit_cost(replace(budget, **change), limit, **options)
    for change in ({"remaining_seconds": 100}, {"line_remaining_usd": 0.01}):
        with pytest.raises(ValueError, match="budget/window"):
            admit_cost(budget, limit, **(options | change))


def test_recording_diagnostic_uses_fixed_rows_without_rng_draws() -> None:

    before = torch.random.get_rng_state().clone()
    observation = {
        "actions_valid": np.tile(np.array([1, 1, 0, 0], dtype=np.float32), (256, 1)),
        "agent_player": np.arange(256, dtype=np.float32).reshape(256, 1),
    }
    result = diagnose("recording", observation)
    assert result["legal_actions"] == 512
    assert {
        "cold_copy",
        "resident_copy",
        "allocate_virtual",
        "pack_actions",
    } <= result.keys()
    assert torch.equal(before, torch.random.get_rng_state())


def test_c1_reader_reconciles_cells_and_rejects_legacy_or_changed_recipe(
    tmp_path: Path,
) -> None:

    root = tmp_path / "step-3-c1"
    run_dir = root / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    leg_name = "c1-l4-w160l8-k1-float32-u8"
    regime = leg_recipe(C1_LEGS[leg_name])
    receipt = TrainingRun(
        id="test-run",
        seed=26401,
        seed_streams={},
        regime=regime,
        regime_digest=canonical_sha256(regime.model_dump(mode="json")),
        identities={"source_commit": "frozen"},
        status="completed",
    )
    path = run_dir / "run.json"
    path.write_text(receipt.model_dump_json())
    (run_dir / "gpu-samples.csv").write_text("2026/11/08 12:00:00.000, 50, 1024\n")
    rows = {
        n: {
            "coordinates/training_seconds": n * 10,
            "coordinates/learner_transitions": n * 100,
            "coordinates/games": n * n,
            "coordinates/environment_decisions": n * n * 30,
            **{key: 1.0 for key in PHASES.values()},
        }
        for n in range(1, 9)
    }
    updates_path = run_dir / "updates.json"
    updates_path.write_text(json.dumps(rows))
    cells = {
        name: {
            "status": "not_needed",
            "reason": "conditional cell not selected",
            "billed_usd": 0,
        }
        for name in C1_LEGS
    }
    cells[leg_name] = {
        "status": "completed",
        "reason": "completed",
        "host": "rental",
        "runs": ["test-run"],
        "billed_usd": 0.01,
    }
    (root / "cells.json").write_text(json.dumps(cells))
    (root / "rentals.json").write_text(
        json.dumps(
            {
                "rental": {
                    "gpu": "l4",
                    "usd_per_hour": 0.6,
                    "billed_usd": 0.5,
                    "outstanding_usd": 0.1,
                    "hours": 1,
                    "bill_reference": "bill.json",
                    "deletion_reference": "deleted.json",
                }
            }
        )
    )
    result = analyze(root, "frozen")
    assert (
        result["billed_usd"] == 0.5
    )  # Includes setup/idle, not the allocated cell cost.
    assert result["outstanding_usd"] == 0.1
    assert result["readings"][leg_name]["ordinals"] == {"test-run": list(range(2, 9))}
    loaded = load_run(root, "test-run", C1_LEGS[leg_name], "rental", "frozen")
    assert [u.values["games"] for u in loaded.updates] == [3, 5, 7, 9, 11, 13, 15]
    assert all(u.values["decisions_per_game"] == 30 for u in loaded.updates)
    # Reject gaps and truncated endpoints before calculating adjacent intervals.
    for missing in (1, 4, 8):
        updates_path.write_text(
            json.dumps({n: row for n, row in rows.items() if n != missing})
        )
        with pytest.raises(ValueError, match="declared endpoint"):
            analyze(root, "frozen")
    updates_path.write_text(json.dumps(rows))
    invalid = {n: row | {"coordinates/games": 1} for n, row in rows.items()}
    updates_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="complete-game exposure"):
        analyze(root, "frozen")
    updates_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="frozen C1 source"):
        analyze(root, "legacy")
    receipt.regime.id = "elemental-throughput-w160l8-float32-v1"
    path.write_text(receipt.model_dump_json())
    with pytest.raises(ValueError, match="C1 hashed v6"):
        analyze(root, "frozen")
    receipt.regime.id = regime.id = "elemental-throughput-c1-w160l8-float32-u8-v1"
    receipt.regime.stages[0].streams = 128
    receipt.regime_digest = canonical_sha256(receipt.regime.model_dump(mode="json"))
    path.write_text(receipt.model_dump_json())
    with pytest.raises(ValueError, match="changed the declared"):
        analyze(root, "frozen")
    del cells[leg_name]
    (root / "cells.json").write_text(json.dumps(cells))
    with pytest.raises(ValueError, match="reconcile every"):
        analyze(root, "frozen")


def test_precision_health_uses_complete_matched_late_updates() -> None:
    metrics = (
        "policy_loss",
        "value_loss",
        "entropy",
        "collection_kl",
        "reference_kl",
        "numerical/clip_fraction",
        "rows",
        "games",
        "decisions_per_game",
    )
    loaded: dict[str, list[Run]] = {}
    for suffix, value in (
        ("float32-u12", 1),
        ("float32-repeat-u12", 2),
        ("tf32-u12", 4),
        ("bf16-u12", 3),
    ):
        run = _run(26401, count=12)
        run = replace(
            run,
            updates=tuple(
                replace(update, values={m: value for m in metrics} | {"rows": 100})
                for update in run.updates
            ),
        )
        loaded["c1-fast-w160l8-k1-" + suffix] = [run]
    result = precision_check(loaded)
    assert result["ordinals"] == list(range(5, 13))
    assert result["differences_from_float32"]["tf32"]["entropy"] == 3
    assert result["differences_from_float32"]["bf16"]["games"] == 2
    assert result["differences_from_float32"]["tf32"]["decisions_per_game"] == 3
    assert result["absolute_difference_over_repeat"]["tf32"]["entropy"] == 3
    assert result["absolute_difference_over_repeat"]["bf16"]["rows"] is None
    key = "c1-fast-w160l8-k1-bf16-u12"
    loaded[key][0] = replace(loaded[key][0], host="other")
    with pytest.raises(ValueError, match="identities"):
        precision_check(loaded)
    del loaded[key]
    assert precision_check(loaded) == {"status": "incomplete", "missing": ["bf16"]}
