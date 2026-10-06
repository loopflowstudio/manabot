"""Cost admission and factorial contrasts; no scientific workload in tests."""

import json
from pathlib import Path

import pytest

from experiments.runners.pooling_filter import (
    ORDER,
    Calibration,
    CalibrationArm,
    recipes,
)
from experiments.runners.run_pooling_filter import followup_plan
from experiments.runners.run_value_screen import screen_plan
from experiments.runners.training_protocol import ResolvedStudy
from manabot.arena.models import canonical_sha256


def calibrated(seconds_per_arm: float = 60) -> Calibration:
    return Calibration(
        arms=tuple(
            CalibrationArm(
                recipe_id=r.id,
                run_path=f"/{r.id}/run.json",
                run_sha256="a" * 64,
                seconds=seconds_per_arm,
            )
            for r in recipes(40, 400)
        ),
        seconds=4 * seconds_per_arm + 10,
        source_commit="a" * 40,
        runtime_identities=screen_plan().runtime_identities,
    )


def test_admission_uses_slowest_arm_and_charges_calibration() -> None:
    fast = calibrated()
    assert fast.admitted_updates() == 800
    fields = fast.model_dump()
    fields["arms"][3]["seconds"] = 120
    fields["seconds"] = 310
    slower = Calibration.model_validate(fields)
    assert slower.admitted_updates() == 400
    assert slower.run_seconds() * 12 + slower.seconds < 21600
    fields["arms"][3]["seconds"] = 200
    fields["seconds"] = 390
    with pytest.raises(ValueError, match="minimum 400"):
        Calibration.model_validate(fields).admitted_updates()
    with pytest.raises(ValueError, match="all four"):
        Calibration.model_validate({**fast.model_dump(), "arms": fast.arms[:3]})


def test_followup_is_fixed_and_does_not_rewrite_original() -> None:
    original = Path("experiments/plans/value-token-screen.json").read_bytes()
    plan = followup_plan(calibrated())
    assert ResolvedStudy.model_validate_json(plan.model_dump_json()) == plan
    assert plan.protocol.training_seeds == (10621, 10622, 10623)
    assert len(plan.recipes) == 4
    assert all(sorted(row) == [0, 1, 2, 3] for row in ORDER)
    for change in ("learning_rate_scale", "advantage_quantile", "min_advantage"):
        fields = plan.model_dump()
        fields["recipes"][0]["stages"][0]["learning"][change] = 0.123
        fields["protocol"]["regime_digests"] = tuple(
            canonical_sha256(r) for r in fields["recipes"]
        )
        with pytest.raises(ValueError, match="preserve frozen"):
            ResolvedStudy.model_validate(fields)
    assert Path("experiments/plans/value-token-screen.json").read_bytes() == original


def test_factorial_pairing_and_incomplete_suppression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.runners import pooling_filter_analysis as analysis

    monkeypatch.setattr(analysis, "verify_saved_inputs", lambda out, data: None)
    cells: list[dict[str, object]] = []
    for arm, recipe in enumerate(recipes(40, 400)):
        for seed in (10621, 10622, 10623):
            for cutoff in range(2):
                cells.append(
                    dict(
                        a=recipe.id,
                        training_seed=seed,
                        cutoff=cutoff,
                        rows=[
                            dict(
                                deal_seed=deal,
                                score_a=(0.2, 0.3, 0.4, 0.7)[arm],
                                failure=None,
                                terminated=True,
                                truncated=False,
                                replay_passed=True,
                            )
                            for deal in range(961160, 961185)
                            for _ in range(4)
                        ],
                    )
                )
    path = tmp_path / "study.json"
    path.write_text(json.dumps(dict(status="completed", comparisons=cells, runs=[])))
    analysis.factorial_report(tmp_path)
    result = json.loads((tmp_path / "factorial.json").read_text())
    assert result["checkpoints"][1]["mean_effects"] == pytest.approx([0.1, 0.3, 0.2])
    before = (tmp_path / "factorial.json").read_bytes()
    analysis.factorial_report(tmp_path)
    assert (tmp_path / "factorial.json").read_bytes() == before
    path.write_text(json.dumps(dict(status="failed", comparisons=cells[:-1], runs=[])))
    analysis.factorial_report(tmp_path)
    assert (
        json.loads((tmp_path / "factorial.json").read_text())["status"] == "unavailable"
    )


def test_supervisor_timeout_kills_process_group_and_keeps_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import signal

    from experiments.runners import run_pooling_filter as supervisor

    killed: list[tuple[int, int]] = []

    class Process:
        pid = 12345

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(
        supervisor.subprocess, "Popen", lambda *args, **kwargs: Process()
    )
    monkeypatch.setattr(
        supervisor.os, "killpg", lambda pid, sig: killed.append((pid, sig))
    )
    with pytest.raises(TimeoutError, match="deadline"):
        supervisor._child(["--fixture"], tmp_path, -1)
    assert killed == [(12345, signal.SIGKILL)]
    assert json.loads((tmp_path / "active-child.json").read_text())["pid"] == 12345
