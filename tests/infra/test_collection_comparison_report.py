"""Synthetic receipts exercise paired conclusions without games or rentals."""

import hashlib
import json
from pathlib import Path

from etude.launch.records import Job, MachineTemplate, OnDemand
from etude.run.records import Attempt, Clock, RunDeclaration, RunFile, SourceKey
import pytest

from manabot.data import file_entry
from scripts.collection_comparison import SEED, SOURCES, Analysis, Preparation
from scripts.measure_collection_growth import Capture, Update
from scripts.report_collection_comparison import compare
from scripts.report_collection_growth import PHASES


def pair(
    tmp_path: Path, *, after_end: int = 500, before_growth: bool = True
) -> tuple[Path, Path]:
    prepared, data = tmp_path / "prepared", tmp_path / "data"
    prepared.mkdir()
    data.mkdir()
    for arm, source in SOURCES.items():
        regime = {"id": arm}
        job = Job(
            command="unused",
            max_hours=1,
            revision=source,
            config=json.dumps({"regime": regime}),
        )
        (prepared / f"{arm}.json").write_text(job.runs(SEED)[0].model_dump_json())
        growing = arm == "before" and before_growth
        end = after_end if arm == "after" else 500
        run_id = f"fixture-{arm}"
        identities = {
            key: "fixture"
            for key in (
                "world",
                "engine_source_sha256",
                "content_digest",
                "observation_abi_sha256",
                "action_abi_sha256",
                "deck_source_sha256",
                "architecture",
                "torch",
            )
        }
        identities.update(source_commit=source, cuda={"device": "NVIDIA L4"})
        receipt = {
            "id": run_id,
            "seed": SEED,
            "metrics_run": run_id,
            "status": "completed" if end == 500 else "paused",
            "regime": regime,
            "identities": identities,
            "recovery_seconds": 0,
            "host": {
                "cpu_model": "fixture",
                "logical_cpus": 6,
                "schedulable_cpus": 6,
                "cpu_quota": 6.0,
            },
            "stages": [
                {
                    "updates": end,
                    "status": "completed" if end == 500 else "paused",
                    "actual_device": "cuda",
                    "actual_threads": 1,
                    "games": end * 100,
                    "learner_transitions": end * 1000,
                }
            ],
            "game_errors": {
                "attempted": end * 100,
                "completed": end * 100,
                "crashed": 0,
                "abandoned": 0,
                "discarded_samples": 0,
                "replacement_samples": 0,
            },
        }
        (data / f"{arm}-training.json").write_text(json.dumps(receipt))
        config = json.dumps(
            {"training": {"run_id": run_id, "regime": regime, "identities": identities}}
        )
        capture = Capture(
            source="fixture://no-games",
            run=run_id,
            observed_at="2026-11-11T00:00:00Z",
            declaration=RunFile(
                experiment="fixture",
                namespace="local",
                declaration=RunDeclaration(
                    source_key=SourceKey(
                        namespace="local", kind="run", source_id=run_id
                    ),
                    clocks=(Clock(name="updates", kind="step", meaning="updates"),),
                    seed=SEED,
                    config_json=config,
                    config_sha256=hashlib.sha256(config.encode()).hexdigest(),
                ),
            ),
            attempts=[Attempt(writer="012345abcdef", started_ms=0, host="fixture")],
            versions={},
            blocks=[],
            updates=[
                Update(
                    update=i,
                    wall_ms=i * 1000,
                    metrics={
                        **{
                            key: 40 + (0.04 * i if growing else 0)
                            for key in PHASES.values()
                        },
                        "coordinates/rss_bytes": (10 + (0.01 * i if growing else 0))
                        * 1e9,
                        "coordinates/games": i * 100,
                        "coordinates/environment_decisions": i * 10000,
                        "rows": 1000,
                    },
                )
                for i in range(1, end + 1)
            ],
        )
        (data / f"{arm}.json").write_text(capture.model_dump_json())
    contract = Preparation(
        analysis=Analysis(),
        supply=OnDemand(name="fixture", template=MachineTemplate(provider="fixture")),
        files={p.name: file_entry(p) for p in prepared.iterdir()},
        source_files_equal=[],
    )
    (prepared / "preparation.json").write_text(contract.model_dump_json())
    return prepared, data


def test_pair_reports_work_normalized_slopes_and_flatness(tmp_path: Path) -> None:
    prepared, data = pair(tmp_path)
    compare(prepared, data, tmp_path / "report")
    result = json.loads((tmp_path / "report/comparison.json").read_text())
    assert result["verdict"] == "flattens through 500 updates"
    stats = result["arms"]["before"]["statistics"]
    assert stats["time/collection"]["slope_per_100_updates"] == pytest.approx(4)
    assert stats["time/collection/per_1000_rows"][
        "slope_per_100_updates"
    ] == pytest.approx(4)
    assert stats["rss_gb"]["slope_per_million_learner_rows"] == pytest.approx(10)
    assert stats["time/collection"]["late_p90"] > stats["time/collection"]["early_p90"]
    with pytest.raises(FileExistsError):
        compare(prepared, data, tmp_path / "report")


@pytest.mark.parametrize("end", [200, 480])
def test_paused_arm_never_counts_as_scientific_completion(
    tmp_path: Path, end: int
) -> None:
    prepared, data = pair(tmp_path, after_end=end)
    compare(prepared, data, tmp_path / "report")
    result = json.loads((tmp_path / "report/comparison.json").read_text())
    assert result["verdict"] == "inconclusive"
    assert any(
        "incomplete scientific target" in issue for issue in result["limitations"]
    )


def test_flat_baseline_cannot_prove_repair(tmp_path: Path) -> None:
    prepared, data = pair(tmp_path, before_growth=False)
    compare(prepared, data, tmp_path / "report")
    result = json.loads((tmp_path / "report/comparison.json").read_text())
    assert result["verdict"].startswith("inconclusive: baseline")


def test_host_difference_withholds_verdict_and_retains_facts(tmp_path: Path) -> None:
    prepared, data = pair(tmp_path)
    path = data / "after-training.json"
    receipt = json.loads(path.read_text())
    receipt["host"]["cpu_quota"] = 4
    path.write_text(json.dumps(receipt))
    compare(prepared, data, tmp_path / "report")
    result = json.loads((tmp_path / "report/comparison.json").read_text())
    assert result["verdict"] == "inconclusive"
    assert result["arms"]["after"]["receipt"]["host"]["cpu_quota"] == 4


def test_rejects_changed_preparation(tmp_path: Path) -> None:
    prepared, data = pair(tmp_path)
    path = prepared / "before.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="prepared input differs"):
        compare(prepared, data, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize(
    "fraction, expected", [(0.3, "partly flattens"), (1.0, "does not meet")]
)
def test_residual_growth_uses_predeclared_reduction_threshold(
    tmp_path: Path,
    fraction: float,
    expected: str,
) -> None:
    prepared, data = pair(tmp_path)
    path = data / "after.json"
    capture = Capture.model_validate_json(path.read_bytes())
    for row in capture.updates:
        for key in PHASES.values():
            row.metrics[key] += 0.04 * row.update * fraction
        row.metrics["coordinates/rss_bytes"] += 0.01 * row.update * fraction * 1e9
    path.write_text(capture.model_dump_json())
    compare(prepared, data, tmp_path / "report")
    result = json.loads((tmp_path / "report/comparison.json").read_text())
    assert result["verdict"].startswith(expected)


def test_sparse_metric_withholds_verdict(tmp_path: Path) -> None:
    prepared, data = pair(tmp_path)
    path = data / "after.json"
    capture = Capture.model_validate_json(path.read_bytes())
    del capture.updates[200].metrics["time/collection/rollout_recording"]
    path.write_text(capture.model_dump_json())
    compare(prepared, data, tmp_path / "report")
    result = json.loads((tmp_path / "report/comparison.json").read_text())
    assert result["verdict"] == "inconclusive"
    assert any("missing or nonfinite" in issue for issue in result["limitations"])


def test_capture_must_bind_the_delivered_scientific_run(tmp_path: Path) -> None:
    prepared, data = pair(tmp_path)
    path = data / "after-training.json"
    receipt = json.loads(path.read_text())
    receipt["metrics_run"] = "different-run"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="identities disagree"):
        compare(prepared, data, tmp_path / "report")
