"""The step-1 throughput probe drives real bench children and keeps their evidence."""

import json
from pathlib import Path

import pytest

from experiments import training_throughput as probe
from manabot.training.records import TrainingRun

RECIPE = (
    Path(__file__).parents[2] / "experiments" / "regimes" / "elemental-long-183-v1.json"
)


@pytest.mark.training
def test_cpu_probe_writes_receipts_and_samples_for_every_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(probe, "journal", lambda _: tmp_path / "journal")
    config = probe.Probe(
        recipe=str(RECIPE),
        device="cpu",
        streams=8,
        phases=(
            probe.Phase(name="solo", children=1, updates=2, first_seed=26001),
            probe.Phase(name="pack-2", children=2, updates=2, first_seed=26002),
        ),
    )

    results = probe.drive(config, output)

    assert [r.status for r in results] == ["completed", "completed"]
    for result in results:
        folder = output / "training-throughput" / result.phase.name
        assert json.loads((folder / "phase.json").read_text())["status"] == "completed"
        samples = [
            probe.Sample.model_validate_json(line)
            for line in (folder / "samples.jsonl").read_text().splitlines()
        ]
        assert samples
        assert result.peak_child_bytes and result.peak_child_bytes > 0
        for child in result.children:
            run = TrainingRun.model_validate_json(
                (folder / f"seed-{child.seed}" / "receipt.json").read_bytes()
            )
            assert run.status == "completed"
            assert run.stages[0].updates == 2
            assert run.stages[0].collection_timings
        pids = {c.pid for sample in samples for c in sample.children}
        assert len(pids) == result.phase.children


def test_phase_is_skipped_when_packed_peak_would_exceed_memory(tmp_path: Path) -> None:
    config = probe.Probe(recipe=str(RECIPE), device="cpu", memory_fraction=0.5)
    phase = probe.Phase(name="pack-4", children=4, updates=1, first_seed=1)
    machine = probe.psutil.virtual_memory().total

    result = probe.run_phase(phase, config, tmp_path, previous_peak=machine // 6)

    assert result.status == "skipped"
    assert not result.children
    assert (tmp_path / "pack-4" / "phase.json").exists()


def test_children_track_locally_not_into_the_launched_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ETUDE_RUN", "launched")
    monkeypatch.setenv("ETUDE_TRACK_DIR", "/track")

    environment = probe._child_environment()

    assert not any(key.startswith("ETUDE_") for key in environment)
