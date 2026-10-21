"""GPU companion preserves learning and registered monitoring milestones."""

from pathlib import Path

from experiments.runners.baseline_gpu import MILESTONES, gpu_recipe
from experiments.runners.baseline_monitoring import DiagnosticMonitor
from experiments.runners.corrected_baseline import recipe


def test_gpu_changes_only_device(tmp_path: Path) -> None:
    gpu = gpu_recipe(25000, 24000)
    cpu = recipe(25000, 24000)
    gpu.stages[0].execution.device = "cpu"
    assert gpu == cpu
    assert MILESTONES == (
        1000,
        2000,
        3000,
        4000,
        5000,
        7500,
        10000,
        12500,
        15000,
        17500,
        20000,
        22500,
        25000,
    )
    monitor = DiagnosticMonitor(
        tmp_path / "run.json", tmp_path / "eval", updates=MILESTONES, seconds=7200
    )
    assert monitor.config.updates == MILESTONES
    assert monitor.config.seconds == 7200
