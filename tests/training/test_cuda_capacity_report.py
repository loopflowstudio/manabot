"""Portable capacity evidence renders without private models or another rental."""

from pathlib import Path
import shutil

from experiments.runners.cuda_capacity_report import render_report


def test_compact_evidence_regenerates_stable_report(tmp_path: Path) -> None:
    evidence = (
        Path(__file__).resolve().parents[2]
        / "experiments/data/cuda-capacity/capacity-report.json"
    )
    source = tmp_path / "capacity-report.json"
    shutil.copyfile(evidence, source)
    report = render_report(source)
    original = report.read_bytes(), (tmp_path / "hardware-throughput.svg").read_bytes()
    assert "ETU-123" in report.read_text()
    assert "TimeoutError" in report.read_text()
    render_report(source)
    assert (
        report.read_bytes(),
        (tmp_path / "hardware-throughput.svg").read_bytes(),
    ) == original
