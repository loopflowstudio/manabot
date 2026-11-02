"""The engine budget fails a same-runner slowdown or a changed digest, never a speedup."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/engine_budget.py"
BASELINE = ROOT / "managym/perf/baseline.json"

BASE = {
    "tolerance": 0.10,
    "reference": {
        "commit": "abc123def456",
        "date": "2026-11-02",
        "run": "https://x",
        "runner": "ubuntu-24.04 GitHub-hosted, AMD EPYC 7763 (4 vCPU), release",
        "reason": "test",
    },
    "bench_search": {
        "worst_s": 10.0,
        "total_s": 300.0,
        "seeds": {
            "1": {"worst_s": 8.0, "total_s": 100.0, "digest": 11},
            "2": {"worst_s": 10.0, "total_s": 100.0, "digest": 22},
            "3": {"worst_s": 9.0, "total_s": 100.0, "digest": 33},
        },
    },
    "bench_steps": {"steps_per_s": 5000.0, "steps": 99923, "games": 512, "digest": 44},
}


def _run(*args: str | Path) -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True
    )
    return completed.returncode, completed.stdout + completed.stderr


def _write_results(
    directory: Path,
    name: str,
    *,
    scale: float = 1.0,
    search_digests: tuple[int, int, int] = (11, 22, 33),
    steps_digest: int = 44,
) -> tuple[Path, Path]:
    """Benchmark output lines where `scale` multiplies every duration."""
    search = directory / f"{name}_search.jsonl"
    lines = [
        json.dumps(
            {
                "seed": seed,
                "decisions": 40,
                "worst_s": round(worst * scale, 3),
                "median_s": 2.0,
                "total_s": round(100.0 * scale, 3),
                "digest": digest,
            }
        )
        for seed, worst, digest in zip((1, 2, 3), (8.0, 10.0, 9.0), search_digests)
    ]
    lines.append(
        json.dumps({"worst_overall_s": 10.0 * scale, "total_overall_s": 300.0 * scale})
    )
    search.write_text("\n".join(lines) + "\n")
    steps = directory / f"{name}_steps.jsonl"
    steps.write_text(
        json.dumps(
            {
                "games": 512,
                "capped": 0,
                "steps": 99923,
                "elapsed_s": 20.0 * scale,
                "steps_per_s": round(5000.0 / scale, 1),
                "digest": steps_digest,
            }
        )
        + "\n"
    )
    return search, steps


def _compare(
    tmp_path: Path,
    *,
    reference_scale: float = 1.4,
    reference_digests: tuple[int, int, int] = (11, 22, 33),
    scale: float = 1.0,
    search_digests: tuple[int, int, int] = (11, 22, 33),
    steps_digest: int = 44,
) -> tuple[int, str]:
    """A slower runner than the recording (reference at 1.4x) hosts both runs."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(BASE))
    reference_search, reference_steps = _write_results(
        tmp_path, "reference", scale=reference_scale, search_digests=reference_digests
    )
    search, steps = _write_results(
        tmp_path,
        "candidate",
        scale=scale,
        search_digests=search_digests,
        steps_digest=steps_digest,
    )
    return _run(
        "compare",
        "--baseline", baseline,
        "--reference-search", reference_search,
        "--reference-steps", reference_steps,
        "--search", search,
        "--steps", steps,
    )  # fmt: skip


def _write_artifact(directory: Path, *, commit: str, runner: str, scale: float) -> Path:
    """What `gh run download --name engine-budget` leaves: both outputs and measurement.json."""
    directory.mkdir()
    _write_results(directory, "bench", scale=scale)
    (directory / "measurement.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "date": "2026-11-03",
                "run": "https://github.com/loopflowstudio/manabot/actions/runs/1",
                "runner": runner,
            }
        )
    )
    return directory


def test_within_tolerance_on_a_slower_runner_passes(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=1.4 * 1.05)
    assert code == 0, output
    assert "PASS" in output
    assert "| search worst reply | 14.000 s | 14.700 s | ok |" in output


def test_speedup_passes(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=0.7)
    assert code == 0, output


def test_twenty_percent_slowdown_fails(tmp_path: Path) -> None:
    """The controlled fixture: the candidate reported 20% slower than the reference."""
    code, output = _compare(tmp_path, scale=1.4 * 1.2)
    assert code == 1, output
    assert (
        "| search worst reply | 14.000 s | 16.800 s | over 15.400 s (+10%) |" in output
    )
    assert "| search total | 420.000 s | 504.000 s | over 462.000 s (+10%) |" in output
    assert (
        "| steps throughput | 3571.4/s | 2976.2/s | under 3246.7/s (-10%) |" in output
    )
    assert "FAIL: 3 check(s)" in output
    assert "record --reason" in output


def test_changed_search_digest_fails_even_when_faster(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=0.8, search_digests=(11, 23, 33))
    assert code == 1, output
    assert "| candidate search seed 2 digest | 22 | 23 | differs |" in output
    assert "FAIL: 1 check(s)" in output


def test_changed_steps_digest_fails_even_when_faster(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=0.8, steps_digest=45)
    assert code == 1, output
    assert "| candidate steps digest | 44 | 45 | differs |" in output


def test_reference_not_matching_the_file_fails_the_file(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=1.4, reference_digests=(11, 22, 34))
    assert code == 1, output
    assert (
        "| baseline file | abc123def | digests differ | re-record from the reference commit |"
        in output
    )
    assert "| reference search seed 3 digest | 33 | 34 | differs |" in output


def test_record_adopts_a_downloaded_artifact_with_its_reason(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    runner = "GitHub-hosted ubuntu-24.04, AMD EPYC 7763 (4 vCPU), release"
    artifact = _write_artifact(
        tmp_path / "first", commit="0123456789abcdef", runner=runner, scale=1.3
    )
    code, output = _run(
        "record", "--baseline", baseline, "--artifact", artifact,
        "--reason", "T6 adds Sagas to the curated decks",
    )  # fmt: skip
    assert code == 0, output
    recorded = json.loads(baseline.read_text())
    assert recorded["reference"] == {
        "commit": "0123456789abcdef",
        "date": "2026-11-03",
        "run": "https://github.com/loopflowstudio/manabot/actions/runs/1",
        "runner": runner,
        "reason": "T6 adds Sagas to the curated decks",
    }
    assert recorded["tolerance"] == 0.10
    assert recorded["bench_search"]["worst_s"] == 13.0
    assert recorded["bench_search"]["total_s"] == 390.0
    assert recorded["bench_steps"]["steps_per_s"] == pytest.approx(
        5000.0 / 1.3, abs=0.1
    )
    # Recording again keeps the tolerance; only an edit of the file moves it.
    artifact = _write_artifact(
        tmp_path / "second",
        commit="fedcba9876543210",
        runner="GitHub-hosted ubuntu-24.04, Intel Xeon 6973P-C (4 vCPU), release",
        scale=1.0,
    )
    code, output = _run(
        "record", "--baseline", baseline, "--artifact", artifact,
        "--reason", "indexed effect records",
    )  # fmt: skip
    assert code == 0, output
    again = json.loads(baseline.read_text())
    assert again["tolerance"] == 0.10
    assert again["reference"]["commit"] == "fedcba9876543210"
    assert again["bench_search"]["worst_s"] == 10.0


def test_record_refuses_a_directory_without_a_measurement(tmp_path: Path) -> None:
    """Hand-assembled output is not a recording: the job's measurement.json names the run."""
    baseline = tmp_path / "baseline.json"
    _write_results(tmp_path, "bench")
    code, output = _run(
        "record", "--baseline", baseline, "--artifact", tmp_path, "--reason", "x"
    )
    assert code == 1, output
    assert "not an Engine Budget artifact" in output
    assert "gh run download <run-id> --name engine-budget" in output
    assert not baseline.exists()


def test_committed_baseline_is_complete() -> None:
    """The committed file names a reachable reference, its runner, run and reason."""
    baseline = json.loads(BASELINE.read_text())
    assert baseline["tolerance"] == 0.10
    reference = baseline["reference"]
    assert len(reference["commit"]) == 40 and reference["run"].startswith("https://")
    assert reference["runner"] and reference["reason"]
    assert set(baseline["bench_search"]["seeds"]) == {"1", "2", "3"}
    assert baseline["bench_steps"]["games"] == 512
    assert baseline["bench_steps"]["steps_per_s"] > 0
