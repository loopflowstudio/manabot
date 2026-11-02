"""The engine budget fails a measured slowdown or a changed digest, never a speedup."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/engine_budget.py"
BASELINE = ROOT / "managym/perf/baseline.json"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("engine_budget", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["engine_budget"] = module
    spec.loader.exec_module(module)
    return module


budget = _module()

BASE = {
    "runner": "ubuntu-24.04 GitHub-hosted, release",
    "tolerance": 0.10,
    "recorded": {"commit": "abc123def", "date": "2026-11-02", "run": "https://x", "reason": "test"},
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


def _write_results(
    directory: Path,
    *,
    scale: float = 1.0,
    search_digests: tuple[int, int, int] = (11, 22, 33),
    steps_digest: int = 44,
) -> tuple[Path, Path]:
    """Benchmark output lines where `scale` multiplies every duration."""
    search = directory / "bench_search.jsonl"
    lines = []
    for seed, worst, digest in zip((1, 2, 3), (8.0, 10.0, 9.0), search_digests):
        lines.append(
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
        )
    lines.append(json.dumps({"worst_overall_s": 10.0 * scale, "total_overall_s": 300.0 * scale}))
    search.write_text("\n".join(lines) + "\n")
    steps = directory / "bench_steps.jsonl"
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


def _compare(tmp_path: Path, **results: object) -> tuple[int, str]:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(BASE))
    search, steps = _write_results(tmp_path, **results)  # type: ignore[arg-type]
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "compare",
            "--baseline",
            str(baseline),
            "--search",
            str(search),
            "--steps",
            str(steps),
        ],
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout + completed.stderr


def test_within_tolerance_passes(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=1.05)
    assert code == 0, output
    assert "PASS" in output


def test_speedup_passes(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=0.5)
    assert code == 0, output


def test_twenty_percent_slowdown_fails(tmp_path: Path) -> None:
    """The controlled fixture: the same workload reported 20% slower everywhere."""
    code, output = _compare(tmp_path, scale=1.2)
    assert code == 1, output
    assert "| search worst reply | 10.000 s | 12.000 s | over 11.000 s (+10%) |" in output
    assert "| search total | 300.000 s | 360.000 s | over 330.000 s (+10%) |" in output
    assert "| steps throughput | 5000.0/s | 4166.7/s | under 4545.5/s (-10%) |" in output
    assert "FAIL: 3 check(s)" in output


def test_changed_search_digest_fails_even_when_faster(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=0.8, search_digests=(11, 23, 33))
    assert code == 1, output
    assert "| search seed 2 digest | 22 | 23 | differs |" in output
    assert "FAIL: 1 check(s)" in output


def test_changed_steps_digest_fails_even_when_faster(tmp_path: Path) -> None:
    code, output = _compare(tmp_path, scale=0.8, steps_digest=45)
    assert code == 1, output
    assert "| steps digest | 44 | 45 | differs |" in output


def test_record_writes_the_baseline_with_its_reason(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    search, steps = _write_results(tmp_path, scale=1.3)
    assert (
        budget.main(
            [
                "record",
                "--baseline",
                str(baseline),
                "--search",
                str(search),
                "--steps",
                str(steps),
                "--commit",
                "0123456789abcdef",
                "--date",
                "2026-11-03",
                "--run",
                "https://github.com/loopflowstudio/manabot/actions/runs/1",
                "--reason",
                "T6 adds Sagas to the curated decks",
                "--runner",
                "ubuntu-24.04 GitHub-hosted, release",
            ]
        )
        == 0
    )
    recorded = json.loads(baseline.read_text())
    assert recorded["recorded"]["reason"] == "T6 adds Sagas to the curated decks"
    assert recorded["tolerance"] == 0.10
    assert recorded["bench_search"]["worst_s"] == 13.0
    assert recorded["bench_search"]["total_s"] == 390.0
    assert recorded["bench_steps"]["steps_per_s"] == pytest.approx(5000.0 / 1.3, abs=0.1)
    # Recording again keeps the runner and tolerance unless moved deliberately.
    search, steps = _write_results(tmp_path, scale=1.0)
    budget.main(
        [
            "record",
            "--baseline",
            str(baseline),
            "--search",
            str(search),
            "--steps",
            str(steps),
            "--commit",
            "fedcba9876543210",
            "--date",
            "2026-11-04",
            "--run",
            "https://github.com/loopflowstudio/manabot/actions/runs/2",
            "--reason",
            "indexed effect records",
        ]
    )
    again = json.loads(baseline.read_text())
    assert again["runner"] == "ubuntu-24.04 GitHub-hosted, release"
    assert again["recorded"]["commit"] == "fedcba9876543210"
    assert again["bench_search"]["worst_s"] == 10.0


def test_committed_baseline_is_complete() -> None:
    """The committed file names its runner, commit, run and reason."""
    baseline = budget.load_baseline(BASELINE)
    assert baseline["runner"]
    assert baseline["tolerance"] == 0.10
    recorded = baseline["recorded"]
    assert len(recorded["commit"]) >= 9 and recorded["run"].startswith("https://")
    assert recorded["reason"]
    assert set(baseline["bench_search"]["seeds"]) == {"1", "2", "3"}
    assert baseline["bench_steps"]["games"] == 512
    assert baseline["bench_steps"]["steps_per_s"] > 0
